"""Turns a result DataFrame into a short natural-language business insight.

If OPENAI_API_KEY is configured, uses a real LLM call for richer phrasing.
Otherwise (default), falls back to a deterministic template engine so the
platform runs fully offline with no external API dependency or cost —
this fallback is always correct and is what a live demo should rely on.
"""
import pandas as pd

from ..config import OPENAI_API_KEY, OPENAI_MODEL


def _template_insight(intent: str, df: pd.DataFrame, meta: dict) -> str:
    if df.empty:
        return "No data matched this question — try widening the time window or removing a filter."

    if intent == "revenue_summary":
        top = df.iloc[0]
        total = df["total_revenue"].sum()
        return (
            f"Across the selected window, total revenue was ${total:,.0f}. "
            f"{top['region']} led with ${top['total_revenue']:,.0f} "
            f"({top['total_revenue'] / total * 100:.0f}% of the total) from {int(top['order_count'])} orders."
        )

    if intent == "top_customers":
        top = df.iloc[0]
        return (
            f"{top['name']} ({top['segment']}, {top['region']}) is the top customer at "
            f"${top['total_revenue']:,.0f} in lifetime revenue. The top {len(df)} customers "
            f"account for ${df['total_revenue'].sum():,.0f} combined."
        )

    if intent == "product_performance":
        top = df.iloc[0]
        return (
            f"'{top['name']}' ({top['category']}) is the top performer with "
            f"{int(top['units_sold'])} units sold and ${top['total_revenue']:,.0f} in revenue."
        )

    if intent == "low_stock":
        risky = df[df["at_risk_of_stockout"]]
        if risky.empty:
            return "No products are currently projected to stock out in the next 7 days."
        names = ", ".join(risky["name"].head(3).tolist())
        return (
            f"{len(risky)} product(s) are projected to run out within 7 days based on demand "
            f"forecasts, including {names}."
        )

    if intent == "churn_risk":
        high_risk = df[df["churn_probability"] >= 60]
        return (
            f"{len(high_risk)} of the top {len(df)} flagged customers exceed a 60% churn "
            f"probability. The highest-risk customer (ID {int(df.iloc[0]['customer_id'])}) sits at "
            f"{df.iloc[0]['churn_probability']}%, driven mainly by low recent order activity."
        )

    if intent == "demand_forecast":
        total = df["predicted_qty"].sum()
        peak_row = df.loc[df["predicted_qty"].idxmax()]
        return (
            f"{meta.get('category', 'This category')} is forecast to sell about "
            f"{total:,.0f} units over the next {len(df)} days, peaking on {peak_row['date']} "
            f"at {peak_row['predicted_qty']:,.0f} units."
        )

    return "Here is the data for your question."


def _llm_insight(intent: str, df: pd.DataFrame, meta: dict) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    summary_csv = df.head(15).to_csv(index=False)
    prompt = (
        f"You are a business data analyst. Intent: {intent}. "
        f"Here is the relevant result data (CSV, truncated):\n{summary_csv}\n"
        "In 2-3 concise sentences, give a business insight a manager would care about. "
        "No preamble, no markdown, just the insight."
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=180,
        timeout=8,
    )
    return response.choices[0].message.content.strip()


def generate_insight(intent: str, df: pd.DataFrame, meta: dict) -> str:
    if OPENAI_API_KEY:
        try:
            return _llm_insight(intent, df, meta)
        except Exception as exc:  # network issues, quota, bad key, etc.
            fallback = _template_insight(intent, df, meta)
            return f"{fallback} (LLM explanation unavailable: {type(exc).__name__}; showing rule-based insight.)"
    return _template_insight(intent, df, meta)
