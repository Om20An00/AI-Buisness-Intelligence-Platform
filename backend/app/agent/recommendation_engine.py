"""Turns model outputs / analytics into a concrete, actionable next step.
Deliberately rule-based (transparent, defensible in an interview) rather
than another LLM call."""
import pandas as pd

from ..config import CHURN_HIGH_RISK_THRESHOLD, CHURN_MEDIUM_RISK_THRESHOLD


def generate_recommendation(intent: str, df: pd.DataFrame, meta: dict) -> str:
    if df.empty:
        return "No action needed — no matching data."

    if intent == "churn_risk":
        high = df[df["churn_probability"] >= CHURN_HIGH_RISK_THRESHOLD * 100]
        if len(high) > 0:
            ids = ", ".join(str(int(x)) for x in high["customer_id"].head(5))
            return f"Prioritize proactive retention outreach for customer(s) {ids} — churn probability above 60%."
        return "No customers in this list exceed the high-risk threshold; standard retention cadence is sufficient."

    if intent == "low_stock":
        risky = df[df["at_risk_of_stockout"]]
        if risky.empty:
            return "No reorder action required this week."
        lines = [
            f"{row['name']}: reorder ~{int(row['forecasted_demand_7d'] - row['stock_qty'] + 10)} units"
            for _, row in risky.head(3).iterrows()
        ]
        return "Reorder recommended — " + "; ".join(lines) + "."

    if intent == "demand_forecast":
        return "Align next week's inventory and staffing plans with the forecasted volume above."

    if intent == "top_customers":
        return "Consider a loyalty or account-management touchpoint for this segment to protect retention."

    if intent == "product_performance":
        worst = df.sort_values("total_revenue").iloc[0]
        return f"'{worst['name']}' is underperforming this list — consider a promotion or bundling it with a top seller."

    if intent == "revenue_summary":
        laggard = df.sort_values("total_revenue").iloc[0]
        return f"{laggard['region']} is the lowest-performing region — worth investigating regional demand drivers."

    return "Review the data above for follow-up actions."
