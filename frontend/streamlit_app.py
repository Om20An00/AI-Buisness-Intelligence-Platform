import os

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="InsightPilot", page_icon="📊", layout="wide")


def api_get(path, params=None):
    try:
        r = requests.get(f"{BACKEND_URL}{path}", params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"Backend request failed: {exc}")
        return None


def api_post(path, payload):
    try:
        r = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"Backend request failed: {exc}")
        return None


st.title("📊 InsightPilot")
st.caption("AI Decision Intelligence Platform — ask a business question in plain English and get SQL analytics, ML predictions, and forecasts back with an explanation and a recommendation.")

tab_ask, tab_dashboard, tab_churn, tab_forecast = st.tabs(
    ["💬 Ask InsightPilot", "📈 Dashboard", "⚠️ Churn Explorer", "📦 Demand Forecast"]
)

# ----------------------------- Ask tab -----------------------------
with tab_ask:
    examples = api_get("/api/query/examples")
    if examples:
        st.write("Try one of these, or type your own question:")
        cols = st.columns(3)
        for i, ex in enumerate(examples["examples"]):
            if cols[i % 3].button(ex, key=f"ex_{i}"):
                st.session_state["question"] = ex

    question = st.text_input(
        "Ask a business question",
        value=st.session_state.get("question", ""),
        placeholder="e.g. Which customers are at risk of churn?",
    )

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Routing question through the AI Analytics Agent..."):
            result = api_post("/api/query", {"question": question})

        if result:
            st.caption(f"Detected intent: `{result['intent']}`  ·  entities: `{result['entities']}`")

            df = pd.DataFrame(result["rows"], columns=result["columns"])
            if df.empty:
                st.warning("No data matched this question.")
            else:
                chart_type = result.get("chart_type", "table")
                if chart_type == "bar" and len(df.columns) >= 2:
                    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                    if numeric_cols:
                        st.bar_chart(df.set_index(df.columns[0])[numeric_cols[0]])
                elif chart_type == "line" and "date" in df.columns:
                    st.line_chart(df.set_index("date"))

                st.dataframe(df, use_container_width=True)

            st.info(f"**Insight:** {result['insight']}")
            st.success(f"**Recommendation:** {result['recommendation']}")

# ----------------------------- Dashboard tab -----------------------------
with tab_dashboard:
    kpis = api_get("/api/dashboard/kpis")
    if kpis:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Revenue", f"${kpis['total_revenue']:,.0f}")
        c2.metric("Total Customers", f"{kpis['total_customers']:,}")
        c3.metric("Churn Rate", f"{kpis['churn_rate']}%")
        c4.metric("Top Category", kpis["top_category"] or "—")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Revenue — Last 30 Days")
            trend = pd.DataFrame(kpis["revenue_trend"])
            if not trend.empty:
                st.line_chart(trend.set_index("date"))
        with col_b:
            st.subheader("Revenue by Category")
            cat = pd.DataFrame(kpis["revenue_by_category"])
            if not cat.empty:
                st.bar_chart(cat.set_index("category"))

# ----------------------------- Churn Explorer tab -----------------------------
with tab_churn:
    st.subheader("Customers ranked by predicted churn risk")
    customers = api_get("/api/dashboard/customers", params={"limit": 50})
    if customers and customers["customers"]:
        clist = customers["customers"]
        options = {f"#{c['customer_id']} — {c['name']} ({c['churn_probability']}% risk)": c["customer_id"] for c in clist}
        selected_label = st.selectbox("Select a customer", list(options.keys()))
        selected_id = options[selected_label]

        detail = api_get(f"/api/churn/{selected_id}")
        if detail:
            st.metric("Churn Probability", f"{detail['churn_probability'] * 100:.1f}%")
            st.write("**Top risk factors** (vs. customer-base average):")
            st.dataframe(pd.DataFrame(detail["top_risk_factors"]), use_container_width=True)
            st.success(f"**Recommendation:** {detail['recommendation']}")

        st.divider()
        st.write("Top 50 customers by churn risk:")
        st.dataframe(pd.DataFrame(clist), use_container_width=True)

# ----------------------------- Demand Forecast tab -----------------------------
with tab_forecast:
    cats = api_get("/api/forecast/categories")
    if cats and cats["categories"]:
        category = st.selectbox("Category", cats["categories"])
        days = st.slider("Forecast horizon (days)", 1, 30, 7)

        if st.button("Generate Forecast", type="primary"):
            fc = api_get("/api/forecast", params={"category": category, "days": days})
            if fc:
                fdf = pd.DataFrame(fc["forecast"])
                st.line_chart(fdf.set_index("date"))
                st.dataframe(fdf, use_container_width=True)
                st.success(f"**Recommendation:** {fc['recommendation']}")
