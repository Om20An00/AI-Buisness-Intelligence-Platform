"""Executes the right data operation for a classified intent: either a
parameterized SQL query against PostgreSQL, or a call into the churn /
demand ML models. Returns (DataFrame, meta dict) so the router can attach
insights + recommendations + chart hints uniformly.
"""
import pandas as pd
from sqlalchemy import text

from ..database import engine
from ..ml import churn_model, demand_model

SYSTEM_CUSTOMER_ID = 0


def _run(query: str, params: dict) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


def revenue_summary(entities: dict):
    region = entities.get("region")
    days = entities.get("window_days", 30)
    query = """
        SELECT c.region, SUM(o.revenue) AS total_revenue, COUNT(*) AS order_count
        FROM orders o
        JOIN customers c ON c.customer_id = o.customer_id
        WHERE o.customer_id != :sys_id
          AND o.order_date >= CURRENT_DATE - (:days || ' days')::interval
          AND (:region IS NULL OR c.region = :region)
        GROUP BY c.region
        ORDER BY total_revenue DESC
    """
    df = _run(query, {"sys_id": SYSTEM_CUSTOMER_ID, "days": days, "region": region})
    meta = {"chart_type": "bar", "x": "region", "y": "total_revenue", "window_days": days}
    return df, meta


def top_customers(entities: dict):
    top_n = entities.get("top_n", 10)
    query = """
        SELECT c.customer_id, c.name, c.region, c.segment,
               SUM(o.revenue) AS total_revenue, COUNT(*) AS order_count
        FROM orders o
        JOIN customers c ON c.customer_id = o.customer_id
        WHERE o.customer_id != :sys_id
        GROUP BY c.customer_id, c.name, c.region, c.segment
        ORDER BY total_revenue DESC
        LIMIT :top_n
    """
    df = _run(query, {"sys_id": SYSTEM_CUSTOMER_ID, "top_n": top_n})
    meta = {"chart_type": "bar", "x": "name", "y": "total_revenue"}
    return df, meta


def product_performance(entities: dict):
    category = entities.get("category")
    query = """
        SELECT p.name, p.category, SUM(o.quantity) AS units_sold, SUM(o.revenue) AS total_revenue
        FROM orders o
        JOIN products p ON p.product_id = o.product_id
        WHERE o.customer_id != :sys_id
          AND (:category IS NULL OR p.category = :category)
        GROUP BY p.name, p.category
        ORDER BY total_revenue DESC
        LIMIT 15
    """
    df = _run(query, {"sys_id": SYSTEM_CUSTOMER_ID, "category": category})
    meta = {"chart_type": "bar", "x": "name", "y": "total_revenue"}
    return df, meta


def low_stock(entities: dict):
    query = """
        SELECT product_id, name, category, stock_qty, unit_price
        FROM products
        ORDER BY stock_qty ASC
        LIMIT 10
    """
    df = _run(query, {})
    # augment with a 7-day demand forecast so "low stock" is forward-looking,
    # not just a snapshot of current inventory
    covers = []
    for _, row in df.iterrows():
        fc = demand_model.forecast_product(int(row["product_id"]), days=7)
        forecasted_7d = float(fc["predicted_qty"].sum()) if not fc.empty else 0.0
        covers.append(forecasted_7d)
    df["forecasted_demand_7d"] = [round(c, 1) for c in covers]
    df["at_risk_of_stockout"] = df["forecasted_demand_7d"] > df["stock_qty"]
    meta = {"chart_type": "table"}
    return df, meta


def churn_risk(entities: dict):
    top_n = entities.get("top_n", 10)
    df = churn_model.predict_all()
    df = df.sort_values("churn_probability", ascending=False).head(top_n)
    df = df[["customer_id", "tenure_months", "monthly_spend", "support_tickets",
             "orders_last_90d", "churn_probability"]]
    df["churn_probability"] = (df["churn_probability"] * 100).round(1)
    meta = {"chart_type": "bar", "x": "customer_id", "y": "churn_probability"}
    return df, meta


def demand_forecast(entities: dict):
    category = entities.get("category")
    days = min(entities.get("window_days", 7), 30)
    if not category:
        category = "Electronics"
    df = demand_model.forecast_category(category, days=days)
    meta = {"chart_type": "line", "x": "date", "y": "predicted_qty", "category": category}
    return df, meta


INTENT_HANDLERS = {
    "revenue_summary": revenue_summary,
    "top_customers": top_customers,
    "product_performance": product_performance,
    "low_stock": low_stock,
    "churn_risk": churn_risk,
    "demand_forecast": demand_forecast,
}


def handle(intent: str, entities: dict):
    handler = INTENT_HANDLERS.get(intent, revenue_summary)
    return handler(entities)
