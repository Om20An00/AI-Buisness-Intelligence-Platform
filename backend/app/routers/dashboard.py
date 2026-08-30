from fastapi import APIRouter
from sqlalchemy import text

from ..database import engine
from ..ml import churn_model

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])
SYSTEM_CUSTOMER_ID = 0


@router.get("/kpis")
def kpis():
    with engine.connect() as conn:
        total_revenue = conn.execute(
            text("SELECT COALESCE(SUM(revenue),0) FROM orders WHERE customer_id != :sid"),
            {"sid": SYSTEM_CUSTOMER_ID},
        ).scalar()
        total_customers = conn.execute(
            text("SELECT COUNT(*) FROM customers WHERE customer_id != :sid"), {"sid": SYSTEM_CUSTOMER_ID}
        ).scalar()
        churned = conn.execute(
            text("SELECT COUNT(*) FROM customers WHERE churned = TRUE AND customer_id != :sid"),
            {"sid": SYSTEM_CUSTOMER_ID},
        ).scalar()
        top_category = conn.execute(
            text(
                """
                SELECT p.category, SUM(o.revenue) AS rev
                FROM orders o JOIN products p ON p.product_id = o.product_id
                WHERE o.customer_id != :sid
                GROUP BY p.category ORDER BY rev DESC LIMIT 1
                """
            ),
            {"sid": SYSTEM_CUSTOMER_ID},
        ).fetchone()

        trend_rows = conn.execute(
            text(
                """
                SELECT order_date, SUM(revenue) AS revenue
                FROM orders
                WHERE customer_id != :sid AND order_date >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY order_date ORDER BY order_date
                """
            ),
            {"sid": SYSTEM_CUSTOMER_ID},
        ).fetchall()

        category_rows = conn.execute(
            text(
                """
                SELECT p.category, SUM(o.revenue) AS revenue
                FROM orders o JOIN products p ON p.product_id = o.product_id
                WHERE o.customer_id != :sid
                GROUP BY p.category ORDER BY revenue DESC
                """
            ),
            {"sid": SYSTEM_CUSTOMER_ID},
        ).fetchall()

    return {
        "total_revenue": float(total_revenue or 0),
        "total_customers": int(total_customers or 0),
        "churn_rate": round((churned or 0) / max(1, total_customers or 1) * 100, 1),
        "top_category": top_category[0] if top_category else None,
        "revenue_trend": [{"date": r[0].isoformat(), "revenue": float(r[1])} for r in trend_rows],
        "revenue_by_category": [{"category": r[0], "revenue": float(r[1])} for r in category_rows],
    }


@router.get("/customers")
def customers(limit: int = 100):
    """Customer list with live churn scores, for the Churn Explorer dropdown."""
    df = churn_model.predict_all().sort_values("churn_probability", ascending=False).head(limit)
    with engine.connect() as conn:
        names = conn.execute(
            text("SELECT customer_id, name, region, segment FROM customers WHERE customer_id != 0")
        ).fetchall()
    name_map = {r[0]: {"name": r[1], "region": r[2], "segment": r[3]} for r in names}

    out = []
    for _, row in df.iterrows():
        cid = int(row["customer_id"])
        meta = name_map.get(cid, {})
        out.append(
            {
                "customer_id": cid,
                "name": meta.get("name"),
                "region": meta.get("region"),
                "segment": meta.get("segment"),
                "churn_probability": round(float(row["churn_probability"]) * 100, 1),
            }
        )
    return {"customers": out}
