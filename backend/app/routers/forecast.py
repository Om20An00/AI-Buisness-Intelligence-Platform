from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from ..database import engine
from ..ml import demand_model
from ..schemas import ForecastResponse

router = APIRouter(prefix="/api/forecast", tags=["forecast"])


@router.get("/categories")
def list_categories():
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT DISTINCT category FROM products ORDER BY category")).fetchall()
    return {"categories": [r[0] for r in rows]}


@router.get("/products")
def list_products(category: Optional[str] = None):
    query = "SELECT product_id, name, category, stock_qty FROM products"
    params = {}
    if category:
        query += " WHERE category = :category"
        params["category"] = category
    with engine.connect() as conn:
        df = conn.execute(text(query), params)
        rows = [dict(r._mapping) for r in df]
    return {"products": rows}


@router.get("", response_model=ForecastResponse)
def forecast(
    category: Optional[str] = Query(None),
    product_id: Optional[int] = Query(None),
    days: int = Query(7, ge=1, le=30),
):
    if not category and not product_id:
        raise HTTPException(status_code=400, detail="Provide either category or product_id")

    if product_id:
        df = demand_model.forecast_product(product_id, days)
        target = f"product #{product_id}"
    else:
        df = demand_model.forecast_category(category, days)
        target = category

    if df.empty:
        raise HTTPException(status_code=404, detail="No history found for this target")

    total = df["predicted_qty"].sum()
    rec = f"Expect ~{total:,.0f} units of demand over the next {days} days for {target} — plan stock and staffing accordingly."

    return ForecastResponse(
        target=target,
        days=days,
        forecast=df.to_dict(orient="records"),
        recommendation=rec,
    )
