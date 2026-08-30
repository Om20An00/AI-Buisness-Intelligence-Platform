from typing import Any, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    intent: str
    entities: dict
    columns: list[str]
    rows: list[dict[str, Any]]
    chart_type: str
    insight: str
    recommendation: str


class ChurnResponse(BaseModel):
    customer_id: int
    churn_probability: float
    top_risk_factors: list[dict[str, Any]]
    recommendation: str


class ForecastResponse(BaseModel):
    target: str
    days: int
    forecast: list[dict[str, Any]]
    recommendation: Optional[str] = None
