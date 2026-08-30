import numpy as np
from fastapi import APIRouter

from ..agent.intent_classifier import classify
from ..agent.sql_analytics import handle
from ..agent.insight_generator import generate_insight
from ..agent.recommendation_engine import generate_recommendation
from ..schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/api", tags=["agent"])


def _sanitize(df):
    """Convert numpy/pandas types to plain python for JSON serialization."""
    return df.replace({np.nan: None}).to_dict(orient="records")


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest):
    classification = classify(payload.question)
    intent, entities = classification["intent"], classification["entities"]

    df, meta = handle(intent, entities)
    insight = generate_insight(intent, df, meta)
    recommendation = generate_recommendation(intent, df, meta)

    return QueryResponse(
        intent=intent,
        entities=entities,
        columns=list(df.columns),
        rows=_sanitize(df),
        chart_type=meta.get("chart_type", "table"),
        insight=insight,
        recommendation=recommendation,
    )


@router.get("/query/examples")
def examples():
    return {
        "examples": [
            "What was our revenue in the North region last 30 days?",
            "Who are our top 10 customers?",
            "Which customers are at risk of churn?",
            "Forecast demand for Electronics next 14 days",
            "Any products running low on stock?",
            "What are the best selling products in Apparel?",
        ]
    }
