from fastapi import APIRouter, HTTPException

from ..ml import churn_model
from ..agent.recommendation_engine import generate_recommendation
from ..schemas import ChurnResponse

router = APIRouter(prefix="/api/churn", tags=["churn"])


@router.get("/{customer_id}", response_model=ChurnResponse)
def get_customer_churn(customer_id: int):
    result = churn_model.predict_customer(customer_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Customer not found")

    if result["churn_probability"] >= 0.6:
        rec = "High risk — prioritize proactive retention outreach (call or personalized offer) within the week."
    elif result["churn_probability"] >= 0.35:
        rec = "Medium risk — add to next retention email campaign and monitor support tickets."
    else:
        rec = "Low risk — no action needed beyond standard engagement."

    return ChurnResponse(**result, recommendation=rec)


@router.get("")
def list_top_risk(limit: int = 20):
    df = churn_model.predict_all().sort_values("churn_probability", ascending=False).head(limit)
    return df[["customer_id", "churn_probability"]].to_dict(orient="records")
