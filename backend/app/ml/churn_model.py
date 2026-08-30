"""XGBoost customer churn classifier.

Features are computed live from the customers/orders tables so predictions
always reflect current data, even for customers added after training.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from ..config import MODEL_DIR, RANDOM_SEED
from ..database import engine

MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.joblib")

FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_spend",
    "support_tickets",
    "orders_last_90d",
    "avg_order_value",
]

FEATURE_LABELS = {
    "tenure_months": "Account tenure (months)",
    "monthly_spend": "Monthly spend",
    "support_tickets": "Support tickets filed",
    "orders_last_90d": "Orders in last 90 days",
    "avg_order_value": "Average order value",
}

# direction=-1 means LOWER values of this feature raise churn risk
FEATURE_RISK_DIRECTION = {
    "tenure_months": -1,
    "monthly_spend": -1,
    "support_tickets": 1,
    "orders_last_90d": -1,
    "avg_order_value": -1,
}


def _build_feature_frame() -> pd.DataFrame:
    query = text(
        """
        SELECT
            c.customer_id,
            c.tenure_months,
            c.monthly_spend,
            c.support_tickets,
            c.churned,
            COALESCE(recent.orders_last_90d, 0) AS orders_last_90d,
            COALESCE(recent.avg_order_value, 0) AS avg_order_value
        FROM customers c
        LEFT JOIN (
            SELECT customer_id,
                   COUNT(*) AS orders_last_90d,
                   AVG(revenue) AS avg_order_value
            FROM orders
            WHERE customer_id != 0
              AND order_date >= CURRENT_DATE - INTERVAL '90 days'
            GROUP BY customer_id
        ) recent ON recent.customer_id = c.customer_id
        WHERE c.customer_id != 0
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df


def train_and_save() -> dict:
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = _build_feature_frame()

    X = df[FEATURE_COLUMNS]
    y = df["churned"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    joblib.dump({"model": model, "feature_importances": dict(zip(FEATURE_COLUMNS, model.feature_importances_))}, MODEL_PATH)
    print(f"[churn_model] Trained. Holdout AUC = {auc:.3f}")
    return {"auc": auc}


def _load():
    if not os.path.exists(MODEL_PATH):
        train_and_save()
    return joblib.load(MODEL_PATH)


def predict_all() -> pd.DataFrame:
    """Return churn probability for every real customer."""
    bundle = _load()
    model = bundle["model"]
    df = _build_feature_frame()
    df["churn_probability"] = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    return df


def predict_customer(customer_id: int) -> dict:
    bundle = _load()
    model = bundle["model"]
    importances = bundle["feature_importances"]
    df = _build_feature_frame()
    row = df[df["customer_id"] == customer_id]
    if row.empty:
        return None

    proba = float(model.predict_proba(row[FEATURE_COLUMNS])[:, 1][0])

    # population means, to explain *why* relative to the customer base
    means = df[FEATURE_COLUMNS].mean()
    stds = df[FEATURE_COLUMNS].std().replace(0, 1)

    risk_factors = []
    for col in FEATURE_COLUMNS:
        value = float(row.iloc[0][col])
        z = (value - means[col]) / stds[col]
        direction = FEATURE_RISK_DIRECTION[col]
        # direction=1 -> higher-than-average value is risky (e.g. support tickets)
        # direction=-1 -> lower-than-average value is risky (e.g. tenure, spend)
        risk_contribution = max(0.0, z * direction) * importances.get(col, 0)
        risk_factors.append(
            {
                "feature": FEATURE_LABELS[col],
                "value": round(value, 2),
                "population_avg": round(float(means[col]), 2),
                "risk_weight": round(float(risk_contribution), 4),
            }
        )
    risk_factors.sort(key=lambda x: x["risk_weight"], reverse=True)

    return {
        "customer_id": customer_id,
        "churn_probability": round(proba, 4),
        "top_risk_factors": risk_factors[:3],
    }


if __name__ == "__main__":
    train_and_save()
