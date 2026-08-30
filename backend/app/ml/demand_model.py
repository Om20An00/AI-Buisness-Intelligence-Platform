"""XGBoost demand forecasting model.

Trained once on the full daily-demand history (per product, per category)
with lag/rolling/seasonal features. At inference time it forecasts forward
day-by-day, feeding each prediction back in as the next day's lag feature
(a standard walk-forward approach for short-horizon forecasting).
"""
import os
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

from ..config import MODEL_DIR, RANDOM_SEED
from ..database import engine

MODEL_PATH = os.path.join(MODEL_DIR, "demand_model.joblib")
SYSTEM_CUSTOMER_ID = 0

FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_mean_14",
    "day_of_week",
    "day_of_month",
    "month",
    "category_code",
]


def _raw_history() -> pd.DataFrame:
    query = text(
        """
        SELECT o.order_date, o.product_id, p.category, SUM(o.quantity) AS qty
        FROM orders o
        JOIN products p ON p.product_id = o.product_id
        WHERE o.customer_id = :sys_id
        GROUP BY o.order_date, o.product_id, p.category
        ORDER BY o.product_id, o.order_date
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"sys_id": SYSTEM_CUSTOMER_ID})
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def _add_features(df: pd.DataFrame, category_map: dict) -> pd.DataFrame:
    df = df.sort_values(["product_id", "order_date"]).copy()
    df["lag_1"] = df.groupby("product_id")["qty"].shift(1)
    df["lag_7"] = df.groupby("product_id")["qty"].shift(7)
    df["rolling_mean_7"] = df.groupby("product_id")["qty"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    df["rolling_mean_14"] = df.groupby("product_id")["qty"].transform(
        lambda s: s.shift(1).rolling(14, min_periods=1).mean()
    )
    df["day_of_week"] = df["order_date"].dt.dayofweek
    df["day_of_month"] = df["order_date"].dt.day
    df["month"] = df["order_date"].dt.month
    df["category_code"] = df["category"].map(category_map)
    return df


def train_and_save() -> dict:
    os.makedirs(MODEL_DIR, exist_ok=True)
    raw = _raw_history()
    categories = sorted(raw["category"].unique())
    category_map = {c: i for i, c in enumerate(categories)}

    df = _add_features(raw, category_map)
    df = df.dropna(subset=FEATURE_COLUMNS)

    X = df[FEATURE_COLUMNS]
    y = df["qty"]

    split_date = df["order_date"].quantile(0.85)
    train_mask = df["order_date"] < split_date
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    model = XGBRegressor(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test)) if len(X_test) else float("nan")

    joblib.dump({"model": model, "category_map": category_map}, MODEL_PATH)
    print(f"[demand_model] Trained. Holdout MAE = {mae:.2f} units/day")
    return {"mae": mae}


def _load():
    if not os.path.exists(MODEL_PATH):
        train_and_save()
    return joblib.load(MODEL_PATH)


def forecast_product(product_id: int, days: int = 7) -> pd.DataFrame:
    bundle = _load()
    model = bundle["model"]
    category_map = bundle["category_map"]

    raw = _raw_history()
    hist = raw[raw["product_id"] == product_id].sort_values("order_date")
    if hist.empty:
        return pd.DataFrame(columns=["date", "predicted_qty"])

    category = hist["category"].iloc[0]
    cat_code = category_map.get(category, 0)
    series = list(hist["qty"].values[-30:])  # rolling working window
    last_date = hist["order_date"].max()

    forecasts = []
    for step in range(1, days + 1):
        future_date = last_date + timedelta(days=step)
        lag_1 = series[-1]
        lag_7 = series[-7] if len(series) >= 7 else series[0]
        rolling_7 = float(np.mean(series[-7:]))
        rolling_14 = float(np.mean(series[-14:])) if len(series) >= 14 else float(np.mean(series))

        features = pd.DataFrame(
            [{
                "lag_1": lag_1,
                "lag_7": lag_7,
                "rolling_mean_7": rolling_7,
                "rolling_mean_14": rolling_14,
                "day_of_week": future_date.dayofweek,
                "day_of_month": future_date.day,
                "month": future_date.month,
                "category_code": cat_code,
            }]
        )
        pred = max(0.0, float(model.predict(features[FEATURE_COLUMNS])[0]))
        forecasts.append({"date": future_date.date().isoformat(), "predicted_qty": round(pred, 1)})
        series.append(pred)

    return pd.DataFrame(forecasts)


def forecast_category(category: str, days: int = 7) -> pd.DataFrame:
    """Sum per-product forecasts up to the category level."""
    with engine.connect() as conn:
        product_ids = pd.read_sql(
            text("SELECT product_id FROM products WHERE category = :cat"),
            conn,
            params={"cat": category},
        )["product_id"].tolist()

    combined = None
    for pid in product_ids:
        f = forecast_product(pid, days)
        if f.empty:
            continue
        f = f.set_index("date")
        combined = f if combined is None else combined.add(f, fill_value=0)

    if combined is None:
        return pd.DataFrame(columns=["date", "predicted_qty"])
    return combined.reset_index()


if __name__ == "__main__":
    train_and_save()
