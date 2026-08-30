#!/bin/sh
set -e

echo "[entrypoint] Waiting for PostgreSQL..."
python3 - <<'EOF'
import time
import sys
from sqlalchemy import create_engine, text
from app.config import DATABASE_URL

for attempt in range(30):
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("[entrypoint] PostgreSQL is ready.")
        sys.exit(0)
    except Exception as exc:
        print(f"[entrypoint] DB not ready yet ({exc.__class__.__name__}), retrying... ({attempt + 1}/30)")
        time.sleep(2)

print("[entrypoint] PostgreSQL never became ready.")
sys.exit(1)
EOF

echo "[entrypoint] Seeding database (idempotent)..."
python3 -m app.seed_data

echo "[entrypoint] Training / loading ML models (idempotent)..."
python3 -m app.ml.churn_model
python3 -m app.ml.demand_model

echo "[entrypoint] Starting API server..."
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
