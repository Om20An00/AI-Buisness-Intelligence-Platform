import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://insightpilot:insightpilot@postgres:5432/insightpilot",
)

# Optional: if set, the insight generator will use a real LLM call for
# natural-language explanations. If unset (default), InsightPilot falls
# back to a deterministic, template-based explanation engine so the whole
# platform runs end-to-end with zero external API cost or dependency.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

MODEL_DIR = os.getenv("MODEL_DIR", "/app/ml_artifacts")

RANDOM_SEED = 42

# Business thresholds used by the recommendation engine
CHURN_HIGH_RISK_THRESHOLD = 0.6
CHURN_MEDIUM_RISK_THRESHOLD = 0.35
LOW_STOCK_DAYS_COVER = 7  # flag a product if forecasted demand outpaces stock within N days
