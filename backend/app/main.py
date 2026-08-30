import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import query, churn, forecast, dashboard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("insightpilot")

app = FastAPI(
    title="InsightPilot — AI Decision Intelligence Platform",
    description="Natural-language business analytics agent combining SQL analytics, "
    "ML prediction (churn), and forecasting (demand) with an AI explanation layer.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router)
app.include_router(churn.router)
app.include_router(forecast.router)
app.include_router(dashboard.router)


@app.get("/")
def root():
    return {
        "service": "InsightPilot",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
