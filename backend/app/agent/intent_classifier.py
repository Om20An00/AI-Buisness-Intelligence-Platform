"""Lightweight, dependency-free intent classification for the AI Analytics
Agent. Deliberately rule/keyword based (not an LLM call) so the whole
platform works instantly with zero external API cost — this is the routing
layer that decides whether a question needs SQL analytics, the churn model,
or the demand-forecasting model.
"""
import re

REGIONS = ["north", "south", "east", "west"]
CATEGORIES = ["electronics", "apparel", "home", "grocery"]

INTENT_PATTERNS = {
    "churn_risk": [r"churn", r"at risk", r"retention", r"risk of leaving", r"attrition"],
    "demand_forecast": [r"forecast", r"predict(ing)? demand", r"next \d*\s*(week|month|day)", r"expected demand", r"projection"],
    "low_stock": [r"stock", r"inventory", r"reorder", r"restock", r"running out"],
    "top_customers": [r"top\s*\d*\s*customers?", r"best\s*\d*\s*customers?", r"highest spending", r"biggest customer"],
    "product_performance": [r"best.?selling", r"top\s*\d*\s*products?", r"product performance", r"worst product"],
    "revenue_summary": [r"revenue", r"sales", r"income", r"earnings"],
}
# Order matters: more specific intents are checked first so a phrase like
# "top 5 customers" isn't swallowed by a looser "revenue" match.
INTENT_PRIORITY = ["churn_risk", "demand_forecast", "low_stock", "top_customers", "product_performance", "revenue_summary"]

DEFAULT_INTENT = "revenue_summary"


def _extract_region(text: str):
    for r in REGIONS:
        if r in text:
            return r.capitalize()
    return None


def _extract_category(text: str):
    for c in CATEGORIES:
        if c in text:
            return c.capitalize()
    return None


def _extract_days(text: str) -> int:
    match = re.search(r"(\d+)\s*day", text)
    if match:
        return max(1, min(60, int(match.group(1))))
    if "week" in text:
        weeks = re.search(r"(\d+)\s*week", text)
        return (int(weeks.group(1)) if weeks else 1) * 7
    if "month" in text:
        months = re.search(r"(\d+)\s*month", text)
        return (int(months.group(1)) if months else 1) * 30
    return 30


def _extract_top_n(text: str) -> int:
    match = re.search(r"top\s*(\d+)", text)
    if match:
        return max(1, min(50, int(match.group(1))))
    return 10


def classify(question: str) -> dict:
    q = question.lower().strip()

    matched_intent = DEFAULT_INTENT
    for intent in INTENT_PRIORITY:
        patterns = INTENT_PATTERNS[intent]
        if any(re.search(p, q) for p in patterns):
            matched_intent = intent
            break

    entities = {
        "region": _extract_region(q),
        "category": _extract_category(q),
        "window_days": _extract_days(q),
        "top_n": _extract_top_n(q),
    }
    return {"intent": matched_intent, "entities": entities, "raw_question": question}
