"""
Generates a synthetic but realistically-structured retail dataset:
customers, products, and orders. Designed to be idempotent (safe to
run on every container start) and to bake in *real, learnable signal*
for the churn and demand-forecasting models, rather than pure noise.

Run directly:  python -m app.seed_data
"""
import random
from datetime import date, timedelta

import numpy as np
from faker import Faker
from sqlalchemy.orm import Session

from .config import RANDOM_SEED
from .database import Base, engine, SessionLocal
from .models import Customer, Product, Order

fake = Faker()
Faker.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

REGIONS = ["North", "South", "East", "West"]
SEGMENTS = ["SMB", "Mid-Market", "Enterprise"]

CATEGORIES = {
    "Electronics": ["Wireless Earbuds", "Smart Speaker", "4K Monitor", "Laptop Stand"],
    "Apparel": ["Running Shoes", "Denim Jacket", "Wool Sweater", "Rain Jacket"],
    "Home": ["Air Purifier", "Coffee Maker", "Desk Lamp", "Cookware Set"],
    "Grocery": ["Organic Coffee Beans", "Protein Bars", "Cold Brew Pack", "Snack Box"],
}

N_CUSTOMERS = 600
DEMAND_HISTORY_DAYS = 365
SYSTEM_CUSTOMER_ID = 0  # holds daily aggregate demand rows used only for forecasting

# category demand behaviour: (base_units_per_day, trend_per_day, weekly_amplitude)
CATEGORY_DEMAND_PROFILE = {
    "Electronics": dict(base=28, trend=0.035, weekly_amp=0.15, month_amp=0.10),
    "Apparel": dict(base=20, trend=0.005, weekly_amp=0.25, month_amp=0.35),
    "Home": dict(base=16, trend=0.012, weekly_amp=0.10, month_amp=0.15),
    "Grocery": dict(base=40, trend=-0.004, weekly_amp=0.05, month_amp=0.05),
}


def _already_seeded(db: Session) -> bool:
    return db.query(Customer).count() > 0


def _make_products(db: Session):
    pid = 1
    products = []
    for category, names in CATEGORIES.items():
        for name in names:
            price = round(random.uniform(15, 250), 2)
            product = Product(
                product_id=pid,
                name=name,
                category=category,
                unit_price=price,
                stock_qty=random.randint(80, 400),
            )
            products.append(product)
            pid += 1
    db.add_all(products)
    db.flush()
    return products


def _make_system_customer(db: Session):
    # Placeholder customer that owns synthetic daily demand rows so those
    # rows never distort per-customer churn features or "top customers".
    sys_customer = Customer(
        customer_id=SYSTEM_CUSTOMER_ID,
        name="SYSTEM_DEMAND_SIGNAL",
        region="N/A",
        segment="N/A",
        signup_date=date.today() - timedelta(days=DEMAND_HISTORY_DAYS),
        tenure_months=0,
        monthly_spend=0,
        support_tickets=0,
        churned=False,
    )
    db.add(sys_customer)
    db.flush()


def _make_customers(db: Session):
    customers = []
    today = date.today()
    for cid in range(1, N_CUSTOMERS + 1):
        tenure = int(np.random.gamma(shape=2.0, scale=10))  # months, skewed toward newer accounts
        tenure = max(1, min(tenure, 60))
        signup = today - timedelta(days=tenure * 30 + random.randint(0, 29))
        segment = random.choices(SEGMENTS, weights=[0.55, 0.30, 0.15])[0]
        base_spend = {"SMB": 80, "Mid-Market": 220, "Enterprise": 550}[segment]
        monthly_spend = round(max(10, np.random.normal(base_spend, base_spend * 0.3)), 2)
        support_tickets = int(np.random.poisson(2 if tenure < 12 else 1))

        # --- synthetic churn rule (documented, not fabricated as "real" data) ---
        # Risk increases with: short tenure, low spend relative to segment norm,
        # high support ticket volume. A logistic-ish score + noise -> label.
        risk_score = (
            -0.05 * tenure
            + 0.9 * (support_tickets / (1 + tenure / 12))
            - 0.004 * (monthly_spend - base_spend)
            + np.random.normal(0, 1.1)
        )
        churn_prob = 1 / (1 + np.exp(-(risk_score - 1.0)))
        churned = np.random.random() < churn_prob

        customers.append(
            Customer(
                customer_id=cid,
                name=fake.name(),
                region=random.choice(REGIONS),
                segment=segment,
                signup_date=signup,
                tenure_months=tenure,
                monthly_spend=monthly_spend,
                support_tickets=support_tickets,
                churned=bool(churned),
            )
        )
    db.add_all(customers)
    db.flush()
    return customers


def _make_customer_orders(db: Session, customers, products):
    """Individual purchase events per customer over the last 180 days.
    Used for churn features (recency/frequency/value) and general SQL
    analytics (revenue by region, top customers, product performance)."""
    order_id = 1
    orders = []
    today = date.today()
    window_days = 180

    for c in customers:
        if c.customer_id == SYSTEM_CUSTOMER_ID:
            continue
        # active (non-churned) customers order more recently/frequently
        n_orders = np.random.poisson(6 if not c.churned else 2)
        for _ in range(n_orders):
            if c.churned:
                # churned customers' activity is concentrated further in the past
                days_ago = random.randint(60, window_days)
            else:
                days_ago = random.randint(0, window_days)
            order_date = today - timedelta(days=days_ago)
            product = random.choice(products)
            qty = random.randint(1, 4)
            revenue = round(qty * product.unit_price * random.uniform(0.9, 1.05), 2)
            orders.append(
                Order(
                    order_id=order_id,
                    customer_id=c.customer_id,
                    product_id=product.product_id,
                    order_date=order_date,
                    quantity=qty,
                    revenue=revenue,
                )
            )
            order_id += 1
    db.add_all(orders)
    db.flush()
    return order_id


def _make_demand_series(db: Session, products, start_order_id):
    """Daily aggregate demand per product for the last DEMAND_HISTORY_DAYS,
    with trend + weekly + monthly seasonality baked in, attributed to the
    SYSTEM_DEMAND_SIGNAL customer. This is what the forecasting model learns from."""
    order_id = start_order_id
    orders = []
    today = date.today()
    start = today - timedelta(days=DEMAND_HISTORY_DAYS)

    for product in products:
        profile = CATEGORY_DEMAND_PROFILE[product.category]
        for day_idx in range(DEMAND_HISTORY_DAYS):
            d = start + timedelta(days=day_idx)
            weekly = 1 + profile["weekly_amp"] * np.sin(2 * np.pi * d.weekday() / 7)
            monthly = 1 + profile["month_amp"] * np.sin(2 * np.pi * d.day / 30)
            trend = 1 + profile["trend"] * day_idx / 30
            noise = np.random.normal(1.0, 0.12)
            qty = max(0, int(round(profile["base"] * weekly * monthly * trend * noise)))
            if qty == 0:
                continue
            revenue = round(qty * product.unit_price, 2)
            orders.append(
                Order(
                    order_id=order_id,
                    customer_id=SYSTEM_CUSTOMER_ID,
                    product_id=product.product_id,
                    order_date=d,
                    quantity=qty,
                    revenue=revenue,
                )
            )
            order_id += 1
    db.add_all(orders)
    db.flush()


def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if _already_seeded(db):
            print("[seed_data] Database already seeded, skipping.")
            return
        print("[seed_data] Seeding synthetic dataset...")
        _make_system_customer(db)
        products = _make_products(db)
        customers = _make_customers(db)
        next_id = _make_customer_orders(db, customers, products)
        _make_demand_series(db, products, next_id)
        db.commit()
        print(f"[seed_data] Done: {len(customers)} customers, {len(products)} products.")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed()
