from sqlalchemy import Column, Integer, String, Float, Date, Boolean, ForeignKey
from sqlalchemy.orm import relationship

from .database import Base


class Customer(Base):
    __tablename__ = "customers"

    customer_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    region = Column(String, index=True)
    segment = Column(String, index=True)
    signup_date = Column(Date)
    tenure_months = Column(Integer)
    monthly_spend = Column(Float)
    support_tickets = Column(Integer, default=0)
    # Historical ground-truth label used only to TRAIN the churn model.
    # Synthetic data: generated from a documented rule + noise (see seed_data.py),
    # not a real business outcome.
    churned = Column(Boolean, default=False)

    orders = relationship("Order", back_populates="customer")


class Product(Base):
    __tablename__ = "products"

    product_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    category = Column(String, index=True)
    unit_price = Column(Float)
    stock_qty = Column(Integer)

    orders = relationship("Order", back_populates="product")


class Order(Base):
    __tablename__ = "orders"

    order_id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey("customers.customer_id"), index=True)
    product_id = Column(Integer, ForeignKey("products.product_id"), index=True)
    order_date = Column(Date, index=True)
    quantity = Column(Integer)
    revenue = Column(Float)

    customer = relationship("Customer", back_populates="orders")
    product = relationship("Product", back_populates="orders")
