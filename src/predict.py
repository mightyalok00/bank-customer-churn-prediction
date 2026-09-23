"""Reusable prediction helpers shared by the app and automated tests."""
from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary", "Rating",
    "Geography", "Gender",
]


def make_customer_frame(**values: object) -> pd.DataFrame:
    """Build a one-row model input in the exact training-column order."""
    missing = [column for column in FEATURE_COLUMNS if column not in values]
    if missing:
        raise ValueError(f"Missing customer features: {', '.join(missing)}")
    return pd.DataFrame([{column: values[column] for column in FEATURE_COLUMNS}])


def risk_priority(probability: float) -> str:
    """Map a churn probability to an operational priority."""
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


def retention_action(probability: float) -> str:
    """Return a human-review recommendation for a risk score."""
    priority = risk_priority(probability)
    if priority == "High":
        return "Assign a relationship-manager review and consider a tailored retention offer."
    if priority == "Medium":
        return "Use a targeted engagement campaign and monitor the customer's response."
    return "Maintain regular service and avoid unnecessary retention spending."
