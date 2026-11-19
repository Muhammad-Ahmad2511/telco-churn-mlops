# src/serving/schema.py
"""Request schema and validation for the churn API."""
from __future__ import annotations

from typing import Any

import pandas as pd

MAX_INSTANCES = 100

YES_NO = ("Yes", "No")
YES_NO_NO_INTERNET = ("Yes", "No", "No internet service")

# Same categories as the raw-data schema in src/data/validation.py.
CATEGORICAL = {
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ("Yes", "No", "No phone service"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    "OnlineSecurity": YES_NO_NO_INTERNET,
    "OnlineBackup": YES_NO_NO_INTERNET,
    "DeviceProtection": YES_NO_NO_INTERNET,
    "TechSupport": YES_NO_NO_INTERNET,
    "StreamingTV": YES_NO_NO_INTERNET,
    "StreamingMovies": YES_NO_NO_INTERNET,
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": YES_NO,
    "PaymentMethod": (
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ),
}

# name: (type, min, max). The upper bounds are loose sanity limits chosen by us.
NUMERIC = {
    "SeniorCitizen": (int, 0, 1),
    "tenure": (int, 0, 120),
    "MonthlyCharges": (float, 0.0, 1000.0),
    "TotalCharges": (float, 0.0, 100000.0),
}

# 18 model inputs: customerID, gender and Churn are not model inputs.
FEATURE_ORDER = (
    "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
    "TotalCharges",
)

SAMPLE_RECORD = {
    "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 12,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
    "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85,
    "TotalCharges": 358.2,
}


def validate_record(record: Any) -> list[str]:
    if not isinstance(record, dict):
        return ["each instance must be a JSON object"]
    errors = []
    missing = [f for f in FEATURE_ORDER if f not in record]
    unknown = sorted(set(record) - set(FEATURE_ORDER))
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    if unknown:
        errors.append(f"unknown fields: {', '.join(unknown)}")
    for name, allowed in CATEGORICAL.items():
        if name in record and record[name] not in allowed:
            errors.append(f"{name} must be one of {list(allowed)}")
    for name, (kind, low, high) in NUMERIC.items():
        if name not in record:
            continue
        value = record[name]
        allowed_types = (int,) if kind is int else (int, float)
        if isinstance(value, bool) or not isinstance(value, allowed_types):
            errors.append(f"{name} must be a number of type {kind.__name__}")
        elif not low <= value <= high:  # NaN and inf also fail this check
            errors.append(f"{name} must be between {low} and {high}")
    return errors


def validate_payload(body: Any) -> tuple[list[dict], list[dict]]:
    """Return (records, errors). Accepts one record or {"instances": [...]}."""
    if not isinstance(body, dict):
        return [], [{"index": None, "errors": ["body must be a JSON object"]}]
    if "instances" in body:
        records = body["instances"]
        if set(body) != {"instances"}:
            return [], [{"index": None, "errors": ["'instances' cannot be mixed with other fields"]}]
        if not isinstance(records, list) or not records:
            return [], [{"index": None, "errors": ["'instances' must be a non-empty list"]}]
        if len(records) > MAX_INSTANCES:
            return [], [{"index": None, "errors": [f"at most {MAX_INSTANCES} instances per request"]}]
    else:
        records = [body]
    errors = []
    for i, record in enumerate(records):
        problems = validate_record(record)
        if problems:
            errors.append({"index": i, "errors": problems})
    return ([] if errors else records), errors


def to_dataframe(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records, columns=list(FEATURE_ORDER))
    return df.astype(
        {
            "SeniorCitizen": "int64",
            "tenure": "int64",
            "MonthlyCharges": "float64",
            "TotalCharges": "float64",
        }
    )