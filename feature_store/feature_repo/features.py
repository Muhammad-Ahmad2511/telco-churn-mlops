"""Feature catalogue for the churn model (spec 12.1).

Plain Python with no Feast import, so the data-quality code and its tests can
use it from the main project environment (pandas 3) while the Feast-specific
files run in a separate environment.

Documentation of every feature
------------------------------
source:      data/processed/telco_clean.parquet (DVC-tracked pipeline output)
transform:   columns are copied unchanged from the processed data, so the
             offline and online stores hold identical values (checked by the
             skew test). Encoding is done separately by
             src/features/preprocessing.py.
freshness:   rebuild the table when the processed data changes (a manual step
             for now: python -m feature_store.build_features); a snapshot
             older than MAX_AGE_DAYS is reported as stale by the quality checks.
owner:       data-ml-engineering (Student 1 in the project specification).
excluded:    customerID is only the entity key; gender is a sensitive attribute
             and is never a feature; Churn is the label, not a feature.
"""

from __future__ import annotations

ENTITY_KEY = "customerID"
EVENT_TIMESTAMP = "event_timestamp"
FEATURE_OWNER = "data-ml-engineering"

# name -> data type ("int64", "float64" or "string")
ACCOUNT_FEATURES: dict[str, str] = {
    "SeniorCitizen": "int64",
    "Partner": "string",
    "Dependents": "string",
    "tenure": "int64",
    "Contract": "string",
    "PaperlessBilling": "string",
    "PaymentMethod": "string",
    "MonthlyCharges": "float64",
    "TotalCharges": "float64",
}

SERVICE_FEATURES: dict[str, str] = {
    "PhoneService": "string",
    "MultipleLines": "string",
    "InternetService": "string",
    "OnlineSecurity": "string",
    "OnlineBackup": "string",
    "DeviceProtection": "string",
    "TechSupport": "string",
    "StreamingTV": "string",
    "StreamingMovies": "string",
}

FEATURE_COLUMNS: dict[str, str] = {**ACCOUNT_FEATURES, **SERVICE_FEATURES}

# Data-quality expectations (tests and build_features.py use these).
MAX_NULL_FRACTION = 0.01  # at most 1% missing values per feature
MAX_AGE_DAYS = 30  # freshness expectation for the feature snapshot
NON_NEGATIVE = ("tenure", "MonthlyCharges", "TotalCharges")
