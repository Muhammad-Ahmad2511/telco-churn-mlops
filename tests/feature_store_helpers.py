"""Shared helpers for the feature tests (synthetic data with the real schema)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEATURE_REPO = ROOT / "feature_store" / "feature_repo"


def load_build_features():
    return importlib.import_module("feature_store.build_features")


def load_feature_catalogue():
    return importlib.import_module("feature_store.feature_repo.features")


def load_materialize():
    return importlib.import_module("feature_store.materialize")


def make_processed(n: int = 50, seed: int = 0) -> pd.DataFrame:
    """A small frame with the same columns and value types as the processed data."""
    rng = np.random.default_rng(seed)
    yes_no = ["Yes", "No"]
    internet = ["Yes", "No", "No internet service"]
    frame = pd.DataFrame(
        {
            "customerID": [f"C{i:05d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(yes_no, n),
            "Dependents": rng.choice(yes_no, n),
            "tenure": rng.integers(0, 72, n),
            "PhoneService": rng.choice(yes_no, n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(internet, n),
            "OnlineBackup": rng.choice(internet, n),
            "DeviceProtection": rng.choice(internet, n),
            "TechSupport": rng.choice(internet, n),
            "StreamingTV": rng.choice(internet, n),
            "StreamingMovies": rng.choice(internet, n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(yes_no, n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)"], n
            ),
            "MonthlyCharges": rng.uniform(18, 120, n).round(2),
        }
    )
    frame["TotalCharges"] = (frame["tenure"] * frame["MonthlyCharges"]).round(2)
    frame["Churn"] = rng.integers(0, 2, n)
    return frame
