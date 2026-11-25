"""End-to-end local integration coverage for the churn prediction path.

This intentionally avoids MLflow and the network. It verifies the boundary
between the real DVC dataset, preprocessing, a fitted model, and Flask serving.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.validation import validate_raw
from src.features.preprocessing import (
    build_preprocessor,
    clean_raw,
    load_config,
    split_data,
)
from src.serving.app import create_app
from src.serving.model_loader import ModelBundle, ModelStore

RAW_CSV = Path("data/raw/telco_churn.csv")


@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw CSV not pulled (dvc pull)")
def test_raw_data_to_fitted_model_to_api_prediction():
    """A real customer row remains valid from the raw dataset through serving."""
    cfg = load_config()
    raw = validate_raw(pd.read_csv(RAW_CSV))
    clean = clean_raw(raw, cfg)
    X_train, X_test, y_train, _ = split_data(clean, cfg)

    model = Pipeline(
        [
            ("preprocess", build_preprocessor(X_train)),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=cfg["random_state"],
                ),
            ),
        ]
    ).fit(X_train, y_train)
    bundle = ModelBundle(
        model=model,
        name="integration-test-model",
        version="local",
        alias="test",
        run_id=None,
        tags={"quality_gate": "passed"},
    )
    client = create_app(ModelStore(loader=lambda: bundle)).test_client()

    record = X_test.iloc[0].to_dict()
    response = client.post("/predict", json=record)

    assert response.status_code == 200
    prediction = response.get_json()["predictions"]
    assert len(prediction) == 1
    assert prediction[0]["churn"] in (0, 1)
    assert 0.0 <= prediction[0]["churn_probability"] <= 1.0
