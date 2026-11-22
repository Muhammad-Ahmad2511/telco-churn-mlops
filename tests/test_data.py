# tests/test_data.py
"""Data tests for raw Telco churn validation (Person A)."""
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import pytest

from src.data.validation import validate_raw

RAW_CSV = Path("data/raw/telco_churn.csv")


def make_valid_df() -> pd.DataFrame:
    """Two valid rows. The second has the known blank TotalCharges quirk."""
    return pd.DataFrame(
        {
            "customerID": ["0001-A", "0002-B"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [12, 0],
            "PhoneService": ["Yes", "No"],
            "MultipleLines": ["No", "No phone service"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["Yes", "No"],
            "OnlineBackup": ["No", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "No internet service"],
            "StreamingMovies": ["No", "No internet service"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [29.85, 56.95],
            "TotalCharges": ["358.2", " "],
            "Churn": ["No", "No"],
        }
    )


def test_valid_sample_passes():
    validate_raw(make_valid_df())


def test_blank_total_charges_is_allowed_at_raw_stage():
    df = make_valid_df()
    assert (df["TotalCharges"].str.strip() == "").sum() == 1
    validate_raw(df)


def test_missing_column_fails():
    df = make_valid_df().drop(columns=["Contract"])
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(df)


def test_extra_column_fails():
    df = make_valid_df()
    df["surprise"] = 1
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(df)


def test_invalid_category_fails():
    df = make_valid_df()
    df.loc[0, "Contract"] = "Weekly"
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(df)


def test_non_numeric_total_charges_fails():
    df = make_valid_df()
    df.loc[0, "TotalCharges"] = "abc"
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(df)


def test_negative_tenure_fails():
    df = make_valid_df()
    df.loc[0, "tenure"] = -5
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(df)


def test_duplicate_customer_id_fails():
    df = make_valid_df()
    df.loc[1, "customerID"] = df.loc[0, "customerID"]
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw(df)


@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw CSV not pulled (dvc pull)")
def test_real_dataset_passes_and_has_expected_shape():
    df = pd.read_csv(RAW_CSV)
    assert df.shape == (7043, 21)
    validate_raw(df)

def test_validation_report_written(tmp_path):
    import json

    from src.data.validation import write_report

    report = tmp_path / "reports" / "validation.json"
    write_report(make_valid_df(), "sample.csv", str(report))
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["passed"] is True
    assert (data["rows"], data["columns"]) == (2, 21)
    assert data["churn_rate"] == 0.0
