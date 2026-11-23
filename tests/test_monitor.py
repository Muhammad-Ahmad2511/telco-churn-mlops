# tests/test_monitor.py
"""Drift and data-quality checks (spec §3.12A). Synthetic data, no network."""
import json

import numpy as np
import pandas as pd
import pytest

from src.monitoring import monitor
from src.serving.schema import CATEGORICAL, FEATURE_ORDER, SAMPLE_RECORD

SETTINGS = {
    "drift": {"p_value_threshold": 0.05, "drift_share": 0.25, "min_rows": 100, "report_dir": "unused"},
    "simulation": {
        "sample_size": 300,
        "monthly_charges_factor": 1.3,
        "tenure_factor": 0.6,
        "shift_share": 0.4,
        "categorical_shift": {
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
            "PaymentMethod": "Electronic check",
        },
    },
}


def make_frame(n: int, seed: int) -> pd.DataFrame:
    """Valid customer rows drawn from one fixed distribution."""
    rng = np.random.default_rng(seed)
    data = {}
    for col in FEATURE_ORDER:
        if col in CATEGORICAL:
            data[col] = rng.choice(CATEGORICAL[col], n)
        elif col == "SeniorCitizen":
            data[col] = rng.integers(0, 2, n)
        elif col == "tenure":
            data[col] = rng.integers(0, 73, n)
        elif col == "MonthlyCharges":
            data[col] = rng.normal(65, 20, n).clip(18, 120).round(2)
    df = pd.DataFrame(data)
    df["TotalCharges"] = (df["MonthlyCharges"] * df["tenure"]).round(2)
    return df[list(FEATURE_ORDER)]


@pytest.fixture(scope="module")
def reference():
    return make_frame(3000, seed=1)


def test_same_distribution_is_not_dataset_drift(reference):
    result = monitor.check_drift(reference, make_frame(500, seed=2), SETTINGS)
    assert result["dataset_drift"] is False
    assert result["drifted_share"] < SETTINGS["drift"]["drift_share"]


def test_simulated_drift_is_detected_on_the_changed_columns(reference):
    current = monitor.simulate_drift(make_frame(500, seed=3), SETTINGS["simulation"])
    result = monitor.check_drift(reference, current, SETTINGS)
    assert result["dataset_drift"] is True
    expected = {"MonthlyCharges", "tenure", "TotalCharges", "Contract", "InternetService", "PaymentMethod"}
    assert expected <= set(result["drifted_columns"])


def test_simulated_drift_keeps_rows_valid_for_the_api():
    from src.serving.schema import validate_record

    drifted = monitor.simulate_drift(make_frame(200, seed=4), SETTINGS["simulation"])
    for record in drifted.head(50).to_dict(orient="records"):
        record = {k: (v.item() if hasattr(v, "item") else v) for k, v in record.items()}
        assert validate_record(record) == []


def test_numeric_and_categorical_tests_are_used(reference):
    result = monitor.check_drift(reference, make_frame(300, seed=5), SETTINGS)
    tests = {c["column"]: c["test"] for c in result["columns"]}
    assert tests["MonthlyCharges"] == "ks"
    assert tests["Contract"] == "chi2"
    assert len(result["columns"]) == len(FEATURE_ORDER)


def test_too_few_rows_is_refused(reference):
    with pytest.raises(ValueError, match="at least"):
        monitor.check_drift(reference, make_frame(20, seed=6), SETTINGS)


def test_data_quality_flags_unknown_category_missing_and_out_of_range():
    current = pd.DataFrame([dict(SAMPLE_RECORD)] * 5)
    current.loc[0, "Contract"] = "Weekly"
    current.loc[1, "MonthlyCharges"] = -5.0
    current.loc[2, "tenure"] = None
    issues = monitor.data_quality(current)
    assert issues["Contract"]["unknown_categories"] == ["Weekly"]
    assert issues["MonthlyCharges"]["out_of_range_share"] == 0.2
    assert issues["tenure"]["missing_share"] == 0.2


def test_clean_data_has_no_quality_issues():
    assert monitor.data_quality(pd.DataFrame([dict(SAMPLE_RECORD)] * 5)) == {}


def test_html_report_and_exit_code(reference, tmp_path, monkeypatch):
    current = monitor.simulate_drift(make_frame(400, seed=7), SETTINGS["simulation"])
    result = monitor.check_drift(reference, current, SETTINGS)
    page = tmp_path / "report.html"
    monitor.write_html(result, reference, current, page, "SIMULATED drift")
    text = page.read_text(encoding="utf-8")
    assert "DATASET DRIFT DETECTED" in text and "data:image/png;base64," in text
    json.dumps(result)  # report must be JSON-serialisable

    monkeypatch.setattr(monitor, "run", lambda *a, **k: result)
    assert monitor.main(["--simulate", "drift", "--exit-code"]) == monitor.DRIFT_EXIT_CODE
    assert monitor.main(["--simulate", "drift"]) == 0  # without --exit-code it only reports
