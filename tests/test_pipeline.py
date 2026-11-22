# tests/test_pipeline.py
"""Unit tests for preprocessing (Person A)."""
import pandas as pd
import pytest

from src.features.preprocessing import (
    build_preprocessor,
    clean_raw,
    drop_unused,
    fix_types,
    load_processed,
    split_data,
)

CFG = {
    "target": "Churn",
    "drop_columns": ["customerID", "gender"],
    "test_size": 0.25,
    "random_state": 42,
}


def make_df(n: int = 40) -> pd.DataFrame:
    """Small frame: first row is the blank-TotalCharges, tenure 0 quirk."""
    return pd.DataFrame(
        {
            "customerID": [f"id-{i}" for i in range(n)],
            "gender": ["Male", "Female"] * (n // 2),
            "tenure": [0] + [10 + i for i in range(n - 1)],
            "MonthlyCharges": [20.0 + i for i in range(n)],
            "TotalCharges": [" "] + [str(100.0 + i) for i in range(n - 1)],
            "Contract": ["Month-to-month", "Two year"] * (n // 2),
            "Churn": ["No", "Yes"] * (n // 2),
        }
    )


def test_blank_total_charges_becomes_zero():
    out = clean_raw(make_df(), CFG)
    assert out["TotalCharges"].dtype == "float64"
    assert out.loc[0, "TotalCharges"] == 0.0
    assert not out["TotalCharges"].isna().any()


def test_blank_total_charges_with_tenure_raises():
    df = make_df()
    df.loc[5, "TotalCharges"] = " "  # tenure is 14 here, so this is not the known quirk
    with pytest.raises(ValueError):
        clean_raw(df, CFG)


def test_target_is_encoded_as_0_1():
    out = clean_raw(make_df(), CFG)
    assert set(out["Churn"].unique()) == {0, 1}


def test_id_and_sensitive_columns_dropped():
    out = clean_raw(make_df(), CFG)
    assert "customerID" not in out.columns
    assert "gender" not in out.columns


def test_split_is_stratified_and_sized():
    out = clean_raw(make_df(), CFG)
    X_train, X_test, y_train, y_test = split_data(out, CFG)
    assert len(X_test) == 10
    assert len(X_train) == 30
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.1)


def test_preprocessor_ignores_unseen_category():
    out = clean_raw(make_df(), CFG)
    X_train, _, _, _ = split_data(out, CFG)
    pre = build_preprocessor(X_train).fit(X_train)
    new = X_train.head(1).copy()
    new["Contract"] = "Weekly"  # never seen in training
    assert pre.transform(new).shape[0] == 1

def test_fix_types_keeps_id_and_gender_for_feature_store_and_fairness():
    out = fix_types(make_df(), CFG)
    assert {"customerID", "gender"} <= set(out.columns)
    assert out["TotalCharges"].dtype == "float64"
    assert set(out["Churn"].unique()) == {0, 1}


def test_clean_raw_equals_fix_types_then_drop_unused():
    df = make_df()
    pd.testing.assert_frame_equal(clean_raw(df, CFG), drop_unused(fix_types(df, CFG), CFG))


def test_processed_parquet_roundtrip(tmp_path):
    """preprocess stage output -> load_processed gives the same frame as clean_raw."""
    cfg = {**CFG, "processed_path": str(tmp_path / "clean.parquet")}
    fix_types(make_df(), cfg).to_parquet(cfg["processed_path"], index=False)
    pd.testing.assert_frame_equal(load_processed(cfg), clean_raw(make_df(), cfg))


def test_load_processed_missing_file_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="dvc repro"):
        load_processed({**CFG, "processed_path": str(tmp_path / "missing.parquet")})
