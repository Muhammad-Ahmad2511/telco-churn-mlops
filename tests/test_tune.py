# tests/test_tune.py
"""Tests for the train/val/test split, tuning grid, model selection and trained-model outputs."""
import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import clean_raw, split_data, split_train_val_test
from src.models.tune import (
    MODEL_TYPES,
    SKOPS_TRUSTED_TYPES,
    build_pipeline,
    candidate_grid,
    select_best,
)

CFG = {
    "target": "Churn",
    "drop_columns": ["customerID", "gender"],
    "test_size": 0.2,
    "val_size": 0.2,
    "random_state": 42,
}


def make_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(20, 110, n).round(2)
    return pd.DataFrame(
        {
            "customerID": [f"id-{i}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": (tenure * monthly).round(2).astype(str),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "Churn": ["Yes", "No", "No", "No"] * (n // 4),
        }
    )


# --- split -----------------------------------------------------------------

def test_three_way_split_sizes_and_no_overlap():
    clean = clean_raw(make_df(), CFG)
    X_train, X_val, X_test, *_ = split_train_val_test(clean, CFG)
    assert (len(X_train), len(X_val), len(X_test)) == (60, 20, 20)
    ids = [set(X.index) for X in (X_train, X_val, X_test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_test_set_matches_baseline_split():
    """Tuned models must be scored on the same test rows as the baseline."""
    clean = clean_raw(make_df(), CFG)
    _, X_test_baseline, _, _ = split_data(clean, CFG)
    _, _, X_test, *_ = split_train_val_test(clean, CFG)
    assert list(X_test.index) == list(X_test_baseline.index)


# --- grid and selection ------------------------------------------------------

def test_candidate_grid_expands_all_combinations():
    grid = candidate_grid(
        {"logistic_regression": {"C": [0.1, 1.0]}, "random_forest": {"max_depth": [4, 8], "min_samples_leaf": [1, 5]}}
    )
    assert len(grid) == 2 + 4
    assert ("logistic_regression", {"C": 0.1}) in grid


def _result(name, recall, f1):
    return {"model_type": name, "params": {}, "val_metrics": {"recall": recall, "f1": f1}}


def test_select_best_prefers_f1_among_eligible():
    results = [_result("a", 0.9, 0.50), _result("b", 0.7, 0.62), _result("c", 0.4, 0.70)]
    # c has the best F1 but misses the recall floor, so b wins.
    assert select_best(results, min_recall=0.6)["model_type"] == "b"


def test_select_best_falls_back_to_highest_recall():
    results = [_result("a", 0.3, 0.60), _result("b", 0.5, 0.40)]
    assert select_best(results, min_recall=0.6)["model_type"] == "b"


def test_select_best_empty_raises():
    with pytest.raises(ValueError):
        select_best([], min_recall=0.6)


def test_unknown_model_type_raises():
    clean = clean_raw(make_df(), CFG)
    X = clean.drop(columns=["Churn"])
    with pytest.raises(ValueError):
        build_pipeline(X, "svm", {}, 42)


# --- model output (spec §3.5 model tests) -----------------------------------

@pytest.mark.parametrize("model_type", sorted(MODEL_TYPES))
def test_trained_model_outputs_valid_predictions(model_type):
    clean = clean_raw(make_df(), CFG)
    X_train, _, X_test, y_train, _, _ = split_train_val_test(clean, CFG)
    pipe = build_pipeline(X_train, model_type, {}, 42).fit(X_train, y_train)

    labels = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)

    assert labels.shape == (len(X_test),)
    assert set(np.unique(labels)) <= {0, 1}
    assert proba.shape == (len(X_test), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("model_type", sorted(MODEL_TYPES))
def test_skops_trusted_types_cover_each_model(model_type):
    """MLflow 3 saves with skops; a missing trusted type makes log_model fail at the very end."""
    skops_io = pytest.importorskip("skops.io")
    clean = clean_raw(make_df(), CFG)
    X_train, _, _, y_train, _, _ = split_train_val_test(clean, CFG)
    pipe = build_pipeline(X_train, model_type, {}, 42).fit(X_train, y_train)
    needed = set(skops_io.get_untrusted_types(data=skops_io.dumps(pipe)))
    assert needed <= set(SKOPS_TRUSTED_TYPES[model_type])
