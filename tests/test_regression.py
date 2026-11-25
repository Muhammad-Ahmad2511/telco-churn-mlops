"""Regression guard for the selected, reproducible production candidate."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.features.preprocessing import load_config, load_processed, split_train_val_test
from src.models.evaluate import compute_metrics
from src.models.tune import build_pipeline

METRICS_PATH = Path("reports/metrics.json")
PROCESSED_PATH = Path("data/processed/telco_clean.parquet")
MAX_METRIC_DROP = 0.01


@pytest.mark.skipif(
    not METRICS_PATH.exists() or not PROCESSED_PATH.exists(),
    reason="DVC outputs or metrics baseline are unavailable",
)
def test_selected_model_does_not_regress_beyond_committed_baseline():
    """Rebuild the recorded winner and reject a material recall or F1 decrease.

    A small tolerance permits harmless library-level floating-point variation,
    while still detecting a changed split, preprocessing step, or model setting.
    """
    baseline = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    cfg = load_config()
    clean = load_processed(cfg)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(clean, cfg)
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    model = build_pipeline(
        X_trainval,
        baseline["selected_model"],
        baseline["selected_params"],
        cfg["random_state"],
    ).fit(X_trainval, y_trainval)
    actual = compute_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1])

    assert baseline["quality_gate"] == "passed"
    for metric in ("recall", "f1"):
        expected = baseline["test"][metric]
        assert actual[metric] >= expected - MAX_METRIC_DROP, (
            f"{metric} regressed: baseline={expected:.4f}, actual={actual[metric]:.4f}, "
            f"allowed_drop={MAX_METRIC_DROP:.4f}"
        )
