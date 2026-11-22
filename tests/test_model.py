# tests/test_model.py
"""Model tests: metrics and quality gate (Person A)."""
import pytest

from src.models.evaluate import check_thresholds, compute_metrics


def test_compute_metrics_perfect_prediction():
    m = compute_metrics([0, 1, 1, 0], [0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    assert m["recall"] == 0.5
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0


def test_gate_passes_when_metrics_meet_thresholds():
    ok, failures = check_thresholds(
        {"recall": 0.7, "f1": 0.6}, {"min_recall": 0.6, "min_f1": 0.55}
    )
    assert ok and failures == []


def test_gate_fails_when_metric_below_threshold():
    ok, failures = check_thresholds(
        {"recall": 0.4, "f1": 0.6}, {"min_recall": 0.6, "min_f1": 0.55}
    )
    assert not ok
    assert len(failures) == 1 and "recall" in failures[0]


def test_gate_unknown_metric_raises():
    with pytest.raises(KeyError):
        check_thresholds({"recall": 0.7}, {"min_auc_pr": 0.5})