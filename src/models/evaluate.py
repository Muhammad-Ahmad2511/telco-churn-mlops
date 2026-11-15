# src/models/evaluate.py
"""Evaluation metrics and the quality gate (Person A)."""
from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Metrics for the positive class (Churn = 1)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def check_thresholds(
    metrics: dict[str, float], thresholds: dict[str, float]
) -> tuple[bool, list[str]]:
    """Keys look like 'min_recall'. Returns (passed, list of failure messages)."""
    failures = []
    for key, minimum in thresholds.items():
        name = key.removeprefix("min_")
        if metrics[name] < minimum:
            failures.append(f"{name}={metrics[name]:.3f} is below minimum {minimum}")
    return (not failures, failures)