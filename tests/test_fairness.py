# tests/test_fairness.py
"""Per-group fairness metrics (spec §3.13). Pure functions, no network."""
import numpy as np
import pandas as pd

from src.monitoring.fairness import gaps, group_metrics


def test_group_metrics_known_values():
    # group A: 2 churners (1 caught), 2 loyal (1 wrongly flagged)
    # group B: 2 churners (both caught), 2 loyal (none flagged)
    y_true = pd.Series([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    groups = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])
    m = group_metrics(y_true, y_pred, groups)
    assert m["A"] == {"n": 4, "actual_churn_rate": 0.5, "selection_rate": 0.5,
                      "recall": 0.5, "false_positive_rate": 0.5, "precision": 0.5}
    assert m["B"]["recall"] == 1.0 and m["B"]["false_positive_rate"] == 0.0
    g = gaps(m)
    assert g["recall_gap"] == 0.5 and g["false_positive_rate_gap"] == 0.5


def test_equal_groups_have_zero_gaps():
    y_true = pd.Series([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    m = group_metrics(y_true, y_pred, pd.Series([0, 0, 1, 1]))
    assert all(v == 0 for v in gaps(m).values())


def test_undefined_rates_are_none_not_errors():
    # group with no churners: recall undefined; no predicted positives: precision undefined
    m = group_metrics(pd.Series([0, 0]), np.array([0, 0]), pd.Series(["X", "X"]))
    assert m["X"]["recall"] is None and m["X"]["precision"] is None
    assert gaps(m)["recall_gap"] is None
