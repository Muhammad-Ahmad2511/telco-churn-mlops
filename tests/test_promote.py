# tests/test_promote.py
"""Tests for the promotion rule (no network: pure function + CLI parsing)."""
from pathlib import Path

import pytest
import yaml

from src.models.promote import decide, main

RULE = {"min_f1_gain": 0.005, "max_recall_drop": 0.02}
PROD = {"recall": 0.783, "f1": 0.613}


def test_promotes_when_f1_improves_and_recall_holds():
    # Real numbers: v2 Random Forest vs v1 Logistic Regression on the same test rows.
    d = decide({"recall": 0.7727, "f1": 0.6289}, PROD, True, RULE)
    assert d.promote


def test_rejects_when_f1_gain_too_small():
    d = decide({"recall": 0.79, "f1": 0.615}, PROD, True, RULE)
    assert not d.promote and any("f1" in r for r in d.reasons)


def test_rejects_when_recall_drops_too_much():
    d = decide({"recall": 0.70, "f1": 0.70}, PROD, True, RULE)
    assert not d.promote and any("recall" in r for r in d.reasons)


def test_rejects_when_gate_failed_even_if_metrics_better():
    d = decide({"recall": 0.90, "f1": 0.80}, PROD, False, RULE)
    assert not d.promote and any("quality gate" in r for r in d.reasons)


def test_boundaries_are_inclusive():
    d = decide({"recall": 0.783 - 0.02, "f1": 0.613 + 0.005}, PROD, True, RULE)
    assert d.promote


def test_all_failures_are_reported():
    d = decide({"recall": 0.5, "f1": 0.5}, PROD, False, RULE)
    assert len(d.reasons) == 3


def test_promotion_config_present():
    cfg = yaml.safe_load(Path("configs/model_params.yaml").read_text(encoding="utf-8"))
    promo = cfg["promotion"]
    assert promo["candidate_alias"] != promo["production_alias"]
    assert promo["min_f1_gain"] >= 0 and promo["max_recall_drop"] >= 0


def test_cli_requires_a_command():
    with pytest.raises(SystemExit):
        main([])


def test_cli_init_requires_version():
    with pytest.raises(SystemExit):
        main(["init"])
