# src/monitoring/fairness.py
"""Fairness and explainability check for registered models (spec §3.13).

Scores each model on the same held-out test rows and compares, per group of a
sensitive attribute:
  recall (TPR)        - share of real churners the model catches
  false positive rate - share of loyal customers wrongly flagged
  precision, selection rate, actual churn rate
`gender` is NOT a model input (dropped in preprocessing), but bias can still enter
through correlated columns, so it is checked. `SeniorCitizen` IS a model input.

Explainability: permutation importance on the test set (drop in F1 when one raw
input column is shuffled).

    python -m src.monitoring.fairness                 # production + v1 for comparison
    python -m src.monitoring.fairness --versions 2    # specific versions
Writes reports/fairness.json (needs .env for the MLflow registry and `dvc pull`).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow import MlflowClient
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

from src.features.preprocessing import load_config, split_train_val_test
from src.serving.model_loader import single_threaded
from src.serving.schema import FEATURE_ORDER

logger = logging.getLogger(__name__)

SENSITIVE = ("gender", "SeniorCitizen")


def group_metrics(y_true: pd.Series, y_pred: np.ndarray, groups: pd.Series) -> dict[str, dict]:
    """Per-group confusion-matrix metrics. Pure function (unit-tested)."""
    out = {}
    for value in sorted(groups.unique(), key=str):
        mask = (groups == value).to_numpy()
        tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1]).ravel()
        out[str(value)] = {
            "n": int(mask.sum()),
            "actual_churn_rate": round(float((tp + fn) / mask.sum()), 4),
            "selection_rate": round(float((tp + fp) / mask.sum()), 4),
            "recall": round(float(tp / (tp + fn)), 4) if tp + fn else None,
            "false_positive_rate": round(float(fp / (fp + tn)), 4) if fp + tn else None,
            "precision": round(float(tp / (tp + fp)), 4) if tp + fp else None,
        }
    return out


def gaps(per_group: dict[str, dict]) -> dict[str, float | None]:
    """Largest difference between groups for each rate (0 = perfectly equal)."""
    result = {}
    for key in ("recall", "false_positive_rate", "precision", "selection_rate"):
        values = [g[key] for g in per_group.values() if g[key] is not None]
        result[f"{key}_gap"] = round(max(values) - min(values), 4) if len(values) > 1 else None
    return result


def test_rows(pipeline_cfg_path: str = "configs/pipeline_config.yaml") -> tuple[pd.DataFrame, pd.Series]:
    """Held-out test rows WITH gender kept (same rows as training/promotion use)."""
    cfg = load_config(pipeline_cfg_path)
    df = pd.read_parquet(cfg["processed_path"]).drop(columns=["customerID"])
    _, _, X_test, _, _, y_test = split_train_val_test(df, cfg)
    return X_test, y_test


def evaluate_version(name: str, version: str, X_test: pd.DataFrame, y_test: pd.Series,
                     importance: bool) -> dict:
    model = single_threaded(mlflow.sklearn.load_model(f"models:/{name}/{version}"))
    features = X_test[list(FEATURE_ORDER)]
    y_pred = model.predict(features)
    result = {
        "version": version,
        "model_type": type(model.named_steps["model"]).__name__,
        "groups": {},
    }
    for attr in SENSITIVE:
        per_group = group_metrics(y_test, y_pred, X_test[attr])
        result["groups"][attr] = {"per_group": per_group, **gaps(per_group)}
    if importance:
        pi = permutation_importance(model, features, y_test, scoring="f1", n_repeats=5,
                                    random_state=42, n_jobs=1)
        ranked = sorted(zip(FEATURE_ORDER, pi.importances_mean, pi.importances_std, strict=True),
                        key=lambda t: t[1], reverse=True)
        result["permutation_importance_f1"] = [
            {"feature": f, "mean": round(float(m), 4), "std": round(float(s), 4)} for f, m, s in ranked
        ]
    return result


def run(versions: list[str] | None, out_path: str = "reports/fairness.json",
        model_cfg_path: str = "configs/model_params.yaml") -> dict:
    load_dotenv()
    cfg = yaml.safe_load(Path(model_cfg_path).read_text(encoding="utf-8"))
    name = cfg["registry"]["model_name"]
    if not versions:
        prod = MlflowClient().get_model_version_by_alias(name, cfg["promotion"]["production_alias"]).version
        versions = sorted({str(prod), "1"})
    X_test, y_test = test_rows()
    report = {
        "model_name": name,
        "test_rows": len(X_test),
        "note": "gender is not a model input; SeniorCitizen is",
        "versions": [evaluate_version(name, v, X_test, y_test, importance=True) for v in versions],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for v in report["versions"]:
        for attr, data in v["groups"].items():
            for group, m in data["per_group"].items():
                logger.info("v%s %-13s %-7s n=%4d churn=%.3f recall=%.3f fpr=%.3f precision=%.3f",
                            v["version"], attr, group, m["n"], m["actual_churn_rate"], m["recall"],
                            m["false_positive_rate"], m["precision"])
            logger.info("v%s %-13s gaps: %s", v["version"], attr,
                        {k: data[k] for k in data if k.endswith("_gap")})
        top = ", ".join(f"{p['feature']} {p['mean']:.3f}" for p in v["permutation_importance_f1"][:5])
        logger.info("v%s top features (F1 drop when shuffled): %s", v["version"], top)
    logger.info("Wrote %s", out_path)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fairness and explainability check.")
    parser.add_argument("--versions", nargs="*", help="registry versions (default: production and 1)")
    run(parser.parse_args().versions)
