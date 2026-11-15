# src/models/train.py
"""Baseline training with MLflow tracking (Person A).

Run from the repo root:
    python -m src.models.train
    python -m src.models.train --enforce-gate   # exit code 1 if the gate fails
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

from src.data.validation import validate_raw
from src.features.preprocessing import (
    build_preprocessor,
    clean_raw,
    load_config,
    split_data,
)
from src.models.evaluate import check_thresholds, compute_metrics

logger = logging.getLogger(__name__)


def git_info() -> tuple[str, bool]:
    """Return (commit hash, has uncommitted changes)."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True
    return commit, dirty


def dvc_md5(raw_path: str) -> str:
    """Data version = md5 stored in the .dvc pointer file."""
    meta = yaml.safe_load(Path(f"{raw_path}.dvc").read_text(encoding="utf-8"))
    return meta["outs"][0]["md5"]


def train(
    model_cfg_path: str = "configs/model_params.yaml",
    pipeline_cfg_path: str = "configs/pipeline_config.yaml",
) -> tuple[dict[str, float], bool]:
    load_dotenv()  # MLFLOW_* variables from the git-ignored .env
    data_cfg = load_config(pipeline_cfg_path)
    cfg = yaml.safe_load(Path(model_cfg_path).read_text(encoding="utf-8"))

    raw = validate_raw(pd.read_csv(data_cfg["raw_path"]))
    clean = clean_raw(raw, data_cfg)
    X_train, X_test, y_train, y_test = split_data(clean, data_cfg)

    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor(X_train)),
            (
                "model",
                LogisticRegression(**cfg["model"]["params"], random_state=data_cfg["random_state"]),
            ),
        ]
    )

    commit, dirty = git_info()
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())

    with mlflow.start_run(run_name=cfg["model"]["name"]):
        mlflow.set_tags(
            {
                "git_commit": commit,
                "git_dirty": str(dirty),
                "dvc_data_md5": dvc_md5(data_cfg["raw_path"]),
                "data_path": data_cfg["raw_path"],
            }
        )
        mlflow.log_params(
            {
                "model_type": cfg["model"]["name"],
                **cfg["model"]["params"],
                "test_size": data_cfg["test_size"],
                "random_state": data_cfg["random_state"],
                "dropped_columns": ",".join(data_cfg["drop_columns"]),
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
        )

        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        mlflow.log_metric("train_seconds", time.perf_counter() - start)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)

        passed, failures = check_thresholds(metrics, cfg["thresholds"])
        mlflow.set_tag("quality_gate", "passed" if passed else "failed")

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            pipeline, name="model", signature=signature, input_example=X_train.head(3)
        )

    for name, value in metrics.items():
        logger.info("%s = %.4f", name, value)
    logger.info("Quality gate: %s %s", "PASSED" if passed else "FAILED", failures)
    return metrics, passed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--enforce-gate", action="store_true")
    args = parser.parse_args()
    _, gate_passed = train()
    if args.enforce_gate and not gate_passed:
        sys.exit(1)