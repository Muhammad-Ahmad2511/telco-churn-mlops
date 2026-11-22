# src/models/tune.py
"""Hyperparameter tuning and model comparison with MLflow (Person A).

Trains every candidate from configs/model_params.yaml -> tuning on the train split,
scores it on the validation split (one nested MLflow run each), picks the winner,
refits it on train + validation, scores it once on the untouched test split and
applies the quality gate.

Run from the repo root (normally via `dvc repro`, stage `tune`):
    python -m src.models.tune
    python -m src.models.tune --enforce-gate   # exit code 1 if the gate fails
Then register the winner with:  python -m src.models.register
"""
from __future__ import annotations

import argparse
import json
import logging
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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from src.features.preprocessing import (
    build_preprocessor,
    load_config,
    load_processed,
    split_train_val_test,
)
from src.models.evaluate import check_thresholds, compute_metrics
from src.models.train import dvc_md5, git_info

logger = logging.getLogger(__name__)

# Fixed settings per model type. Tuned values come from the config grid.
# class_weight="balanced" because only ~27% of customers churn.
MODEL_TYPES = {
    "logistic_regression": lambda rs, **p: LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=rs, **p
    ),
    "random_forest": lambda rs, **p: RandomForestClassifier(
        class_weight="balanced", n_jobs=-1, random_state=rs, **p
    ),
    "hist_gradient_boosting": lambda rs, **p: HistGradientBoostingClassifier(
        class_weight="balanced", random_state=rs, **p
    ),
}

# MLflow 3 saves sklearn models with skops, which rejects tree internals unless they are
# explicitly trusted. Safe here: we train and save these models ourselves.
# Only the exact type each family needs is listed (checked with skops.io.get_untrusted_types).
SKOPS_TRUSTED_TYPES = {
    "logistic_regression": [],
    "random_forest": ["sklearn.tree._tree.Tree"],
    "hist_gradient_boosting": ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"],
}


def build_pipeline(X: pd.DataFrame, model_type: str, params: dict, random_state: int) -> Pipeline:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type '{model_type}'. Known: {sorted(MODEL_TYPES)}")
    return Pipeline(
        [
            ("preprocess", build_preprocessor(X)),
            ("model", MODEL_TYPES[model_type](random_state, **params)),
        ]
    )


def candidate_grid(candidates: dict) -> list[tuple[str, dict]]:
    """Expand {model_type: {param: [values]}} into (model_type, params) pairs."""
    return [
        (model_type, dict(params))
        for model_type, grid in candidates.items()
        for params in ParameterGrid(grid)
    ]


def select_best(results: list[dict], min_recall: float, metric: str = "f1") -> dict:
    """Best validation `metric` among candidates with recall >= min_recall.

    If no candidate reaches min_recall, fall back to the highest recall
    (the quality gate on the test set will then decide).
    """
    if not results:
        raise ValueError("No tuning results to select from.")
    eligible = [r for r in results if r["val_metrics"]["recall"] >= min_recall]
    if eligible:
        return max(eligible, key=lambda r: r["val_metrics"][metric])
    logger.warning("No candidate reached recall >= %.2f; choosing highest recall.", min_recall)
    return max(results, key=lambda r: r["val_metrics"]["recall"])


def _log_plots(y_true, y_pred, y_proba) -> None:
    for name, draw in (
        ("confusion_matrix.png", lambda ax: ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)),
        ("roc_curve.png", lambda ax: RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)),
        ("pr_curve.png", lambda ax: PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax)),
    ):
        fig, ax = plt.subplots()
        draw(ax)
        mlflow.log_figure(fig, name)
        plt.close(fig)


def write_reports(report_dir: str, run_id: str, best: dict, metrics: dict, passed: bool,
                  comparison: pd.DataFrame) -> None:
    """Outputs of the DVC `tune` stage (`dvc metrics show` reads metrics.json)."""
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "mlflow_run_id": run_id,
        "selected_model": best["model_type"],
        "selected_params": best["params"],
        "quality_gate": "passed" if passed else "failed",
        "test": {k: round(v, 4) for k, v in metrics.items()},
        "best_val": {k: round(v, 4) for k, v in best["val_metrics"].items()},
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    comparison.to_csv(out / "model_comparison.csv", index=False)


def tune(
    model_cfg_path: str = "configs/model_params.yaml",
    pipeline_cfg_path: str = "configs/pipeline_config.yaml",
    report_dir: str = "reports",
) -> tuple[dict, dict[str, float], bool]:
    load_dotenv()
    data_cfg = load_config(pipeline_cfg_path)
    cfg = yaml.safe_load(Path(model_cfg_path).read_text(encoding="utf-8"))
    tuning = cfg["tuning"]
    thresholds = cfg["thresholds"]
    rs = data_cfg["random_state"]

    # Raw data was validated by the DVC `validate` stage; preprocess wrote the Parquet file.
    clean = load_processed(data_cfg)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(clean, data_cfg)

    commit, dirty = git_info()
    lineage = {
        "git_commit": commit,
        "git_dirty": str(dirty),
        "dvc_data_md5": dvc_md5(data_cfg["raw_path"]),
        "data_path": data_cfg["processed_path"],
    }
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())

    with mlflow.start_run(run_name="tuning") as parent:
        mlflow.set_tags({**lineage, "run_type": "tuning"})
        mlflow.log_params(
            {
                "selection_metric": tuning["selection_metric"],
                "min_recall": thresholds["min_recall"],
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
                "dropped_columns": ",".join(data_cfg["drop_columns"]),
                "random_state": rs,
            }
        )

        # 1. Train each candidate on train, score on validation.
        results = []
        for model_type, params in candidate_grid(tuning["candidates"]):
            with mlflow.start_run(run_name=model_type, nested=True):
                mlflow.set_tags({**lineage, "run_type": "candidate"})
                mlflow.log_params({"model_type": model_type, **params})
                pipe = build_pipeline(X_train, model_type, params, rs)
                start = time.perf_counter()
                pipe.fit(X_train, y_train)
                seconds = time.perf_counter() - start
                val_metrics = compute_metrics(
                    y_val, pipe.predict(X_val), pipe.predict_proba(X_val)[:, 1]
                )
                mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
                mlflow.log_metric("train_seconds", seconds)
            results.append({"model_type": model_type, "params": params, "val_metrics": val_metrics})
            logger.info(
                "%-24s %-55s val_recall=%.3f val_f1=%.3f",
                model_type, params, val_metrics["recall"], val_metrics["f1"],
            )

        comparison = pd.DataFrame(
            [
                {"model_type": r["model_type"], "params": str(r["params"]),
                 **{f"val_{k}": round(v, 4) for k, v in r["val_metrics"].items()}}
                for r in results
            ]
        ).sort_values(f"val_{tuning['selection_metric']}", ascending=False)
        mlflow.log_table(comparison, "model_comparison.json")
        logger.info("Model comparison (validation):\n%s", comparison.to_string(index=False))

        # 2. Refit the winner on train + validation and score once on test.
        best = select_best(results, thresholds["min_recall"], tuning["selection_metric"])
        X_trainval = pd.concat([X_train, X_val])
        y_trainval = pd.concat([y_train, y_val])
        final = build_pipeline(X_trainval, best["model_type"], best["params"], rs)
        start = time.perf_counter()
        final.fit(X_trainval, y_trainval)
        mlflow.log_metric("train_seconds", time.perf_counter() - start)

        y_pred = final.predict(X_test)
        y_proba = final.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)
        mlflow.log_metrics({f"best_val_{k}": v for k, v in best["val_metrics"].items()})

        passed, failures = check_thresholds(metrics, thresholds)
        mlflow.set_tags(
            {"quality_gate": "passed" if passed else "failed", "selected_model": best["model_type"]}
        )
        mlflow.log_params({"model_type": best["model_type"], **{f"best_{k}": v for k, v in best["params"].items()}})

        _log_plots(y_test, y_pred, y_proba)
        signature = infer_signature(X_trainval, final.predict(X_trainval))
        mlflow.sklearn.log_model(
            final,
            name="model",
            signature=signature,
            input_example=X_trainval.head(3),
            skops_trusted_types=SKOPS_TRUSTED_TYPES[best["model_type"]] or None,
        )

    write_reports(report_dir, parent.info.run_id, best, metrics, passed, comparison)
    logger.info("Selected: %s %s", best["model_type"], best["params"])
    for name, value in metrics.items():
        logger.info("test %s = %.4f", name, value)
    logger.info("Quality gate: %s %s", "PASSED" if passed else "FAILED", failures)
    return best, metrics, passed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--enforce-gate", action="store_true")
    args = parser.parse_args()
    _, _, gate_passed = tune()
    if args.enforce_gate and not gate_passed:
        sys.exit(1)
