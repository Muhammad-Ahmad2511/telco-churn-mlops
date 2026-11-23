# src/models/promote.py
"""Controlled model promotion and rollback via MLflow registry aliases (Person A).

Aliases:  `candidate` = newest gate-passed model (set by register.py)
          `production` = the model the API serves (MODEL_ALIAS=production)

Run from the repo root (needs .env and the processed data: `dvc pull`):
    python -m src.models.promote init --version 1   # one-time: set the first production model
    python -m src.models.promote compare            # report only, changes nothing
    python -m src.models.promote promote            # compare, move `production` only if the rule passes
    python -m src.models.promote rollback           # move `production` back to the previous version
    python -m src.models.promote rollback --to-version 1

Promotion is a manual step (approval gate). Nothing promotes a model automatically.
Every decision is logged as an MLflow run (run_type=promotion/rollback) and as
model-version tags, so the history is auditable.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import yaml
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.features.preprocessing import load_config, load_processed, split_train_val_test
from src.models.evaluate import check_thresholds, compute_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Decision:
    promote: bool
    reasons: list[str]


def decide(
    candidate: dict[str, float],
    production: dict[str, float],
    candidate_gate_passed: bool,
    rule: dict,
) -> Decision:
    """Pure promotion rule (unit-tested). Metrics must come from the same test rows."""
    reasons = []
    if not candidate_gate_passed:
        reasons.append("candidate failed the quality gate")

    f1_needed = production["f1"] + rule["min_f1_gain"]
    if candidate["f1"] < f1_needed:
        reasons.append(
            f"f1 {candidate['f1']:.4f} < production {production['f1']:.4f} + {rule['min_f1_gain']}"
        )

    recall_floor = production["recall"] - rule["max_recall_drop"]
    if candidate["recall"] < recall_floor:
        reasons.append(
            f"recall {candidate['recall']:.4f} < production {production['recall']:.4f}"
            f" - {rule['max_recall_drop']}"
        )

    if reasons:
        return Decision(False, reasons)
    return Decision(
        True,
        [
            f"f1 {candidate['f1']:.4f} >= {f1_needed:.4f}",
            f"recall {candidate['recall']:.4f} >= {recall_floor:.4f}",
            "quality gate passed",
        ],
    )


def _settings(model_cfg_path: str) -> tuple[dict, str]:
    cfg = yaml.safe_load(Path(model_cfg_path).read_text(encoding="utf-8"))
    return cfg, cfg["registry"]["model_name"]


def _version_for_alias(client: MlflowClient, name: str, alias: str):
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def _test_set(pipeline_cfg_path: str):
    data_cfg = load_config(pipeline_cfg_path)
    _, _, X_test, _, _, y_test = split_train_val_test(load_processed(data_cfg), data_cfg)
    return X_test, y_test


def _score(name: str, version: str, X_test, y_test) -> dict[str, float]:
    model = mlflow.sklearn.load_model(f"models:/{name}/{version}")
    return compute_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1])


def init_production(version: str, model_cfg_path: str = "configs/model_params.yaml") -> None:
    """One-time bootstrap: mark an existing version as the first production model."""
    load_dotenv()
    cfg, name = _settings(model_cfg_path)
    client = MlflowClient()
    alias = cfg["promotion"]["production_alias"]
    current = _version_for_alias(client, name, alias)
    if current is not None:
        raise SystemExit(
            f"'{alias}' already points to version {current.version}. Use promote or rollback."
        )
    mv = client.get_model_version(name, version)
    if mv.tags.get("quality_gate") != "passed":
        raise SystemExit(f"Version {version} did not pass the quality gate.")
    client.set_registered_model_alias(name, alias, version)
    client.set_model_version_tag(name, version, "promotion_decision", "initial_production")
    logger.info("Set %s@%s -> version %s", name, alias, version)


def compare_and_maybe_promote(
    apply: bool,
    model_cfg_path: str = "configs/model_params.yaml",
    pipeline_cfg_path: str = "configs/pipeline_config.yaml",
) -> Decision:
    load_dotenv()
    cfg, name = _settings(model_cfg_path)
    rule = cfg["promotion"]
    client = MlflowClient()

    cand = _version_for_alias(client, name, rule["candidate_alias"])
    prod = _version_for_alias(client, name, rule["production_alias"])
    if cand is None:
        raise SystemExit(f"No '{rule['candidate_alias']}' alias. Run src.models.register first.")
    if prod is None:
        raise SystemExit(f"No '{rule['production_alias']}' alias. Run: promote init --version <v>")
    if cand.version == prod.version:
        logger.info("Candidate and production are both version %s. Nothing to do.", cand.version)
        return Decision(False, ["candidate is already production"])

    X_test, y_test = _test_set(pipeline_cfg_path)
    cand_metrics = _score(name, cand.version, X_test, y_test)
    prod_metrics = _score(name, prod.version, X_test, y_test)
    gate_ok = (
        cand.tags.get("quality_gate") == "passed"
        and check_thresholds(cand_metrics, cfg["thresholds"])[0]
    )
    decision = decide(cand_metrics, prod_metrics, gate_ok, rule)

    logger.info("%-10s %-9s %-9s %-9s", "metric", f"prod v{prod.version}", f"cand v{cand.version}", "diff")
    for key in ("recall", "f1", "precision", "roc_auc", "accuracy"):
        logger.info(
            "%-10s %-9.4f %-9.4f %+.4f",
            key, prod_metrics[key], cand_metrics[key], cand_metrics[key] - prod_metrics[key],
        )
    verdict = "PROMOTE" if decision.promote else "REJECT"
    logger.info("Decision: %s (%s)", verdict, "; ".join(decision.reasons))

    if not apply:
        logger.info("compare only: nothing changed. Run `promote` to apply.")
        return decision

    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=f"promotion v{cand.version} vs v{prod.version}"):
        mlflow.set_tags(
            {
                "run_type": "promotion",
                "decision": verdict.lower(),
                "candidate_version": cand.version,
                "production_version": prod.version,
            }
        )
        mlflow.log_params({k: rule[k] for k in ("min_f1_gain", "max_recall_drop")})
        mlflow.log_metrics({f"candidate_{k}": v for k, v in cand_metrics.items()})
        mlflow.log_metrics({f"production_{k}": v for k, v in prod_metrics.items()})
        mlflow.log_text("\n".join(decision.reasons), "decision.txt")

    client.set_model_version_tag(name, cand.version, "promotion_decision",
                                 "promoted" if decision.promote else "rejected")
    client.set_model_version_tag(name, cand.version, "promotion_reason",
                                 "; ".join(decision.reasons)[:250])
    if decision.promote:
        client.set_model_version_tag(name, cand.version, "promoted_from", prod.version)
        client.set_registered_model_alias(name, rule["production_alias"], cand.version)
        logger.info("%s@%s: version %s -> %s", name, rule["production_alias"], prod.version, cand.version)
    return decision


def rollback(to_version: str | None = None, model_cfg_path: str = "configs/model_params.yaml") -> str:
    """Move `production` back to the version it was promoted from (or to --to-version)."""
    load_dotenv()
    cfg, name = _settings(model_cfg_path)
    alias = cfg["promotion"]["production_alias"]
    client = MlflowClient()
    current = _version_for_alias(client, name, alias)
    if current is None:
        raise SystemExit(f"No '{alias}' alias to roll back.")
    target = to_version or current.tags.get("promoted_from")
    if not target:
        raise SystemExit(
            f"Version {current.version} has no 'promoted_from' tag. Use --to-version <v>."
        )
    if target == current.version:
        raise SystemExit(f"Production is already version {target}.")
    client.get_model_version(name, target)  # fails clearly if the version does not exist

    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=f"rollback v{current.version} -> v{target}"):
        mlflow.set_tags(
            {"run_type": "rollback", "from_version": current.version, "to_version": target}
        )
    client.set_registered_model_alias(name, alias, target)
    client.set_model_version_tag(name, current.version, "promotion_decision", "rolled_back")
    logger.info("Rolled back %s@%s: version %s -> %s", name, alias, current.version, target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote or roll back the production model.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_init = sub.add_parser("init", help="set the first production version")
    p_init.add_argument("--version", required=True)
    sub.add_parser("compare", help="compare candidate vs production, change nothing")
    sub.add_parser("promote", help="compare and promote if the rule passes")
    p_rb = sub.add_parser("rollback", help="move production back")
    p_rb.add_argument("--to-version", default=None)
    args = parser.parse_args(argv)

    if args.command == "init":
        init_production(args.version)
    elif args.command in ("compare", "promote"):
        decision = compare_and_maybe_promote(apply=args.command == "promote")
        if args.command == "promote" and not decision.promote:
            return 1  # non-zero so CI/Airflow can see a rejected candidate
    else:
        rollback(args.to_version)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
