# src/models/register.py
"""Register a clean, gate-passed run in the MLflow Model Registry (Person A).

Run from the repo root:
    python -m src.models.register               # latest eligible run
    python -m src.models.register --run-id <id> # a specific run

Registering is NOT deployment. Promotion to "production" is a manual approval gate.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow
import yaml
from dotenv import load_dotenv
from mlflow import MlflowClient

logger = logging.getLogger(__name__)


def find_run(experiment_name: str, run_id: str | None):
    if run_id:
        return mlflow.get_run(run_id)
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="tags.quality_gate = 'passed' and tags.git_dirty = 'False'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        raise SystemExit("No run with quality_gate=passed and git_dirty=False found.")
    return runs[0]


def register(run_id: str | None = None, cfg_path: str = "configs/model_params.yaml") -> str:
    load_dotenv()
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    name = cfg["registry"]["model_name"]

    found = find_run(cfg["mlflow"]["experiment_name"], run_id)
    run = mlflow.get_run(found.info.run_id)  # search results omit run.outputs
    tags = run.data.tags
    if tags.get("quality_gate") != "passed":
        raise SystemExit(f"Run {run.info.run_id} failed the quality gate. Not registering.")
    if tags.get("git_dirty") != "False":
        logger.warning("Run %s was trained from uncommitted code.", run.info.run_id)

    model_outputs = run.outputs.model_outputs if run.outputs else []
    if not model_outputs:
        raise SystemExit(f"Run {run.info.run_id} has no logged model.")
    source = f"models:/{model_outputs[0].model_id}"
    version = mlflow.register_model(source, name).version
    client = MlflowClient()
    for key in ("git_commit", "dvc_data_md5", "quality_gate"):
        client.set_model_version_tag(name, version, key, tags.get(key, "unknown"))
    client.set_registered_model_alias(name, "candidate", version)

    logger.info("Registered %s version %s (alias: candidate)", name, version)
    return str(version)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    register(parser.parse_args().run_id)