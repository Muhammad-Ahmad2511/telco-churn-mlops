# src/serving/model_loader.py
"""Load the approved model from the MLflow Model Registry.

Smoke test (needs .env): python -m src.serving.model_loader
Alias comes from env MODEL_ALIAS (default: production; set by src.models.promote).
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow.sklearn
import yaml
from dotenv import load_dotenv
from mlflow import MlflowClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    name: str
    version: str
    alias: str
    run_id: str | None
    tags: dict[str, str] = field(default_factory=dict)


def load_registered_model(cfg_path: str = "configs/model_params.yaml") -> ModelBundle:
    load_dotenv()
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    name = cfg["registry"]["model_name"]
    alias = os.getenv("MODEL_ALIAS", "production")

    mv = MlflowClient().get_model_version_by_alias(name, alias)
    model = None
    for uri in (f"models:/{name}/{mv.version}", mv.source):
        try:
            model = mlflow.sklearn.load_model(uri)
            logger.info("Loaded %s version %s from %s", name, mv.version, uri)
            break
        except Exception:  # noqa: BLE001 - MLflow raises different errors; try next URI
            logger.warning("Could not load model from %s, trying next option", uri)
    if model is None:
        raise RuntimeError(f"Could not load {name} version {mv.version}")

    return ModelBundle(
        model=model,
        name=name,
        version=str(mv.version),
        alias=alias,
        run_id=mv.run_id or None,
        tags=dict(mv.tags),
    )


class ModelStore:
    """Holds the loaded model. Retries loading if an earlier attempt failed."""

    def __init__(self, loader: Callable[[], ModelBundle] = load_registered_model):
        self._loader = loader
        self._bundle: ModelBundle | None = None

    def get(self) -> ModelBundle | None:
        if self._bundle is None:
            try:
                self._bundle = self._loader()
            except Exception:
                logger.exception("Model load failed")
        return self._bundle


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from src.serving.schema import SAMPLE_RECORD, to_dataframe

    bundle = load_registered_model()
    print("model:", bundle.name, "version:", bundle.version, "alias:", bundle.alias)
    print("run_id:", bundle.run_id)
    print("tags:", bundle.tags)
    X = to_dataframe([SAMPLE_RECORD])
    print("prediction:", bundle.model.predict(X))
    print("churn probability:", bundle.model.predict_proba(X)[:, 1])