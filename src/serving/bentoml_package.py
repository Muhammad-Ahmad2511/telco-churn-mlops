"""Package the approved MLflow model as a versioned BentoML artifact (spec 12.2).

The model is fetched from the MLflow Model Registry by alias (default
``production``, i.e. the version approved with ``src.models.promote``) and saved
into the local BentoML model store together with its MLflow lineage. The Bento
built from it carries the model, so the serving container needs no registry
access or credentials.

    python -m src.serving.bentoml_package              # alias from MODEL_ALIAS or "production"
    python -m src.serving.bentoml_package --alias candidate
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from src.serving.schema import FEATURE_ORDER

BENTO_MODEL_NAME = "telco-churn-classifier"


def package_model(
    model: Any,
    *,
    registry_name: str,
    version: str,
    alias: str,
    run_id: str | None = None,
    tags: dict[str, str] | None = None,
):
    """Save a fitted scikit-learn pipeline to the BentoML model store."""
    import bentoml

    return bentoml.sklearn.save_model(
        BENTO_MODEL_NAME,
        model,
        signatures={"predict": {"batchable": False}, "predict_proba": {"batchable": False}},
        labels={
            "mlflow_model": registry_name,
            "mlflow_version": str(version),
            "mlflow_alias": alias,
        },
        metadata={
            "mlflow_run_id": run_id,
            "lineage_tags": dict(tags or {}),
            "feature_order": list(FEATURE_ORDER),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the registry model for BentoML.")
    parser.add_argument("--alias", default=None, help="registry alias (default: production)")
    args = parser.parse_args()
    if args.alias:
        os.environ["MODEL_ALIAS"] = args.alias

    from src.serving.model_loader import load_registered_model

    bundle = load_registered_model()
    saved = package_model(
        bundle.model,
        registry_name=bundle.name,
        version=bundle.version,
        alias=bundle.alias,
        run_id=bundle.run_id,
        tags=bundle.tags,
    )
    print(f"packaged {bundle.name} v{bundle.version} (@{bundle.alias}) as {saved.tag}")


if __name__ == "__main__":
    main()
