"""BentoML serving path for the churn model (spec 12.2).

The Flask API (src/serving/app.py) stays the required API. This service packages
the same model as a versioned Bento and uses the same request validation
(src/serving/schema.py), so both paths accept and reject the same inputs.

    bentoml serve src.serving.bentoml_service:ChurnService
"""

from __future__ import annotations

import bentoml
from bentoml.exceptions import UnprocessableEntity

from src.serving.bentoml_package import BENTO_MODEL_NAME
from src.serving.schema import to_dataframe, validate_payload


@bentoml.service(name="telco_churn", traffic={"timeout": 30})
class ChurnService:
    bento_model = bentoml.models.BentoModel(f"{BENTO_MODEL_NAME}:latest")

    def __init__(self) -> None:
        self.model = bentoml.sklearn.load_model(self.bento_model)
        stored = bentoml.models.get(str(self.bento_model.tag))
        self.labels = dict(stored.info.labels)
        self.metadata = dict(stored.info.metadata)
        self.bento_tag = str(stored.tag)

    @bentoml.api(route="/predict")
    def predict(self, instances: list[dict]) -> dict:
        """Predict churn for 1 to 100 customers; invalid input returns HTTP 422."""
        records, errors = validate_payload({"instances": instances})
        if errors:
            raise UnprocessableEntity(f"invalid input: {errors}")
        frame = to_dataframe(records)
        probabilities = self.model.predict_proba(frame)[:, 1]
        predictions = self.model.predict(frame)
        return {
            "model_name": self.labels.get("mlflow_model"),
            "model_version": self.labels.get("mlflow_version"),
            "predictions": [
                {"churn": int(label), "churn_probability": round(float(p), 4)}
                for label, p in zip(predictions, probabilities, strict=True)
            ],
        }

    @bentoml.api(route="/model-info")
    def model_info(self) -> dict:
        """Model name, version, alias, run id and lineage tags of this Bento."""
        return {
            "bento_model": self.bento_tag,
            "model_name": self.labels.get("mlflow_model"),
            "model_version": self.labels.get("mlflow_version"),
            "alias": self.labels.get("mlflow_alias"),
            "mlflow_run_id": self.metadata.get("mlflow_run_id"),
            "lineage_tags": self.metadata.get("lineage_tags", {}),
            "features": self.metadata.get("feature_order", []),
        }
