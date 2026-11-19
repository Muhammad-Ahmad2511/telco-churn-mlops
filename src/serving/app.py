# src/serving/app.py
"""Flask REST API for churn prediction.

Local run:  python -m src.serving.app
Production: gunicorn "src.serving.app:create_app()"  (Docker, later)
"""
from __future__ import annotations

import logging
import time

from flask import Flask, jsonify, request

from src.serving.model_loader import ModelStore
from src.serving.schema import FEATURE_ORDER, to_dataframe, validate_payload

logger = logging.getLogger("telco_api")


def create_app(store: ModelStore | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # reject huge bodies
    store = store or ModelStore()

    def unavailable():
        return jsonify(error="model_unavailable", message="Model is not loaded."), 503

    @app.get("/health")
    def health():
        if store.get() is None:
            return jsonify(status="unavailable", model_loaded=False), 503
        return jsonify(status="ok", model_loaded=True), 200

    @app.get("/model-info")
    def model_info():
        bundle = store.get()
        if bundle is None:
            return unavailable()
        return jsonify(
            model_name=bundle.name,
            model_version=bundle.version,
            alias=bundle.alias,
            run_id=bundle.run_id,
            git_commit=bundle.tags.get("git_commit"),
            dvc_data_md5=bundle.tags.get("dvc_data_md5"),
            quality_gate=bundle.tags.get("quality_gate"),
            features=list(FEATURE_ORDER),
        )

    @app.post("/predict")
    def predict():
        started = time.perf_counter()
        body = request.get_json(silent=True)
        if body is None:
            return (
                jsonify(
                    error="invalid_json",
                    message="Send a JSON body with Content-Type: application/json.",
                ),
                400,
            )
        records, errors = validate_payload(body)
        if errors:
            return jsonify(error="validation_failed", details=errors), 422
        bundle = store.get()
        if bundle is None:
            return unavailable()

        X = to_dataframe(records)
        labels = bundle.model.predict(X)
        probabilities = bundle.model.predict_proba(X)[:, 1]
        predictions = [
            {"churn": int(label), "churn_probability": round(float(p), 4)}
            for label, p in zip(labels, probabilities)
        ]
        logger.info(
            "predict instances=%d model_version=%s latency_ms=%.1f",
            len(predictions),
            bundle.version,
            (time.perf_counter() - started) * 1000,
        )
        return jsonify(
            model_name=bundle.name,
            model_version=bundle.version,
            predictions=predictions,
        )

    @app.errorhandler(404)
    def not_found(_err):
        return jsonify(error="not_found", message="Unknown endpoint."), 404

    @app.errorhandler(405)
    def method_not_allowed(_err):
        return jsonify(error="method_not_allowed", message="Method not allowed."), 405

    @app.errorhandler(413)
    def too_large(_err):
        return jsonify(error="payload_too_large", message="Request body too large."), 413

    @app.errorhandler(500)
    def server_error(_err):
        return jsonify(error="internal_error", message="Unexpected server error."), 500

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    create_app().run(host="127.0.0.1", port=5000)