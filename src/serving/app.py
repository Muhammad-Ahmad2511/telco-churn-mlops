# src/serving/app.py
"""Flask REST API for churn prediction.

Local run:  python -m src.serving.app
Production: gunicorn "src.serving.app:create_app()"  (docker/Dockerfile.api)
Metrics:    GET /metrics (Prometheus text format, see src/serving/metrics.py)
"""
from __future__ import annotations

import logging
import os
import time

from flask import Flask, Response, g, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.serving.metrics import create_metrics
from src.serving.model_loader import ModelBundle, ModelStore
from src.serving.schema import FEATURE_ORDER, to_dataframe, validate_payload

logger = logging.getLogger("telco_api")


def configure_logging() -> None:
    """Send INFO logs to stdout (-> docker logs / CloudWatch) unless already configured.

    Only counts, model version and latency are logged, never request contents.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )


def create_app(store: ModelStore | None = None) -> Flask:
    configure_logging()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # reject huge bodies
    store = store or ModelStore()
    metrics = create_metrics()
    app.extensions["metrics"] = metrics

    def track_model(bundle: ModelBundle | None) -> ModelBundle | None:
        """Keep the model gauges in sync with what the API can actually serve."""
        metrics.model_loaded.set(0 if bundle is None else 1)
        if bundle is not None:
            metrics.model_info.clear()
            metrics.model_info.labels(bundle.name, bundle.version, bundle.alias).set(1)
        return bundle

    @app.before_request
    def start_timer():
        g.started = time.perf_counter()

    @app.after_request
    def record_request(response):
        # Route template (e.g. "/predict"), never the raw path, to keep label cardinality low.
        endpoint = request.url_rule.rule if request.url_rule else "unmatched"
        if endpoint != "/metrics":
            metrics.requests.labels(endpoint, request.method, str(response.status_code)).inc()
            metrics.latency.labels(endpoint).observe(time.perf_counter() - g.started)
        return response

    @app.get("/metrics")
    def prometheus_metrics():
        return Response(generate_latest(metrics.registry), mimetype=CONTENT_TYPE_LATEST)

    def unavailable():
        return jsonify(error="model_unavailable", message="Model is not loaded."), 503

    @app.get("/health")
    def health():
        if track_model(store.get()) is None:
            return jsonify(status="unavailable", model_loaded=False), 503
        return jsonify(status="ok", model_loaded=True), 200

    @app.get("/model-info")
    def model_info():
        bundle = track_model(store.get())
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
            metrics.validation_failures.labels("invalid_json").inc()
            return (
                jsonify(
                    error="invalid_json",
                    message="Send a JSON body with Content-Type: application/json.",
                ),
                400,
            )
        records, errors = validate_payload(body)
        if errors:
            metrics.validation_failures.labels("schema").inc()
            return jsonify(error="validation_failed", details=errors), 422
        bundle = track_model(store.get())
        if bundle is None:
            return unavailable()

        X = to_dataframe(records)
        labels = bundle.model.predict(X)
        probabilities = bundle.model.predict_proba(X)[:, 1]
        predictions = [
            {"churn": int(label), "churn_probability": round(float(p), 4)}
            for label, p in zip(labels, probabilities)
        ]
        for item in predictions:
            metrics.predictions.labels(bundle.version, str(item["churn"])).inc()
            metrics.churn_probability.labels(bundle.version).observe(item["churn_probability"])
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
    create_app().run(host="127.0.0.1", port=5000)