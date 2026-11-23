# tests/test_metrics.py
"""Prometheus metrics exposed by the API (spec §3.12). Uses a fake model, no network."""
import numpy as np
import pytest

from src.serving.app import create_app
from src.serving.model_loader import ModelBundle, ModelStore
from src.serving.schema import SAMPLE_RECORD


class FakeModel:
    def predict_proba(self, X):
        p = np.full(len(X), 0.8)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return np.ones(len(X), dtype=int)


BUNDLE = ModelBundle(
    model=FakeModel(), name="telco-churn-classifier", version="2",
    alias="production", run_id="abc", tags={},
)


def broken_loader():
    raise RuntimeError("registry down")


@pytest.fixture
def app():
    return create_app(ModelStore(loader=lambda: BUNDLE))


def value(app, name, labels=None):
    v = app.extensions["metrics"].registry.get_sample_value(name, labels or {})
    return 0.0 if v is None else v


def test_metrics_endpoint_serves_prometheus_text(app):
    resp = app.test_client().get("/metrics")
    assert resp.status_code == 200
    assert resp.mimetype == "text/plain"
    body = resp.get_data(as_text=True)
    for name in ("telco_api_requests_total", "telco_model_loaded", "telco_predictions_total"):
        assert name in body


def test_request_counter_and_latency(app):
    client = app.test_client()
    client.get("/health")
    client.get("/health")
    labels = {"endpoint": "/health", "method": "GET", "status": "200"}
    assert value(app, "telco_api_requests_total", labels) == 2
    assert value(app, "telco_api_request_latency_seconds_count", {"endpoint": "/health"}) == 2


def test_metrics_endpoint_does_not_count_itself(app):
    client = app.test_client()
    client.get("/metrics")
    body = client.get("/metrics").get_data(as_text=True)
    assert 'endpoint="/metrics"' not in body


def test_predictions_counted_by_model_version(app):
    app.test_client().post("/predict", json={"instances": [SAMPLE_RECORD, SAMPLE_RECORD]})
    assert value(app, "telco_predictions_total", {"model_version": "2", "prediction": "1"}) == 2
    assert value(app, "telco_churn_probability_count", {"model_version": "2"}) == 2


def test_validation_failures_counted_by_reason(app):
    client = app.test_client()
    client.post("/predict", json={"tenure": "abc"})
    client.post("/predict", data="not json", content_type="text/plain")
    assert value(app, "telco_validation_failures_total", {"reason": "schema"}) == 1
    assert value(app, "telco_validation_failures_total", {"reason": "invalid_json"}) == 1
    assert value(app, "telco_api_requests_total",
                 {"endpoint": "/predict", "method": "POST", "status": "422"}) == 1


def test_unknown_paths_share_one_label(app):
    client = app.test_client()
    client.get("/random-1")
    client.get("/random-2")
    labels = {"endpoint": "unmatched", "method": "GET", "status": "404"}
    assert value(app, "telco_api_requests_total", labels) == 2


def test_model_gauges_when_loaded(app):
    app.test_client().get("/health")
    assert value(app, "telco_model_loaded") == 1
    info = {"model_name": "telco-churn-classifier", "model_version": "2", "alias": "production"}
    assert value(app, "telco_model_info", info) == 1


def test_model_loaded_zero_when_registry_down():
    app = create_app(ModelStore(loader=broken_loader))
    assert app.test_client().get("/health").status_code == 503
    assert value(app, "telco_model_loaded") == 0
