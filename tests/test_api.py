# tests/test_api.py
"""API tests with a fake model (no network needed)."""
import numpy as np
import pytest

from src.serving.app import create_app
from src.serving.model_loader import ModelBundle, ModelStore
from src.serving.schema import MAX_INSTANCES, SAMPLE_RECORD


class FakeModel:
    def predict_proba(self, X):
        p = np.full(len(X), 0.8)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return np.ones(len(X), dtype=int)


BUNDLE = ModelBundle(
    model=FakeModel(),
    name="telco-churn-classifier",
    version="1",
    alias="candidate",
    run_id="abc123",
    tags={"git_commit": "deadbeef", "dvc_data_md5": "cafe", "quality_gate": "passed"},
)


def broken_loader():
    raise RuntimeError("registry down")


@pytest.fixture
def client():
    return create_app(ModelStore(loader=lambda: BUNDLE)).test_client()


@pytest.fixture
def client_no_model():
    return create_app(ModelStore(loader=broken_loader)).test_client()


def record(**changes):
    out = dict(SAMPLE_RECORD)
    out.update(changes)
    return out


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok", "model_loaded": True}


def test_health_503_when_model_missing(client_no_model):
    r = client_no_model.get("/health")
    assert r.status_code == 503
    assert r.get_json()["model_loaded"] is False


def test_model_info_fields(client):
    body = client.get("/model-info").get_json()
    assert body["model_name"] == "telco-churn-classifier"
    assert body["model_version"] == "1"
    assert body["git_commit"] == "deadbeef"
    assert len(body["features"]) == 18


def test_predict_single_record(client):
    r = client.post("/predict", json=record())
    assert r.status_code == 200
    body = r.get_json()
    assert body["model_version"] == "1"
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["churn"] in (0, 1)
    assert 0.0 <= body["predictions"][0]["churn_probability"] <= 1.0


def test_predict_batch(client):
    r = client.post("/predict", json={"instances": [record(), record(tenure=40)]})
    assert r.status_code == 200
    assert len(r.get_json()["predictions"]) == 2


def test_predict_invalid_json_returns_400(client):
    r = client.post("/predict", data="{bad json", content_type="application/json")
    assert r.status_code == 400
    assert r.get_json()["error"] == "invalid_json"


def test_predict_wrong_content_type_returns_400(client):
    r = client.post("/predict", data="hello", content_type="text/plain")
    assert r.status_code == 400


def test_predict_missing_field_returns_422(client):
    bad = record()
    del bad["Contract"]
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
    assert r.get_json()["details"][0]["index"] == 0


def test_predict_invalid_category_returns_422(client):
    assert client.post("/predict", json=record(Contract="Weekly")).status_code == 422


def test_predict_unknown_field_returns_422(client):
    assert client.post("/predict", json=record(gender="Male")).status_code == 422


def test_predict_wrong_type_returns_422(client):
    assert client.post("/predict", json=record(tenure="twelve")).status_code == 422


def test_predict_bool_is_not_a_number(client):
    assert client.post("/predict", json=record(SeniorCitizen=True)).status_code == 422


def test_predict_empty_instances_returns_422(client):
    assert client.post("/predict", json={"instances": []}).status_code == 422


def test_predict_too_many_instances_returns_422(client):
    payload = {"instances": [record()] * (MAX_INSTANCES + 1)}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_503_when_model_missing(client_no_model):
    assert client_no_model.post("/predict", json=record()).status_code == 503


def test_unknown_route_returns_json_404(client):
    r = client.get("/nope")
    assert r.status_code == 404
    assert r.get_json()["error"] == "not_found"


def test_error_response_does_not_echo_input(client):
    r = client.post("/predict", json=record(Contract="SECRET-VALUE-123"))
    assert r.status_code == 422
    assert "SECRET-VALUE-123" not in r.get_data(as_text=True)

def test_single_threaded_sets_n_jobs_for_serving():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from src.serving.model_loader import single_threaded

    pipe = Pipeline([("model", RandomForestClassifier(n_jobs=-1))])
    assert single_threaded(pipe).named_steps["model"].n_jobs == 1
    bare = LogisticRegression(n_jobs=-1)
    assert single_threaded(bare).n_jobs == 1


def test_configure_logging_adds_handler_only_once(monkeypatch):
    import logging

    from src.serving.app import configure_logging

    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers = []
    try:
        configure_logging()
        configure_logging()
        assert len(root.handlers) == 1
    finally:
        root.handlers = saved


def test_model_store_loads_once_and_never_blocks_other_callers():
    """Regression: concurrent /health calls each downloaded the model and starved /metrics."""
    import threading
    import time

    calls = []
    release = threading.Event()

    def slow_loader():
        calls.append(1)
        release.wait(timeout=5)
        return BUNDLE

    store = ModelStore(loader=slow_loader)
    loader_thread = store.warm_up()
    time.sleep(0.05)  # let the warm-up thread start loading

    started = time.perf_counter()
    results = [store.get() for _ in range(20)]  # e.g. probes arriving during the download
    assert time.perf_counter() - started < 0.5  # answered immediately, nobody waited
    assert results == [None] * 20

    release.set()
    loader_thread.join(timeout=5)
    assert store.get() is BUNDLE
    assert len(calls) == 1  # one download, not one per request


def test_model_store_retries_after_failed_load():
    attempts = []

    def flaky_loader():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("registry down")
        return BUNDLE

    store = ModelStore(loader=flaky_loader)
    assert store.get() is None
    assert store.get() is BUNDLE
    assert len(attempts) == 2
