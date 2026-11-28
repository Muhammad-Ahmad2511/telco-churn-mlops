"""Integration tests for the BentoML serving path (need `bentoml`; skipped without it).

A temporary BentoML model store is created, a stand-in model with the project's
18 input columns is packaged into it, and a real `bentoml serve` process is
started and called over HTTP.
"""

import http.client
import json
import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from src.serving.bentoml_smoke_test import call as http_call
from src.serving.bentoml_smoke_test import run as run_smoke_test
from src.serving.schema import SAMPLE_RECORD

pytest.importorskip("bentoml")

ROOT = Path(__file__).resolve().parents[1]
LINEAGE = {
    "git_commit": "a" * 40,
    "dvc_data_md5": "b" * 32,
    "feast_version": "0.68.0",
    "feature_repo_commit": "c" * 40,
}

PACKAGE_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from src.serving.bentoml_package import package_model
    from src.serving.schema import CATEGORICAL, NUMERIC, to_dataframe

    rng = np.random.default_rng(0)
    rows = []
    for _ in range(300):
        row = {k: str(rng.choice(list(v))) for k, v in CATEGORICAL.items()}
        row["SeniorCitizen"] = int(rng.integers(0, 2))
        row["tenure"] = int(rng.integers(0, 72))
        row["MonthlyCharges"] = float(rng.uniform(18, 120))
        row["TotalCharges"] = row["MonthlyCharges"] * row["tenure"]
        rows.append(row)
    X = to_dataframe(rows)
    y = (X["tenure"] < 12).astype(int)
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
        ("num", "passthrough", list(NUMERIC)),
    ])
    model = Pipeline([
        ("pre", pre),
        ("model", RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=1)),
    ]).fit(X, y)
    package_model(
        model, registry_name="telco-churn-classifier", version="2", alias="production",
        run_id="run-abc", tags=LINEAGE_PLACEHOLDER,
    )
    """
).replace("LINEAGE_PLACEHOLDER", repr(LINEAGE))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _call(url, method="GET", body=None, raw=None):
    """Return (status, body text). `raw` sends bytes as they are (malformed JSON tests)."""
    if raw is None:
        status, payload = http_call(url, method, body)
        return status, payload if isinstance(payload, str) else json.dumps(payload)
    parts = http.client.HTTPConnection(url.split("//")[1].split("/")[0], timeout=30)
    try:
        parts.request(method, "/" + url.split("//")[1].split("/", 1)[1], body=raw,
                      headers={"Content-Type": "application/json"})
        response = parts.getresponse()
        return response.status, response.read().decode()
    finally:
        parts.close()


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    env = {**os.environ, "BENTOML_HOME": str(tmp_path_factory.mktemp("bentoml_home"))}
    subprocess.run(
        [sys.executable, "-c", PACKAGE_SCRIPT], cwd=ROOT, env=env, check=True, timeout=180
    )
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable, "-m", "bentoml", "serve",
            "src.serving.bentoml_service:ChurnService", "--port", str(port),
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            if process.poll() is not None:
                pytest.fail("bentoml serve exited during start-up")
            try:
                if _call(f"{base}/readyz")[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(1)
        else:
            pytest.fail("bentoml serve did not become ready in time")
        yield base
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()


def test_health_endpoints(server):
    assert _call(f"{server}/livez")[0] == 200
    assert _call(f"{server}/readyz")[0] == 200


def test_model_info_has_version_and_lineage(server):
    status, body = _call(f"{server}/model-info", "POST", {})
    info = json.loads(body)
    assert status == 200
    assert info["model_name"] == "telco-churn-classifier"
    assert info["model_version"] == "2"
    assert info["alias"] == "production"
    assert info["mlflow_run_id"] == "run-abc"
    assert info["lineage_tags"] == LINEAGE
    assert len(info["features"]) == 18


def test_predict_one_and_many(server):
    status, body = _call(f"{server}/predict", "POST", {"instances": [SAMPLE_RECORD]})
    result = json.loads(body)
    assert status == 200
    assert result["model_version"] == "2"
    assert len(result["predictions"]) == 1
    prediction = result["predictions"][0]
    assert prediction["churn"] in (0, 1)
    assert 0 <= prediction["churn_probability"] <= 1

    status, body = _call(f"{server}/predict", "POST", {"instances": [SAMPLE_RECORD] * 5})
    predictions = json.loads(body)["predictions"]
    assert status == 200 and len(predictions) == 5
    assert len({p["churn_probability"] for p in predictions}) == 1  # identical rows


def test_prediction_matches_the_packaged_model(server):
    """Same input, same output: the served model is the packaged model."""
    short_tenure = {**SAMPLE_RECORD, "tenure": 2}
    long_tenure = {**SAMPLE_RECORD, "tenure": 70}
    _, body = _call(f"{server}/predict", "POST", {"instances": [short_tenure, long_tenure]})
    short, long = json.loads(body)["predictions"]
    assert short["churn_probability"] > long["churn_probability"]  # stand-in learned tenure < 12


@pytest.mark.parametrize(
    "field, value",
    [
        ("Contract", "Weekly"),
        ("tenure", -1),
        ("MonthlyCharges", "cheap"),
        ("SeniorCitizen", 2),
    ],
)
def test_invalid_values_are_rejected_with_422(server, field, value):
    status, body = _call(
        f"{server}/predict", "POST", {"instances": [{**SAMPLE_RECORD, field: value}]}
    )
    assert status == 422
    assert field in body


def test_missing_and_unknown_fields_are_rejected_with_422(server):
    missing = {k: v for k, v in SAMPLE_RECORD.items() if k != "tenure"}
    assert _call(f"{server}/predict", "POST", {"instances": [missing]})[0] == 422
    unknown = {**SAMPLE_RECORD, "gender": "Male"}
    assert _call(f"{server}/predict", "POST", {"instances": [unknown]})[0] == 422


def test_instance_count_limits(server):
    assert _call(f"{server}/predict", "POST", {"instances": []})[0] == 422
    too_many = {"instances": [SAMPLE_RECORD] * 101}
    assert _call(f"{server}/predict", "POST", too_many)[0] == 422


def test_malformed_requests_return_400_not_500(server):
    assert _call(f"{server}/predict", "POST", raw=b"{bad")[0] == 400
    assert _call(f"{server}/predict", "POST", SAMPLE_RECORD)[0] == 400  # no "instances" key
    assert _call(f"{server}/predict", "POST", {"instances": "x"})[0] == 400


def test_wrong_method_is_rejected(server):
    assert _call(f"{server}/predict")[0] == 405


def test_smoke_test_script_passes_against_the_service(server):
    assert run_smoke_test(server) == []


def test_smoke_test_script_reports_failures_when_the_service_is_down():
    failures = run_smoke_test(f"http://127.0.0.1:{_free_port()}")
    assert failures  # connection errors are reported, not raised
