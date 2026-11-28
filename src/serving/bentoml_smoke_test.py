"""Repeatable smoke test for the BentoML service (local `bentoml serve` or a container).

    python -m src.serving.bentoml_smoke_test                       # http://localhost:3000
    python -m src.serving.bentoml_smoke_test --url http://localhost:3055

Exits with code 1 if any check fails, so it can also run in CI.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import urllib.parse

from src.serving.schema import SAMPLE_RECORD


def call(url: str, method: str = "GET", body: dict | None = None) -> tuple[int, object]:
    """Send one request; returns (status, body). Status 0 means the server was unreachable."""
    parts = urllib.parse.urlsplit(url)
    connection_class = (
        http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    )
    payload = json.dumps(body) if body is not None else None
    try:
        connection = connection_class(parts.netloc, timeout=30)
        try:
            connection.request(
                method,
                parts.path or "/",
                body=payload,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            raw = response.read().decode()
            status = response.status
        finally:
            connection.close()
    except (OSError, http.client.HTTPException) as error:  # server not reachable
        return 0, str(error)
    return status, json.loads(raw) if raw.startswith(("{", "[")) else raw


def run(base: str) -> list[str]:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: object = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  -> {detail}"))
        if not ok:
            failures.append(name)

    status, _ = call(f"{base}/livez")
    check("GET /livez returns 200", status == 200, status)
    status, _ = call(f"{base}/readyz")
    check("GET /readyz returns 200", status == 200, status)

    status, info = call(f"{base}/model-info", "POST", {})
    check(
        "POST /model-info returns model name and version",
        status == 200 and isinstance(info, dict) and bool(info.get("model_version")),
        (status, info),
    )

    status, out = call(f"{base}/predict", "POST", {"instances": [SAMPLE_RECORD]})
    ok = status == 200 and isinstance(out, dict) and len(out.get("predictions", [])) == 1
    check("POST /predict returns one prediction", ok, (status, out))
    if ok:
        prediction = out["predictions"][0]
        check(
            "prediction has churn 0/1 and probability in [0, 1]",
            prediction["churn"] in (0, 1) and 0 <= prediction["churn_probability"] <= 1,
            prediction,
        )

    bad = {**SAMPLE_RECORD, "Contract": "Weekly"}
    status, _ = call(f"{base}/predict", "POST", {"instances": [bad]})
    check("invalid category is rejected with 422", status == 422, status)
    status, _ = call(f"{base}/predict", "POST", {"instances": []})
    check("empty instances list is rejected with 422", status == 422, status)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the BentoML churn service.")
    parser.add_argument("--url", default="http://localhost:3000")
    args = parser.parse_args()
    failures = run(args.url.rstrip("/"))
    if failures:
        print(f"\n{len(failures)} check(s) failed")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
