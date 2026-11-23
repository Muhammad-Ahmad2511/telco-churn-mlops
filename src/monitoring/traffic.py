# src/monitoring/traffic.py
"""Send demo traffic to the churn API so dashboards and alerts have data.

Uses real customer rows from the processed dataset (SIMULATED traffic, not real users).

    python -m src.monitoring.traffic                          # 200 valid requests
    python -m src.monitoring.traffic --n 300 --invalid 0.5    # 50% bad requests -> validation alert
    python -m src.monitoring.traffic --url http://<aws-endpoint>:5000
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter

from src.features.preprocessing import load_config, load_processed
from src.serving.schema import FEATURE_ORDER

INT_FIELDS = {"SeniorCitizen", "tenure"}


def sample_records(n: int, seed: int = 0) -> list[dict]:
    df = load_processed(load_config()).sample(n=n, replace=True, random_state=seed)
    records = []
    for row in df[list(FEATURE_ORDER)].to_dict(orient="records"):
        records.append({k: int(v) if k in INT_FIELDS else (float(v) if isinstance(v, float) else v)
                        for k, v in row.items()})
    return records


def make_invalid(record: dict, rng: random.Random) -> dict:
    bad = dict(record)
    kind = rng.choice(["wrong_type", "bad_category", "missing_field", "out_of_range"])
    if kind == "wrong_type":
        bad["tenure"] = "twelve"
    elif kind == "bad_category":
        bad["Contract"] = "Weekly"
    elif kind == "missing_field":
        bad.pop("MonthlyCharges")
    else:
        bad["MonthlyCharges"] = -5.0
    return bad


def post(url: str, payload: dict, timeout: float = 10.0) -> int:
    req = urllib.request.Request(
        f"{url.rstrip('/')}/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as err:
        return err.code
    except urllib.error.URLError:
        return 0  # connection refused / API down


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Send simulated traffic to /predict.")
    parser.add_argument("--url", default="http://127.0.0.1:5000")
    parser.add_argument("--n", type=int, default=200, help="number of requests")
    parser.add_argument("--invalid", type=float, default=0.0, help="share of invalid requests (0-1)")
    parser.add_argument("--delay", type=float, default=0.05, help="seconds between requests")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    statuses: Counter[int] = Counter()
    for record in sample_records(args.n, args.seed):
        payload = make_invalid(record, rng) if rng.random() < args.invalid else record
        statuses[post(args.url, payload)] += 1
        time.sleep(args.delay)
    print("status counts:", dict(sorted(statuses.items())), "(0 = connection failed)")


if __name__ == "__main__":
    main()
