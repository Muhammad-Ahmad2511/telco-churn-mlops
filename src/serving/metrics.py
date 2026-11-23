# src/serving/metrics.py
"""Prometheus metrics for the churn API (spec §3.12 / §3.12A).

Exposed at GET /metrics and scraped by Prometheus (docker-compose / k8s).
Each app gets its own CollectorRegistry so tests can create many apps.

Note: gunicorn runs 1 worker with several threads (see docker/Dockerfile.api).
With several worker *processes* every process would keep its own counters and a
scrape would only see one of them.
"""
from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


@dataclass
class ApiMetrics:
    registry: CollectorRegistry
    requests: Counter
    latency: Histogram
    predictions: Counter
    churn_probability: Histogram
    validation_failures: Counter
    model_loaded: Gauge
    model_info: Gauge


def create_metrics() -> ApiMetrics:
    registry = CollectorRegistry()
    return ApiMetrics(
        registry=registry,
        requests=Counter(
            "telco_api_requests_total",
            "HTTP requests by endpoint, method and status code.",
            ["endpoint", "method", "status"],
            registry=registry,
        ),
        latency=Histogram(
            "telco_api_request_latency_seconds",
            "Request latency in seconds by endpoint.",
            ["endpoint"],
            buckets=LATENCY_BUCKETS,
            registry=registry,
        ),
        predictions=Counter(
            "telco_predictions_total",
            "Predictions served, by model version and predicted class.",
            ["model_version", "prediction"],
            registry=registry,
        ),
        churn_probability=Histogram(
            "telco_churn_probability",
            "Distribution of predicted churn probability (watch for output drift).",
            ["model_version"],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=registry,
        ),
        validation_failures=Counter(
            "telco_validation_failures_total",
            "Rejected requests (data-quality failures) by reason.",
            ["reason"],
            registry=registry,
        ),
        model_loaded=Gauge(
            "telco_model_loaded",
            "1 if a model is loaded and the API can predict, else 0.",
            registry=registry,
        ),
        model_info=Gauge(
            "telco_model_info",
            "Always 1; labels identify the model being served.",
            ["model_name", "model_version", "alias"],
            registry=registry,
        ),
    )
