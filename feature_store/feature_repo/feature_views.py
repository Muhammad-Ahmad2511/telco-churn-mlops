"""Feast sources, feature views and the feature service for the churn model."""

from datetime import timedelta

from entities import customer
from feast import FeatureService, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String

from features import (
    ACCOUNT_FEATURES,
    EVENT_TIMESTAMP,
    FEATURE_OWNER,
    MAX_AGE_DAYS,
    SERVICE_FEATURES,
)

_TYPES = {"int64": Int64, "float64": Float64, "string": String}

customer_source = FileSource(
    name="customer_features_source",
    path="data/customer_features.parquet",
    timestamp_field=EVENT_TIMESTAMP,
    description="Customer features built by feature_store/build_features.py",
)

_TAGS = {
    "owner": FEATURE_OWNER,
    "freshness": f"rebuild when processed data changes; max age {MAX_AGE_DAYS} days",
    "source": "data/processed/telco_clean.parquet (DVC)",
}

customer_account_features = FeatureView(
    name="customer_account_features",
    entities=[customer],
    ttl=timedelta(days=3650),
    schema=[Field(name=n, dtype=_TYPES[t]) for n, t in ACCOUNT_FEATURES.items()],
    source=customer_source,
    owner=FEATURE_OWNER,
    description="Account, billing and demographic (non-sensitive) features",
    tags=_TAGS,
)

customer_service_features = FeatureView(
    name="customer_service_features",
    entities=[customer],
    ttl=timedelta(days=3650),
    schema=[Field(name=n, dtype=_TYPES[t]) for n, t in SERVICE_FEATURES.items()],
    source=customer_source,
    owner=FEATURE_OWNER,
    description="Phone, internet and streaming service features",
    tags=_TAGS,
)

churn_features = FeatureService(
    name="churn_features",
    features=[customer_account_features, customer_service_features],
    description="All features used by the churn model, for training and serving",
)
