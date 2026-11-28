"""Materialize features to the online store and query them (spec 12.1).

Runs in the Feast environment (.venv-feast):

    python -m feature_store.build_features        # main environment
    cd feature_store/feature_repo && feast apply  # register definitions
    python -m feature_store.materialize           # offline -> online + demo queries
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from feast import FeatureStore

REPO_PATH = Path(__file__).resolve().parent / "feature_repo"
FEATURE_SERVICE = "churn_features"
ENTITY_KEY = "customerID"


def get_store(repo_path: Path = REPO_PATH) -> FeatureStore:
    return FeatureStore(repo_path=str(repo_path))


def materialize_features(store: FeatureStore, start: datetime, end: datetime) -> None:
    """Copy feature values with timestamps in [start, end] into the online store."""
    store.materialize(start_date=start, end_date=end)


def get_training_features(store: FeatureStore, entity_df: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time correct training features from the offline store.

    entity_df needs the entity key and an event_timestamp column.
    """
    service = store.get_feature_service(FEATURE_SERVICE)
    return store.get_historical_features(entity_df=entity_df, features=service).to_df()


def get_online_features(store: FeatureStore, customer_ids: list[str]) -> pd.DataFrame:
    """Low-latency features for inference from the online store."""
    service = store.get_feature_service(FEATURE_SERVICE)
    rows = [{ENTITY_KEY: customer_id} for customer_id in customer_ids]
    return pd.DataFrame(store.get_online_features(features=service, entity_rows=rows).to_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize features and run demo queries.")
    parser.add_argument("--repo", type=Path, default=REPO_PATH)
    args = parser.parse_args()

    store = get_store(args.repo)
    source = pd.read_parquet(args.repo / "data" / "customer_features.parquet")
    start = source["event_timestamp"].min().to_pydatetime() - timedelta(days=1)
    end = datetime.now(UTC)
    materialize_features(store, start, end)
    print(f"materialized {len(source)} rows from {start:%Y-%m-%d} to {end:%Y-%m-%d}")

    sample = source[ENTITY_KEY].head(3).tolist()
    print("online features for", sample)
    print(get_online_features(store, sample).T)

    entity_df = source[[ENTITY_KEY]].head(5).copy()
    entity_df["event_timestamp"] = end
    print("offline (training) features:")
    print(get_training_features(store, entity_df).T)


if __name__ == "__main__":
    main()
