"""Feast offline/online retrieval tests (need Feast: run with .venv-feast).

In an environment without Feast these tests are skipped, not failed.
"""

import importlib
import shutil
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from feature_store_helpers import (
    FEATURE_REPO,
    load_build_features,
    load_feature_catalogue,
    load_materialize,
    make_processed,
)

pytest.importorskip("feast")
cli = importlib.import_module("feast.cli.cli").cli
CliRunner = importlib.import_module("click.testing").CliRunner

bf = load_build_features()
catalogue = load_feature_catalogue()
mat = load_materialize()
FEATURES = list(catalogue.FEATURE_COLUMNS)


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    """A temporary copy of the feature repo with a freshly built source table."""
    path = tmp_path_factory.mktemp("feature_repo")
    for name in (
        "feature_store.yaml",
        "entities.py",
        "features.py",
        "feature_views.py",
    ):
        shutil.copy(FEATURE_REPO / name, path / name)
    (path / "data").mkdir()

    snapshot = datetime.now(UTC) - timedelta(days=2)
    source = bf.build_feature_frame(make_processed(60), snapshot)
    source.to_parquet(path / "data" / "customer_features.parquet", index=False)

    result = CliRunner().invoke(cli, ["--chdir", str(path), "apply"])
    assert result.exit_code == 0, result.output
    return path, source


@pytest.fixture(scope="module")
def store(repo):
    return mat.get_store(repo[0])


def test_apply_registers_definitions(store):
    assert [e.name for e in store.list_entities()] == ["customer"]
    names = {v.name for v in store.list_feature_views()}
    assert names == {"customer_account_features", "customer_service_features"}
    assert [s.name for s in store.list_feature_services()] == ["churn_features"]


def test_feature_views_cover_every_catalogue_feature(store):
    registered = {
        field.name for view in store.list_feature_views() for field in view.features
    }
    assert registered == set(FEATURES)


def test_offline_training_retrieval(repo, store):
    _, source = repo
    entity_df = source[["customerID"]].copy()
    entity_df["event_timestamp"] = datetime.now(UTC)
    training = mat.get_training_features(store, entity_df)

    assert len(training) == len(source)
    assert training[FEATURES].notna().all().all()
    merged = training.merge(source, on="customerID", suffixes=("_t", "_s"))
    for column in ("tenure", "MonthlyCharges", "Contract"):
        assert (merged[f"{column}_t"] == merged[f"{column}_s"]).all()


def test_point_in_time_excludes_future_features(repo, store):
    """Rows requested for a time before the snapshot must not see its features."""
    _, source = repo
    entity_df = source[["customerID"]].head(5).copy()
    entity_df["event_timestamp"] = datetime.now(UTC) - timedelta(days=10)
    training = mat.get_training_features(store, entity_df)
    assert training["tenure"].isna().all()


def test_materialize_and_online_retrieval(repo, store):
    _, source = repo
    start = datetime.now(UTC) - timedelta(days=5)
    mat.materialize_features(store, start, datetime.now(UTC))

    ids = source["customerID"].head(5).tolist()
    online = mat.get_online_features(store, ids).set_index("customerID")
    expected = source.set_index("customerID").loc[ids]
    for column in ("tenure", "MonthlyCharges", "Contract", "PaymentMethod"):
        assert online.loc[ids, column].tolist() == expected[column].tolist()


def test_unknown_customer_returns_missing_features(repo, store):
    start = datetime.now(UTC) - timedelta(days=5)
    mat.materialize_features(store, start, datetime.now(UTC))
    online = mat.get_online_features(store, ["does-not-exist"])
    assert online["tenure"].isna().all()


def test_no_training_serving_skew(repo, store):
    """The same customer gets identical feature values offline and online."""
    _, source = repo
    start = datetime.now(UTC) - timedelta(days=5)
    mat.materialize_features(store, start, datetime.now(UTC))

    ids = source["customerID"].head(20).tolist()
    entity_df = pd.DataFrame({"customerID": ids, "event_timestamp": datetime.now(UTC)})
    offline = mat.get_training_features(store, entity_df).set_index("customerID")
    online = mat.get_online_features(store, ids).set_index("customerID")

    for column in FEATURES:
        assert offline.loc[ids, column].tolist() == online.loc[ids, column].tolist(), column


def test_stale_features_are_not_materialized(tmp_path, repo):
    """Features outside the materialization window never reach the online store."""
    path = tmp_path / "repo"
    path.mkdir()
    for name in ("feature_store.yaml", "entities.py", "features.py", "feature_views.py"):
        shutil.copy(repo[0] / name, path / name)
    (path / "data").mkdir()
    shutil.copy(
        repo[0] / "data" / "customer_features.parquet",
        path / "data" / "customer_features.parquet",
    )
    result = CliRunner().invoke(cli, ["--chdir", str(path), "apply"])
    assert result.exit_code == 0, result.output

    store = mat.get_store(path)
    old_window_end = datetime.now(UTC) - timedelta(days=5)
    mat.materialize_features(store, old_window_end - timedelta(days=5), old_window_end)
    ids = repo[1]["customerID"].head(3).tolist()
    assert mat.get_online_features(store, ids)["tenure"].isna().all()
