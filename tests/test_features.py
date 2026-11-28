"""Feature-table tests that need only pandas (run in the main environment)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import yaml
from feature_store_helpers import (
    load_build_features,
    load_feature_catalogue,
    make_processed,
)

bf = load_build_features()
catalogue = load_feature_catalogue()
CONFIG = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "configs" / "pipeline_config.yaml").read_text()
)["data"]


@pytest.fixture
def frame():
    return bf.build_feature_frame(make_processed(), datetime.now(UTC))


def test_frame_has_key_timestamp_and_features_only(frame):
    expected = {catalogue.ENTITY_KEY, catalogue.EVENT_TIMESTAMP, *catalogue.FEATURE_COLUMNS}
    assert set(frame.columns) == expected


def test_sensitive_and_label_columns_are_not_features(frame):
    assert "gender" not in frame.columns
    assert CONFIG["target"] not in frame.columns
    for dropped in CONFIG["drop_columns"]:
        if dropped != catalogue.ENTITY_KEY:
            assert dropped not in catalogue.FEATURE_COLUMNS


def test_catalogue_groups_do_not_overlap():
    assert not set(catalogue.ACCOUNT_FEATURES) & set(catalogue.SERVICE_FEATURES)


def test_feature_types_match_catalogue(frame):
    for column, dtype in catalogue.FEATURE_COLUMNS.items():
        if dtype in ("int64", "float64"):
            assert str(frame[column].dtype) == dtype, column


def test_event_timestamp_is_utc_and_present(frame):
    assert str(frame[catalogue.EVENT_TIMESTAMP].dt.tz) == "UTC"
    assert frame[catalogue.EVENT_TIMESTAMP].notna().all()


def test_entity_keys_are_unique_and_not_null(frame):
    assert frame[catalogue.ENTITY_KEY].is_unique
    assert frame[catalogue.ENTITY_KEY].notna().all()


def test_valid_frame_passes_quality_checks(frame):
    assert bf.check_feature_quality(frame) == []


def test_missing_column_is_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        bf.build_feature_frame(make_processed().drop(columns=["tenure"]))


def test_duplicate_key_is_rejected():
    processed = make_processed()
    processed.loc[1, "customerID"] = processed.loc[0, "customerID"]
    with pytest.raises(ValueError, match="duplicate"):
        bf.build_feature_frame(processed)


def test_excess_nulls_are_reported(frame):
    frame.loc[: len(frame) // 2, "MonthlyCharges"] = None
    problems = bf.check_feature_quality(frame)
    assert any("MonthlyCharges" in p and "nulls" in p for p in problems)


def test_null_key_is_reported(frame):
    frame.loc[0, "customerID"] = None
    assert "null entity keys" in bf.check_feature_quality(frame)


def test_invalid_values_are_reported(frame):
    frame.loc[0, "tenure"] = -5
    frame.loc[1, "SeniorCitizen"] = 7
    problems = bf.check_feature_quality(frame)
    assert any("tenure" in p and "negative" in p for p in problems)
    assert any("SeniorCitizen" in p for p in problems)


def test_stale_snapshot_is_reported():
    old = datetime.now(UTC) - timedelta(days=catalogue.MAX_AGE_DAYS + 5)
    stale = bf.build_feature_frame(make_processed(), old)
    assert any("stale" in p for p in bf.check_feature_quality(stale))


def test_naive_timestamp_is_treated_as_utc():
    naive = datetime(2026, 11, 1, 12, 0, 0)
    result = bf.build_feature_frame(make_processed(), naive)
    expected = pd.Timestamp("2026-11-01 12:00:00", tz="UTC")
    assert result[catalogue.EVENT_TIMESTAMP].iloc[0] == expected
