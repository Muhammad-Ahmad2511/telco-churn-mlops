"""Tests for data ingestion without DVC or AWS network access."""
from __future__ import annotations

import io

import pandas as pd
import pytest

from src.data.ingestion import load_raw_data, parse_s3_uri


def test_loads_local_csv(tmp_path):
    source = tmp_path / "customers.csv"
    expected = pd.DataFrame({"customerID": ["a", "b"], "Churn": ["No", "Yes"]})
    expected.to_csv(source, index=False)

    pd.testing.assert_frame_equal(load_raw_data(source), expected)


def test_uses_raw_path_from_pipeline_config(tmp_path):
    source = tmp_path / "customers.csv"
    pd.DataFrame({"customerID": ["a"], "Churn": ["No"]}).to_csv(source, index=False)
    config = tmp_path / "pipeline.yaml"
    config.write_text(f"data:\n  raw_path: {source.as_posix()}\n", encoding="utf-8")

    loaded = load_raw_data(config_path=str(config))

    assert loaded.to_dict(orient="records") == [{"customerID": "a", "Churn": "No"}]


def test_missing_local_source_explains_how_to_restore_dvc_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="dvc pull"):
        load_raw_data(tmp_path / "missing.csv")


def test_parse_s3_uri_requires_bucket_and_key():
    assert parse_s3_uri("s3://telco-data/raw/churn.csv") == ("telco-data", "raw/churn.csv")
    with pytest.raises(ValueError, match="s3://"):
        parse_s3_uri("s3://telco-data")


def test_loads_s3_csv_with_injected_client():
    calls = []

    class FakeS3Client:
        def get_object(self, **kwargs):
            calls.append(kwargs)
            return {"Body": io.BytesIO(b"customerID,Churn\na,No\n")}

    loaded = load_raw_data("s3://telco-data/raw/churn.csv", s3_client=FakeS3Client())

    assert calls == [{"Bucket": "telco-data", "Key": "raw/churn.csv"}]
    assert loaded.to_dict(orient="records") == [{"customerID": "a", "Churn": "No"}]
