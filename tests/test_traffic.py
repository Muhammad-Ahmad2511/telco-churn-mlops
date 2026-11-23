# tests/test_traffic.py
"""The demo traffic generator must send valid rows, and its invalid rows must really be invalid."""
import random
from pathlib import Path

import pytest

from src.monitoring.traffic import make_invalid, sample_records
from src.serving.schema import SAMPLE_RECORD, validate_record

PROCESSED = Path("data/processed/telco_clean.parquet")


@pytest.mark.skipif(not PROCESSED.exists(), reason="processed data not pulled (dvc pull)")
def test_sampled_dataset_rows_pass_api_validation():
    for record in sample_records(50, seed=1):
        assert validate_record(record) == [], record


@pytest.mark.parametrize("seed", range(20))
def test_invalid_records_fail_api_validation(seed):
    assert validate_record(make_invalid(SAMPLE_RECORD, random.Random(seed))) != []


def test_make_invalid_does_not_modify_input():
    original = dict(SAMPLE_RECORD)
    make_invalid(SAMPLE_RECORD, random.Random(0))
    assert SAMPLE_RECORD == original
