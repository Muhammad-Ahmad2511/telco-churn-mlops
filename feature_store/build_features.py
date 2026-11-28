"""Build the Feast source table from the processed churn data.

Runs in the main project environment (pandas only, no Feast needed):

    python -m feature_store.build_features

The raw data has no timestamp, so every row gets the time of the snapshot as its
event_timestamp. Quality checks run before anything is written.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from feature_store.feature_repo.features import (
    ENTITY_KEY,
    EVENT_TIMESTAMP,
    FEATURE_COLUMNS,
    MAX_AGE_DAYS,
    MAX_NULL_FRACTION,
    NON_NEGATIVE,
)

DEFAULT_INPUT = Path("data/processed/telco_clean.parquet")
DEFAULT_OUTPUT = Path("feature_store/feature_repo/data/customer_features.parquet")


def build_feature_frame(
    processed: pd.DataFrame, event_timestamp: datetime | None = None
) -> pd.DataFrame:
    """Select key + features, add event_timestamp. Raises on missing or duplicate keys."""
    missing = [c for c in [ENTITY_KEY, *FEATURE_COLUMNS] if c not in processed.columns]
    if missing:
        raise ValueError(f"processed data is missing columns: {missing}")
    if processed[ENTITY_KEY].duplicated().any():
        raise ValueError(f"duplicate values in entity key {ENTITY_KEY}")

    frame = processed[[ENTITY_KEY, *FEATURE_COLUMNS]].copy()
    for column, dtype in FEATURE_COLUMNS.items():
        if dtype in ("int64", "float64"):
            frame[column] = frame[column].astype(dtype)

    stamp = pd.Timestamp(event_timestamp or datetime.now(UTC))
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    frame[EVENT_TIMESTAMP] = stamp
    return frame.reset_index(drop=True)


def check_feature_quality(frame: pd.DataFrame, now: datetime | None = None) -> list[str]:
    """Return a list of problems (empty list = the feature table is valid)."""
    problems: list[str] = []
    expected = [ENTITY_KEY, EVENT_TIMESTAMP, *FEATURE_COLUMNS]
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        return [f"missing columns: {missing}"]

    if frame[ENTITY_KEY].isna().any():
        problems.append("null entity keys")
    if frame[ENTITY_KEY].duplicated().any():
        problems.append("duplicate entity keys")
    if frame[EVENT_TIMESTAMP].isna().any():
        problems.append("null event timestamps")

    for column in FEATURE_COLUMNS:
        share = frame[column].isna().mean()
        if share > MAX_NULL_FRACTION:
            problems.append(f"{column}: {share:.1%} nulls (max {MAX_NULL_FRACTION:.0%})")

    for column in NON_NEGATIVE:
        if (frame[column] < 0).any():
            problems.append(f"{column}: negative values")
    if not frame["SeniorCitizen"].dropna().isin([0, 1]).all():
        problems.append("SeniorCitizen: values outside {0, 1}")

    reference = pd.Timestamp(now or datetime.now(UTC))
    if reference.tzinfo is None:
        reference = reference.tz_localize("UTC")
    newest = frame[EVENT_TIMESTAMP].max()
    if pd.notna(newest) and reference - newest > timedelta(days=MAX_AGE_DAYS):
        problems.append(f"stale snapshot: newest timestamp {newest} older than {MAX_AGE_DAYS} days")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Feast source table.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frame = build_feature_frame(pd.read_parquet(args.input))
    problems = check_feature_quality(frame)
    if problems:
        raise SystemExit("feature quality check failed:\n- " + "\n- ".join(problems))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(f"wrote {len(frame)} rows x {frame.shape[1]} columns to {args.output}")


if __name__ == "__main__":
    main()
