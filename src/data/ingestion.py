"""Data ingestion for the Telco churn pipeline.

The default source is the DVC-managed local CSV in ``configs/pipeline_config.yaml``.
An Airflow task may instead pass an ``s3://bucket/key.csv`` URI. Ingestion only
retrieves data; ``src.data.validation`` remains responsible for schema checks.
"""
from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from src.features.preprocessing import load_config

logger = logging.getLogger(__name__)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return ``(bucket, key)`` for a complete S3 URI."""
    parsed = urlparse(uri)
    key = parsed.path.lstrip("/")
    if parsed.scheme != "s3" or not parsed.netloc or not key:
        raise ValueError("S3 source must use the form s3://<bucket>/<key>.csv")
    return parsed.netloc, key


def load_s3_csv(uri: str, s3_client: Any | None = None) -> pd.DataFrame:
    """Read a CSV from S3 without writing a second unversioned raw-data copy.

    ``s3_client`` is injectable so Airflow can provide its authenticated client
    and tests do not need AWS credentials. When it is omitted, boto3 must be
    installed in the orchestration environment.
    """
    bucket, key = parse_s3_uri(uri)
    if s3_client is None:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "Reading from S3 requires boto3 in the orchestration environment."
            ) from exc
        s3_client = boto3.client("s3")

    response = s3_client.get_object(Bucket=bucket, Key=key)
    payload = response["Body"].read()
    return pd.read_csv(io.BytesIO(payload))


def load_raw_data(
    source: str | Path | None = None,
    *,
    config_path: str = "configs/pipeline_config.yaml",
    s3_client: Any | None = None,
) -> pd.DataFrame:
    """Load raw data from the configured local DVC file or an explicit S3 URI."""
    resolved_source = str(source) if source is not None else load_config(config_path)["raw_path"]
    if resolved_source.startswith("s3://"):
        return load_s3_csv(resolved_source, s3_client=s3_client)

    path = Path(resolved_source)
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw data file not found: {path}. Run `dvc pull` or provide an s3:// source."
        )
    return pd.read_csv(path)


def main(argv: list[str] | None = None) -> int:
    """Load a source and report its shape; validation is a separate pipeline stage."""
    parser = argparse.ArgumentParser(description="Load Telco churn raw data.")
    parser.add_argument("--source", help="local CSV path or s3://bucket/key.csv")
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    args = parser.parse_args(argv)

    try:
        frame = load_raw_data(args.source, config_path=args.config)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, pd.errors.ParserError) as exc:
        logger.error("Data ingestion failed: %s", exc)
        return 1

    source = args.source or load_config(args.config)["raw_path"]
    logger.info("Loaded %s: %d rows, %d columns", source, *frame.shape)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
