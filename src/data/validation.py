# src/data/validation.py
"""Raw data validation with Pandera (Person A).

"""
from __future__ import annotations

import logging
import sys

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

logger = logging.getLogger(__name__)

YES_NO = ["Yes", "No"]
YES_NO_NO_INTERNET = ["Yes", "No", "No internet service"]


def _blank_or_numeric(series: pd.Series) -> pd.Series:
    """TotalCharges arrives as text. Each value must be blank or a number.

    The 11 blank values (tenure = 0) are a known quirk, fixed later in code.
    """
    text = series.astype("string").str.strip()
    is_blank = text.eq("").fillna(False)
    is_number = pd.to_numeric(text, errors="coerce").notna()
    return is_blank | is_number


RAW_SCHEMA = DataFrameSchema(
    columns={
        "customerID": Column(nullable=False, unique=True),
        "gender": Column(checks=Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(int, checks=Check.isin([0, 1])),
        "Partner": Column(checks=Check.isin(YES_NO)),
        "Dependents": Column(checks=Check.isin(YES_NO)),
        "tenure": Column(int, checks=Check.ge(0)),
        "PhoneService": Column(checks=Check.isin(YES_NO)),
        "MultipleLines": Column(checks=Check.isin(["Yes", "No", "No phone service"])),
        "InternetService": Column(checks=Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineSecurity": Column(checks=Check.isin(YES_NO_NO_INTERNET)),
        "OnlineBackup": Column(checks=Check.isin(YES_NO_NO_INTERNET)),
        "DeviceProtection": Column(checks=Check.isin(YES_NO_NO_INTERNET)),
        "TechSupport": Column(checks=Check.isin(YES_NO_NO_INTERNET)),
        "StreamingTV": Column(checks=Check.isin(YES_NO_NO_INTERNET)),
        "StreamingMovies": Column(checks=Check.isin(YES_NO_NO_INTERNET)),
        "Contract": Column(checks=Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaperlessBilling": Column(checks=Check.isin(YES_NO)),
        "PaymentMethod": Column(
            checks=Check.isin(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            )
        ),
        "MonthlyCharges": Column(float, checks=Check.ge(0)),
        "TotalCharges": Column(
            nullable=False,
            checks=Check(
                _blank_or_numeric,
                error="TotalCharges must be blank or numeric text",
            ),
        ),
        "Churn": Column(checks=Check.isin(YES_NO)),
    },
    strict=True,  # extra or missing columns fail
    name="raw_telco_churn",
)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw dataframe. Raises pandera SchemaErrors on failure."""
    validated = RAW_SCHEMA.validate(df, lazy=True)  # lazy: report all errors
    logger.info("Raw validation passed: %d rows, %d columns", *validated.shape)
    return validated


def validate_file(path: str) -> pd.DataFrame:
    """Read a CSV and validate it."""
    return validate_raw(pd.read_csv(path))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/telco_churn.csv"
    try:
        validate_file(csv_path)
    except pa.errors.SchemaErrors as err:
        logger.error("Validation FAILED:\n%s", err.failure_cases)
        sys.exit(1)