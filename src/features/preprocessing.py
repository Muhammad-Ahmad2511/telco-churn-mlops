# src/features/preprocessing.py
"""Cleaning, splitting and encoding for Telco churn (Person A).

Raw CSV is never edited. All fixes happen here, in code.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def load_config(path: str = "configs/pipeline_config.yaml") -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)["data"]


def clean_raw(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Fix TotalCharges, encode the target, drop ID and sensitive columns."""
    out = df.copy()

    # TotalCharges arrives as text. The 11 blank rows all have tenure == 0.
    total = pd.to_numeric(out["TotalCharges"].astype("string").str.strip(), errors="coerce")
    fixable = total.isna() & (out["tenure"] == 0)
    total = total.mask(fixable, 0.0)
    if total.isna().any():
        raise ValueError(
            f"{int(total.isna().sum())} TotalCharges values are missing "
            "and tenure is not 0; refusing to guess."
        )
    out["TotalCharges"] = total.astype("float64")
    logger.info("Set TotalCharges to 0.0 for %d rows with tenure 0", int(fixable.sum()))

    target = cfg["target"]
    out[target] = (out[target] == "Yes").astype("int64")

    return out.drop(columns=cfg["drop_columns"])


def split_data(df: pd.DataFrame, cfg: dict):
    """Stratified train/test split. Returns X_train, X_test, y_train, y_test."""
    target = cfg["target"]
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X,
        y,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=y,
    )


def split_train_val_test(df: pd.DataFrame, cfg: dict):
    """Stratified train/validation/test split.

    The test set is identical to split_data() (same size and seed), so tuned models
    are scored on exactly the rows the baseline used.
    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    """
    X_trainval, X_test, y_trainval, y_test = split_data(df, cfg)
    val_share = cfg["val_size"] / (1 - cfg["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_share,
        random_state=cfg["random_state"],
        stratify=y_trainval,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Scale numeric columns, one-hot encode the rest. Unseen categories are ignored."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )