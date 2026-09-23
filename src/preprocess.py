"""Data loading and preprocessing helpers.

The uploaded CSV contains one combined column named `Geography:str,Gender:str`.
This file extracts the two useful categorical columns from it and keeps the
modeling workflow clean, reproducible, and easy to explain.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

TARGET_COLUMN = "Exited"
ID_COLUMNS = ["id", "CustomerId", "Surname"]
COMBINED_COLUMN = "Geography:str,Gender:str"
MODEL_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary", "Rating",
    "Geography", "Gender",
]


def extract_geography_gender(value: object) -> tuple[str, str]:
    """Extract Geography and Gender from the combined string column.

    Example value:
    {'Geography': 'Paris-France','Gender': 'Male'}
    """
    if pd.isna(value):
        return "Unknown", "Unknown"

    text = str(value).strip()
    parsed: object = None
    for parser in (ast.literal_eval, json.loads):
        try:
            parsed = parser(text)
            if isinstance(parsed, dict):
                break
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            parsed = None

    if isinstance(parsed, dict):
        geography = str(parsed.get("Geography", "Unknown")).strip()
        gender = str(parsed.get("Gender", "Unknown")).strip()
    else:
        geography_match = re.search(
            r"[\"']?Geography[\"']?\s*:\s*[\"']([^\"']*)[\"']", text, re.IGNORECASE
        )
        gender_match = re.search(
            r"[\"']?Gender[\"']?\s*:\s*[\"']([^\"']*)[\"']", text, re.IGNORECASE
        )
        geography = geography_match.group(1).strip() if geography_match else "Unknown"
        gender = gender_match.group(1).strip() if gender_match else "Unknown"
    geography = geography.replace("Berlin-Gernay", "Berlin-Germany")
    if geography == "" or geography.lower() in {"nan", "none"}:
        geography = "Unknown"
    if gender == "" or gender.lower() in {"nan", "none"}:
        gender = "Unknown"
    return geography, gender


def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    """Load raw churn data and normalize accidental header whitespace/BOMs."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(column).lstrip("\ufeff").strip() for column in df.columns]
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataset without leaking target information.

    Main actions:
    - Extract Geography and Gender.
    - Convert numeric-looking columns to numeric.
    - Remove rows with invalid target values.
    - Replace unrealistic values with NaN so model pipelines can impute them.
    - Drop exact duplicate rows.
    """
    clean_df = df.copy()

    if COMBINED_COLUMN in clean_df.columns:
        extracted = clean_df[COMBINED_COLUMN].apply(extract_geography_gender)
        extracted_geography = extracted.apply(lambda item: item[0])
        extracted_gender = extracted.apply(lambda item: item[1])
        if "Geography" not in clean_df.columns:
            clean_df["Geography"] = extracted_geography
        else:
            missing = clean_df["Geography"].isna() | clean_df["Geography"].astype(str).str.strip().eq("")
            clean_df.loc[missing, "Geography"] = extracted_geography[missing]
        if "Gender" not in clean_df.columns:
            clean_df["Gender"] = extracted_gender
        else:
            missing = clean_df["Gender"].isna() | clean_df["Gender"].astype(str).str.strip().eq("")
            clean_df.loc[missing, "Gender"] = extracted_gender[missing]
        clean_df = clean_df.drop(columns=[COMBINED_COLUMN])

    required_columns = set(MODEL_FEATURES + [TARGET_COLUMN])
    missing_columns = sorted(required_columns.difference(clean_df.columns))
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns after parsing: "
            + ", ".join(missing_columns)
        )

    numeric_cols = [
        "id", "CustomerId", "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Rating", "Exited"
    ]
    for col in numeric_cols:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    clean_df = clean_df[clean_df[TARGET_COLUMN].isin([0, 1])].copy()
    clean_df[TARGET_COLUMN] = clean_df[TARGET_COLUMN].astype(int)

    if "Age" in clean_df.columns:
        clean_df.loc[(clean_df["Age"] < 18) | (clean_df["Age"] > 100), "Age"] = np.nan
    if "CreditScore" in clean_df.columns:
        clean_df.loc[(clean_df["CreditScore"] < 300) | (clean_df["CreditScore"] > 900), "CreditScore"] = np.nan
    if "Tenure" in clean_df.columns:
        clean_df.loc[(clean_df["Tenure"] < 0) | (clean_df["Tenure"] > 20), "Tenure"] = np.nan
    if "Balance" in clean_df.columns:
        clean_df.loc[clean_df["Balance"] < 0, "Balance"] = np.nan
    if "EstimatedSalary" in clean_df.columns:
        clean_df.loc[clean_df["EstimatedSalary"] < 0, "EstimatedSalary"] = np.nan
    if "NumOfProducts" in clean_df.columns:
        clean_df.loc[(clean_df["NumOfProducts"] < 1) | (clean_df["NumOfProducts"] > 10), "NumOfProducts"] = np.nan

    clean_df = clean_df.drop(columns=ID_COLUMNS, errors="ignore")
    return clean_df.drop_duplicates().reset_index(drop=True)


def get_feature_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Create features X, target y, numeric features, and categorical features."""
    X = df.drop(columns=ID_COLUMNS + [TARGET_COLUMN], errors="ignore")
    y = df[TARGET_COLUMN].astype(int)
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = [col for col in X.columns if col not in categorical_features]
    return X, y, numeric_features, categorical_features
