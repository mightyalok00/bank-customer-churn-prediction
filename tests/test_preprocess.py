import numpy as np
import pandas as pd
import pytest

from src.preprocess import COMBINED_COLUMN, clean_data, extract_geography_gender


def test_extract_geography_gender_from_combined_value():
    value = "{'Geography': 'Paris-France','Gender': 'Female'}"
    assert extract_geography_gender(value) == ("Paris-France", "Female")


def test_extract_geography_gender_normalizes_typo_and_handles_missing():
    assert extract_geography_gender('{"Geography":"Berlin-Gernay","Gender":"Male"}') == (
        "Berlin-Germany",
        "Male",
    )
    assert extract_geography_gender(np.nan) == ("Unknown", "Unknown")


def test_clean_data_extracts_categories_and_drops_identity_columns():
    raw = pd.DataFrame(
        {
            "id": [1], "CustomerId": [1001], "Surname": ["Example"],
            "CreditScore": [650], "Age": [140], "Tenure": [5],
            "Balance": [1000.0], "NumOfProducts": [2], "HasCrCard": [1],
            "IsActiveMember": [1], "EstimatedSalary": [50000.0],
            "Rating": [4], "Exited": [0],
            COMBINED_COLUMN: ["{'Geography':'Paris-France','Gender':'Female'}"],
        }
    )

    cleaned = clean_data(raw)

    assert {"id", "CustomerId", "Surname", COMBINED_COLUMN}.isdisjoint(cleaned.columns)
    assert cleaned.loc[0, "Geography"] == "Paris-France"
    assert cleaned.loc[0, "Gender"] == "Female"
    assert pd.isna(cleaned.loc[0, "Age"])


def test_clean_data_rejects_missing_model_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        clean_data(pd.DataFrame({"Exited": [0]}))
