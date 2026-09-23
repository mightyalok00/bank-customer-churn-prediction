import pytest

from src.predict import FEATURE_COLUMNS, make_customer_frame, retention_action, risk_priority


def customer_values():
    return {
        "CreditScore": 650, "Age": 40, "Tenure": 5, "Balance": 50000.0,
        "NumOfProducts": 2, "HasCrCard": 1, "IsActiveMember": 1,
        "EstimatedSalary": 100000.0, "Rating": 3,
        "Geography": "Paris-France", "Gender": "Female",
    }


def test_make_customer_frame_preserves_training_order():
    frame = make_customer_frame(**customer_values())
    assert frame.columns.tolist() == FEATURE_COLUMNS
    assert frame.shape == (1, 11)


def test_make_customer_frame_rejects_missing_features():
    values = customer_values()
    values.pop("Age")
    with pytest.raises(ValueError, match="Age"):
        make_customer_frame(**values)


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.20, "Low"), (0.40, "Medium"), (0.69, "Medium"), (0.70, "High")],
)
def test_risk_priority_boundaries(probability, expected):
    assert risk_priority(probability) == expected


def test_retention_action_and_invalid_probability():
    assert "relationship-manager" in retention_action(0.80)
    with pytest.raises(ValueError):
        risk_priority(1.01)
