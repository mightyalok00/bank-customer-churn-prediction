"""Central project paths for Bank Customer Churn Prediction."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROOT_DATA_PATH = PROJECT_ROOT / "Bank_churn.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Bank_churn.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "bank_churn_cleaned.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
IMAGES_DIR = PROJECT_ROOT / "images"


def get_data_path() -> Path:
    """Return the project-local raw CSV path."""
    for candidate in (ROOT_DATA_PATH, RAW_DATA_PATH):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Bank_churn.csv not found. Put it in the project root or data/raw/. "
        f"Checked: {ROOT_DATA_PATH} and {RAW_DATA_PATH}."
    )
