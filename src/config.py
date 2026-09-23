"""Central project paths for Bank Customer Churn Prediction."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Bank_churn.csv"
ROOT_DATA_PATH = PROJECT_ROOT / "Bank_churn.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "bank_churn_cleaned.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
IMAGES_DIR = PROJECT_ROOT / "images"


def get_data_path() -> Path:
    """Return the preferred project-local raw CSV path.

    The clean repository structure is `data/raw/Bank_churn.csv`.
    A root-level `Bank_churn.csv` fallback is kept only for older local copies
    so existing notebooks or Streamlit deployments do not break immediately.
    """
    for candidate in (RAW_DATA_PATH, ROOT_DATA_PATH):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Bank_churn.csv not found. Put it in data/raw/Bank_churn.csv. "
        f"Checked: {RAW_DATA_PATH} and legacy fallback {ROOT_DATA_PATH}."
    )
