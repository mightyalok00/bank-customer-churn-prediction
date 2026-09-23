"""Create model evaluation charts and feature-importance report."""
from __future__ import annotations

from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import IMAGES_DIR, MODEL_PATH, PROCESSED_DATA_PATH, REPORTS_DIR
from src.preprocess import get_feature_target
from src.train import build_preprocessor


def get_feature_names(pipeline: Pipeline, numeric_features, categorical_features) -> list[str]:
    """Return feature names after preprocessing."""
    preprocessor = pipeline.named_steps["preprocessor"]
    names = list(numeric_features)
    if categorical_features:
        onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        names.extend(onehot.get_feature_names_out(categorical_features).tolist())
    return names


def main() -> None:
    """Generate confusion matrix, ROC curve, feature importance, and tree image."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X, y, numeric_features, categorical_features = get_feature_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipeline = joblib.load(MODEL_PATH)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Stayed", "Churned"])
    plt.title("Confusion Matrix - Final Churn Model")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve - Final Churn Model")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "roc_curve.png", dpi=160)
    plt.close()

    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(pipeline, numeric_features, categorical_features)
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        fi = fi.sort_values("importance", ascending=False)
        fi.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
        top_fi = fi.head(15).iloc[::-1]
        plt.figure(figsize=(9, 6))
        plt.barh(top_fi["feature"], top_fi["importance"])
        plt.title("Top 15 Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "feature_importance.png", dpi=160)
        plt.close()

    # Shallow readable tree for Gini explanation.
    shallow_tree = Pipeline([
        ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
        ("model", DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_leaf=300, random_state=42)),
    ])
    shallow_tree.fit(X_train, y_train)
    shallow_names = get_feature_names(shallow_tree, numeric_features, categorical_features)
    plt.figure(figsize=(24, 10))
    plot_tree(
        shallow_tree.named_steps["model"],
        feature_names=shallow_names,
        class_names=["Stayed", "Churned"],
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.title("Decision Tree Visualization - Gini Criterion")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "decision_tree_visual.png", dpi=160)
    plt.close()
    print("Charts saved in images/ and feature importance saved in reports/.")


if __name__ == "__main__":
    main()
