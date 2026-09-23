"""Create model evaluation charts and feature-importance report."""
from __future__ import annotations

from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
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
    """Generate holdout, cross-validation, threshold, and explainability artifacts."""
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

    plt.style.use("seaborn-v0_8-whitegrid")

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

    PrecisionRecallDisplay.from_predictions(y_test, y_proba, name="Final model")
    plt.axhline(y=float(y_test.mean()), color="gray", linestyle="--", label="Churn baseline")
    plt.title("Precision–Recall Curve - Final Churn Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "precision_recall_curve.png", dpi=160)
    plt.close()

    CalibrationDisplay.from_predictions(y_test, y_proba, n_bins=10, strategy="quantile")
    plt.title("Probability Calibration - Final Churn Model")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "calibration_curve.png", dpi=160)
    plt.close()

    report = pd.DataFrame(
        classification_report(
            y_test,
            y_pred,
            target_names=["Stayed", "Churned"],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report.to_csv(REPORTS_DIR / "classification_report.csv")

    threshold_rows = []
    for threshold in np.arange(0.30, 0.71, 0.05):
        threshold_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, threshold_pred).ravel()
        threshold_rows.append({
            "threshold": round(float(threshold), 2),
            "accuracy": accuracy_score(y_test, threshold_pred),
            "precision": precision_score(y_test, threshold_pred, zero_division=0),
            "recall": recall_score(y_test, threshold_pred, zero_division=0),
            "f1_score": f1_score(y_test, threshold_pred, zero_division=0),
            "specificity": tn / (tn + fp),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        })
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(REPORTS_DIR / "threshold_analysis.csv", index=False)
    threshold_df.plot(
        x="threshold",
        y=["precision", "recall", "f1_score"],
        marker="o",
        figsize=(9, 5),
    )
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Decision Threshold Trade-offs")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "threshold_analysis.png", dpi=160)
    plt.close()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1_score": "f1",
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
        },
        n_jobs=1,
    )
    cv_summary = pd.DataFrame([
        {
            "metric": metric.removeprefix("test_"),
            "mean": values.mean(),
            "standard_deviation": values.std(ddof=1),
        }
        for metric, values in cv_scores.items()
        if metric.startswith("test_")
    ])
    cv_summary.to_csv(REPORTS_DIR / "cross_validation.csv", index=False)

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
    print("Evaluation charts and validation reports saved successfully.")


if __name__ == "__main__":
    main()
