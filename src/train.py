"""Train Decision Tree and Random Forest churn models.

Models trained:
1. Decision Tree with Gini Impurity
2. Decision Tree with Entropy
3. Random Forest
4. Tuned Random Forest
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_data_path, METADATA_PATH, MODEL_PATH, PROCESSED_DATA_PATH, REPORTS_DIR
from src.preprocess import clean_data, get_feature_target, load_raw_data


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    """Build preprocessing for numeric and categorical columns."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features)
    ])


def evaluate_model(name: str, pipeline: Pipeline, X_train, X_test, y_train, y_test) -> dict:
    """Return train/test metrics for a fitted model."""
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    return {
        "model": name,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "precision": precision_score(y_test, test_pred, zero_division=0),
        "recall": recall_score(y_test, test_pred, zero_division=0),
        "f1_score": f1_score(y_test, test_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, test_proba),
        "pr_auc": average_precision_score(y_test, test_proba),
        "brier_score": brier_score_loss(y_test, test_proba),
        "overfit_gap_accuracy": train_accuracy - test_accuracy,
    }


def main() -> None:
    """Run the full model-training workflow and save outputs."""
    csv_path = get_data_path()
    raw_df = load_raw_data(csv_path)
    clean_df = clean_data(raw_df)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(PROCESSED_DATA_PATH, index=False)

    X, y, numeric_features, categorical_features = get_feature_target(clean_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = {
        "Decision Tree - Gini": DecisionTreeClassifier(
            criterion="gini", max_depth=6, min_samples_leaf=100, class_weight="balanced", random_state=42
        ),
        "Decision Tree - Entropy": DecisionTreeClassifier(
            criterion="entropy", max_depth=6, min_samples_leaf=100, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=40, max_depth=8, min_samples_leaf=80, class_weight="balanced", random_state=42, n_jobs=1
        ),
    }

    fitted = {}
    metrics = []
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        metrics.append(evaluate_model(name, pipe, X_train, X_test, y_train, y_test))

    # Compact tuning grid keeps the project reproducible on normal laptops.
    rf_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1)),
    ])
    param_grid = {
        "model__n_estimators": [50],
        "model__max_depth": [8],
        "model__min_samples_leaf": [80],
        "model__max_features": ["sqrt"],
    }
    search = GridSearchCV(rf_pipe, param_grid, scoring="f1", cv=2, n_jobs=1)
    search.fit(X_train, y_train)
    fitted["Tuned Random Forest"] = search.best_estimator_
    metrics.append(evaluate_model("Tuned Random Forest", search.best_estimator_, X_train, X_test, y_train, y_test))

    metrics_df = pd.DataFrame(metrics).sort_values("f1_score", ascending=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    best_model_name = metrics_df.iloc[0]["model"]
    best_pipeline = fitted[best_model_name]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    metadata = {
        "csv_path_used": str(csv_path),
        "raw_shape": list(raw_df.shape),
        "clean_shape": list(clean_df.shape),
        "target_column": "Exited",
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "best_model": str(best_model_name),
        "best_params": search.best_params_,
        "random_state": 42,
        "test_size": 0.20,
        "decision_threshold": 0.50,
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metrics_df.to_string(index=False))
    print(f"Best model saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
