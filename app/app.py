"""Interactive bank-customer churn prediction and model-evidence dashboard."""
from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_PATH, PROCESSED_DATA_PATH, REPORTS_DIR
from src.predict import make_customer_frame, retention_action, risk_priority

st.set_page_config(
    page_title="Bank customer churn predictor",
    page_icon=":material/account_balance:",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Load the fitted pipeline once per app process."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run python src/train.py.")
    return joblib.load(MODEL_PATH)


@st.cache_data(max_entries=2)
def load_data() -> pd.DataFrame:
    """Load the cleaned portfolio dataset."""
    if not PROCESSED_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_DATA_PATH}. Run python src/train.py."
        )
    return pd.read_csv(PROCESSED_DATA_PATH)


st.title("Bank customer churn predictor", icon=":material/account_balance:")
st.caption(
    "Decision-support demo using a balanced Gini decision tree. "
    "Predictions should be reviewed by a person before any customer action."
)

try:
    model = load_model()
    df = load_data()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(str(error), icon=":material/error:")
    st.stop()

with st.container(horizontal=True):
    st.metric("Cleaned customers", f"{len(df):,}", border=True)
    st.metric("Observed churn", f"{df['Exited'].mean():.1%}", border=True)
    st.metric("Final model", "Decision Tree - Gini", border=True)
    st.metric("Test ROC-AUC", "0.870", border=True)

view = st.segmented_control(
    "Workspace",
    ["Predict risk", "Explore customers", "Model evidence", "About"],
    default="Predict risk",
    key="main_view",
)

if view == "Predict risk":
    st.subheader("Score one customer", icon=":material/person_search:")
    st.caption("Adjust the profile, choose an operating threshold, and submit once.")

    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            credit_score = st.slider("Credit score", 300, 900, 650, key="credit_score")
            age = st.slider("Age", 18, 100, 40, key="age")
            tenure = st.slider("Tenure (years)", 0, 20, 5, key="tenure")
            balance = st.number_input(
                "Account balance", min_value=0.0, value=50000.0, step=1000.0, key="balance"
            )
            estimated_salary = st.number_input(
                "Estimated salary", min_value=0.0, value=100000.0, step=1000.0, key="salary"
            )
        with right:
            num_products = st.slider("Number of products", 1, 4, 2, key="products")
            has_cr_card = st.segmented_control(
                "Has a credit card", [1, 0], default=1,
                format_func=lambda value: "Yes" if value == 1 else "No",
                key="credit_card",
            )
            is_active_member = st.segmented_control(
                "Active member", [1, 0], default=1,
                format_func=lambda value: "Yes" if value == 1 else "No",
                key="active_member",
            )
            rating = st.slider("Customer rating", 1, 5, 3, key="rating")
            geography = st.selectbox(
                "Geography", sorted(df["Geography"].dropna().unique()), key="geography"
            )
            gender_options = sorted(df["Gender"].dropna().unique())
            gender = st.segmented_control(
                "Gender", gender_options, default=gender_options[0], key="gender"
            )

        decision_threshold = st.slider(
            "Decision threshold", 0.30, 0.70, 0.50, 0.05,
            help="Lower thresholds find more churners but create more false alarms.",
            key="decision_threshold",
        )
        submitted = st.form_submit_button(
            "Predict churn risk", type="primary", icon=":material/query_stats:",
            key="predict_submit",
        )

    input_df = make_customer_frame(
        CreditScore=credit_score,
        Age=age,
        Tenure=tenure,
        Balance=balance,
        NumOfProducts=num_products,
        HasCrCard=has_cr_card,
        IsActiveMember=is_active_member,
        EstimatedSalary=estimated_salary,
        Rating=rating,
        Geography=geography,
        Gender=gender,
    )

    if submitted:
        probability = float(model.predict_proba(input_df)[0, 1])
        prediction = int(probability >= decision_threshold)
        priority = risk_priority(probability)
        with st.container(horizontal=True):
            st.metric("Churn probability", f"{probability:.1%}", border=True)
            st.metric(
                "Threshold result", "Churn risk" if prediction else "Likely to stay", border=True
            )
            st.metric("Review priority", priority, border=True)
        st.progress(probability, text=f"Risk score: {probability:.1%}")
        action = retention_action(probability)
        if priority == "High":
            st.error(action, icon=":material/priority_high:")
        elif priority == "Medium":
            st.warning(action, icon=":material/warning:")
        else:
            st.success(action, icon=":material/check_circle:")
        with st.expander("Customer profile used", icon=":material/badge:"):
            st.dataframe(input_df, hide_index=True)

if view == "Explore customers":
    st.subheader("Explore cleaned customers", icon=":material/filter_alt:")
    geo_filter = st.multiselect(
        "Geography", sorted(df["Geography"].dropna().unique()),
        default=sorted(df["Geography"].dropna().unique()), key="geo_filter"
    )
    gender_filter = st.pills(
        "Gender", sorted(df["Gender"].dropna().unique()),
        default=sorted(df["Gender"].dropna().unique()), selection_mode="multi",
        key="gender_filter"
    )
    churn_filter = st.segmented_control(
        "Customer outcome", ["All", "Stayed", "Churned"], default="All",
        key="churn_filter"
    )
    age_range = st.slider(
        "Age range", int(df["Age"].min()), int(df["Age"].max()),
        (int(df["Age"].min()), int(df["Age"].max())), key="age_range"
    )

    filtered = df[
        df["Geography"].isin(geo_filter)
        & df["Gender"].isin(gender_filter)
        & df["Age"].between(*age_range)
    ]
    if churn_filter != "All":
        filtered = filtered[filtered["Exited"] == (1 if churn_filter == "Churned" else 0)]

    with st.container(horizontal=True):
        st.metric("Matching customers", f"{len(filtered):,}", border=True)
        st.metric(
            "Churn rate", f"{filtered['Exited'].mean():.1%}" if len(filtered) else "—",
            border=True
        )
        st.metric(
            "Average age", f"{filtered['Age'].mean():.1f}" if len(filtered) else "—",
            border=True
        )
    st.dataframe(filtered.head(250), hide_index=True, key="customer_table")
    st.download_button(
        "Download filtered CSV", filtered.to_csv(index=False), "filtered_churn_data.csv",
        "text/csv", icon=":material/download:"
    )

if view == "Model evidence":
    st.subheader("Model evidence", icon=":material/analytics:")
    comparison_path = REPORTS_DIR / "model_comparison.csv"
    feature_path = REPORTS_DIR / "feature_importance.csv"
    threshold_path = REPORTS_DIR / "threshold_analysis.csv"
    cv_path = REPORTS_DIR / "cross_validation.csv"

    if comparison_path.exists():
        with st.container(border=True):
            st.markdown("**Holdout model comparison**")
            st.dataframe(pd.read_csv(comparison_path), hide_index=True)
    if cv_path.exists():
        with st.container(border=True):
            st.markdown("**Five-fold cross-validation**")
            st.dataframe(pd.read_csv(cv_path), hide_index=True)
    if threshold_path.exists():
        with st.container(border=True):
            st.markdown("**Decision-threshold trade-offs**")
            st.line_chart(
                pd.read_csv(threshold_path), x="threshold",
                y=["precision", "recall", "f1_score"]
            )
    if feature_path.exists():
        with st.container(border=True):
            st.markdown("**Top global feature importances**")
            st.bar_chart(pd.read_csv(feature_path).head(12).set_index("feature"))

    charts = [
        ("confusion_matrix.png", "Confusion matrix"),
        ("roc_curve.png", "ROC curve"),
        ("precision_recall_curve.png", "Precision–recall curve"),
        ("calibration_curve.png", "Probability calibration"),
        ("threshold_analysis.png", "Threshold trade-offs"),
        ("feature_importance.png", "Feature importance"),
        ("decision_tree_visual.png", "Readable decision tree"),
    ]
    for index in range(0, len(charts), 2):
        columns = st.columns(2)
        for column, (image_name, caption) in zip(columns, charts[index:index + 2]):
            image_path = PROJECT_ROOT / "images" / image_name
            if image_path.exists():
                with column.container(border=True):
                    st.image(str(image_path), caption=caption)

if view == "About":
    st.subheader("Scope and responsible use", icon=":material/info:")
    st.markdown(
        """
        This educational portfolio project demonstrates reproducible preprocessing,
        Decision Trees using Gini and Entropy, Random Forest comparison, threshold
        evaluation, explainability, and an interactive decision-support workflow.

        The model is associative, not causal. A real bank would need representative
        temporal data, calibration, security controls, fairness and compliance review,
        monitored pilots, and human oversight before using it with customers.
        """
    )
    st.link_button(
        "View the GitHub repository",
        "https://github.com/mightyalok00/bank-customer-churn-prediction",
        icon=":material/code:",
    )
