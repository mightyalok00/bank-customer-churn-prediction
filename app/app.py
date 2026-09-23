"""Streamlit app for Bank Customer Churn Prediction.

Run from the project root:
    streamlit run app/app.py
"""
from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_PATH, PROCESSED_DATA_PATH, REPORTS_DIR

st.set_page_config(page_title="Bank Customer Churn Predictor", page_icon="🏦", layout="wide")
st.title("🏦 Bank Customer Churn Prediction")
st.caption("Decision Tree + Random Forest | Gini Impurity | Entropy | Feature Importance | Business Retention")

@st.cache_resource
def load_model():
    """Load the saved model once."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run: python src/train.py"
        )
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    """Load cleaned dataset for filter and dashboard views."""
    if not PROCESSED_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_DATA_PATH}. Run: python src/train.py"
        )
    return pd.read_csv(PROCESSED_DATA_PATH)

model = load_model()
df = load_data()

prediction_tab, dashboard_tab, model_tab, about_tab = st.tabs([
    "🔮 Predict Churn", "📊 Filter Dashboard", "🧠 Model Evidence", "ℹ️ About"
])

with prediction_tab:
    st.subheader("🔮 Single-Customer Churn Predictor")
    left, right = st.columns(2)
    with left:
        credit_score = st.slider("Credit Score", 300, 900, 650)
        age = st.slider("Age", 18, 100, 40)
        tenure = st.slider("Tenure", 0, 20, 5)
        balance = st.number_input("Balance", min_value=0.0, value=50000.0, step=1000.0)
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=100000.0, step=1000.0)
    with right:
        num_products = st.slider("Number of Products", 1, 4, 2)
        has_cr_card = st.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes ✅" if x == 1 else "No ❌")
        is_active_member = st.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes ✅" if x == 1 else "No ❌")
        rating = st.slider("Rating", 1, 5, 3)
        geography = st.selectbox("Geography", sorted(df["Geography"].dropna().unique().tolist()))
        gender = st.selectbox("Gender", sorted(df["Gender"].dropna().unique().tolist()))

    input_df = pd.DataFrame([{
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary,
        "Rating": rating,
        "Geography": geography,
        "Gender": gender,
    }])
    st.write("Selected customer profile")
    st.dataframe(input_df, width="stretch")

    if st.button("🚀 Predict Churn Risk"):
        probability = float(model.predict_proba(input_df)[0, 1])
        prediction = int(probability >= 0.50)
        col1, col2, col3 = st.columns(3)
        col1.metric("Churn Probability", f"{probability:.1%}")
        col2.metric("Prediction", "⚠️ Churn Risk" if prediction else "✅ Likely to Stay")
        col3.metric("Suggested Priority", "High" if probability >= 0.70 else "Medium" if probability >= 0.40 else "Low")
        if probability >= 0.70:
            st.error("High risk: assign relationship-manager call and targeted retention offer.")
        elif probability >= 0.40:
            st.warning("Medium risk: use app/email engagement campaign and monitor behavior.")
        else:
            st.success("Low risk: maintain regular engagement and avoid unnecessary retention spending.")

with dashboard_tab:
    st.subheader("📊 Filter Option: Explore Rows and Columns")
    row_limit = st.slider("Rows to show", 5, 500, 50)
    selected_columns = st.multiselect("Choose columns to display", df.columns.tolist(), default=df.columns.tolist())
    geo_filter = st.multiselect("Filter by Geography", sorted(df["Geography"].dropna().unique().tolist()), default=sorted(df["Geography"].dropna().unique().tolist()))
    gender_filter = st.multiselect("Filter by Gender", sorted(df["Gender"].dropna().unique().tolist()), default=sorted(df["Gender"].dropna().unique().tolist()))
    churn_filter = st.multiselect("Filter by Exited", [0, 1], default=[0, 1], format_func=lambda x: "Stayed" if x == 0 else "Churned")
    age_range = st.slider("Filter by Age", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))

    filtered = df[
        df["Geography"].isin(geo_filter)
        & df["Gender"].isin(gender_filter)
        & df["Exited"].isin(churn_filter)
        & df["Age"].between(age_range[0], age_range[1])
    ]
    st.write(f"Filtered shape: **{filtered.shape[0]} rows × {len(selected_columns)} columns**")
    st.dataframe(filtered[selected_columns].head(row_limit), width="stretch")
    st.download_button("⬇️ Download filtered CSV", filtered[selected_columns].to_csv(index=False), "filtered_churn_data.csv", "text/csv")

with model_tab:
    st.subheader("🧠 Model Evidence")
    comparison_path = REPORTS_DIR / "model_comparison.csv"
    feature_path = REPORTS_DIR / "feature_importance.csv"
    if comparison_path.exists():
        st.write("Model comparison")
        st.dataframe(pd.read_csv(comparison_path), width="stretch")
    if feature_path.exists():
        st.write("Top feature importances")
        st.dataframe(pd.read_csv(feature_path).head(15), width="stretch")
    for img_name, caption in [
        ("confusion_matrix.png", "Confusion Matrix"),
        ("roc_curve.png", "ROC Curve"),
        ("feature_importance.png", "Feature Importance"),
        ("decision_tree_visual.png", "Readable Decision Tree")
    ]:
        path = PROJECT_ROOT / "images" / img_name
        if path.exists():
            st.image(str(path), caption=caption, width="stretch")

with about_tab:
    st.subheader("ℹ️ Project Scope")
    st.write("""
    This is an educational portfolio project for binary churn classification.
    It demonstrates data cleaning, EDA, Gini Impurity, Decision Tree, Entropy,
    Random Forest, model comparison, feature importance, and business recommendations.
    It should not be used directly for real banking decisions without compliance,
    fairness testing, external validation, and monitoring.
    """)
