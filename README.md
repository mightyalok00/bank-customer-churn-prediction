# 🏦 Bank Customer Churn Prediction using Decision Tree and Random Forest

![Python](https://img.shields.io/badge/Python-3.12--3.14-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![CI](https://github.com/mightyalok00/bank-customer-churn-prediction/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Portfolio-Ready-brightgreen)

A complete binary classification portfolio project that predicts whether a bank customer will churn using **Gini Impurity**, **Decision Tree**, **Entropy**, and **Random Forest**.

## 🎯 Business Objective

Predict the target column `Exited`:

- `0` = customer stayed
- `1` = customer churned

The business goal is to help a bank identify high-risk customers early and take retention action before they leave.

## 📊 Dataset Summary

| Item | Value |
|---|---:|
| Raw rows | 165,034 |
| Raw columns | 14 |
| Cleaned rows | 165,031 |
| Cleaned columns | 12 |
| Target column | `Exited` |
| Churn rate | 21.16% |
| Stayed customers | 130,110 |
| Churned customers | 34,921 |

## 🧠 Models Used

1. Decision Tree using **Gini Impurity**
2. Decision Tree using **Entropy**
3. Random Forest
4. Tuned Random Forest

## ✅ Model Comparison

| Model | Test accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier score |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Decision Tree - Gini | 0.8036 | 0.5247 | 0.7632 | **0.6218** | 0.8695 | 0.6747 | 0.1434 |
| Decision Tree - Entropy | 0.8024 | 0.5227 | 0.7613 | 0.6198 | **0.8702** | **0.6819** | **0.1427** |
| Random Forest | 0.7971 | 0.5135 | **0.7796** | 0.6192 | 0.8700 | 0.6712 | 0.1468 |
| Tuned Random Forest | 0.7956 | 0.5112 | 0.7795 | 0.6175 | 0.8695 | 0.6701 | 0.1470 |

**Selected final model:** `Decision Tree - Gini` based on the strongest F1-score in this project run.

## 🧪 Validation and Threshold Analysis

Five-fold stratified cross-validation confirms that the selected model is stable rather than dependent on one train/test split.

| Metric | CV mean | Standard deviation |
|:--|--:|--:|
| Accuracy | 0.8029 | 0.0063 |
| Precision | 0.5238 | 0.0111 |
| Recall | 0.7642 | 0.0123 |
| F1-score | 0.6214 | 0.0054 |
| ROC-AUC | 0.8719 | 0.0030 |
| PR-AUC | 0.6815 | 0.0041 |

The default threshold of `0.50` identifies 5,330 churners with 76.32% recall. The included threshold report demonstrates the business trade-off: lowering the threshold finds more churners but increases false-positive retention contacts.

## 🔍 Key Business Findings

- Overall churn rate is **21.16%**.
- Highest churn geography is **Berlin-Germany**.
- Highest churn age group is **51-60**.
- Top model features include: Age, NumOfProducts, IsActiveMember, Balance, Geography_Berlin-Germany.
- The final model correctly identifies 5,330 churners and misses 1,654 churners in the test set at the 0.50 threshold.

## 📚 Complete Question Bank

`reports/all_216_question_answers.md` contains the original 216 project, interview, business, ethics, deployment, and portfolio questions with evidence-based answers from the current project run.

- All 216 prompts match the original Word question bank.
- Every prompt has a substantive answer in the report and notebook.
- The requested executive summary for question 212 is exactly 150 words.

## 🖥️ Streamlit Application

The app includes:

- A batched customer-input form with an adjustable decision threshold
- Churn probability, classification, review priority, and a retention recommendation
- Filterable customer exploration with downloadable results
- Holdout, cross-validation, threshold, calibration, and feature-importance evidence
- Cached model/data loading, responsible-use guidance, and a native responsive theme

## 📈 Visual Outputs

- `images/confusion_matrix.png`
- `images/roc_curve.png`
- `images/precision_recall_curve.png`
- `images/calibration_curve.png`
- `images/threshold_analysis.png`
- `images/feature_importance.png`
- `images/decision_tree_visual.png`

## 🚀 Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-dev.txt
python src/train.py
python src/evaluate.py
python -m pytest
streamlit run streamlit_app.py
```

Dependencies are pinned to exact tested versions. GitHub Actions repeats the automated test suite for every push and pull request.

## 📁 Repository Structure

```text
bank-customer-churn-prediction/
├── app/
│   └── app.py
├── .github/workflows/
│   └── ci.yml
├── .streamlit/
│   └── config.toml
├── data/
│   └── processed/bank_churn_cleaned.csv
├── images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── calibration_curve.png
│   ├── threshold_analysis.png
│   ├── feature_importance.png
│   └── decision_tree_visual.png
├── models/
│   ├── churn_model.pkl
│   └── model_metadata.json
├── notebooks/
│   └── bank_customer_churn_prediction.ipynb
├── reports/
│   ├── model_comparison.csv
│   ├── cross_validation.csv
│   ├── threshold_analysis.csv
│   ├── classification_report.csv
│   ├── feature_importance.csv
│   ├── project_report.md
│   └── all_216_question_answers.md
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── predict.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   ├── test_app.py
│   ├── test_predict.py
│   └── test_preprocess.py
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── Bank_churn.csv
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── streamlit_app.py
└── .gitignore
```

## ⚠️ Limitations

This project uses an educational Kaggle-style dataset. It should not be used directly for real banking decisions without fairness testing, compliance review, external validation, monitoring, and human oversight.

The raw and cleaned CSV files are included for reproducibility and contain educational/synthetic-style records. Real customer data and credentials must never be committed.

## 🤝 Contributing and License

Development and pull-request guidance is provided in `CONTRIBUTING.md`. The project is available under the MIT License.

## 👤 Author

**Alok Agarwal**  
Data Analytics • Data Science • Machine Learning • SEO/Digital Marketing
