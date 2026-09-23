# 🏦 Bank Customer Churn Prediction using Decision Tree and Random Forest

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
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

| model                   |   train_accuracy |   test_accuracy |   precision |   recall |   f1_score |   roc_auc |   overfit_gap_accuracy |
|:------------------------|-----------------:|----------------:|------------:|---------:|-----------:|----------:|-----------------------:|
| Decision Tree - Gini    |         0.806262 |        0.803587 |    0.524658 | 0.763173 |   0.621828 |  0.869526 |            0.002675 |
| Decision Tree - Entropy |         0.805301 |        0.802375 |    0.522658 | 0.761312 |   0.619805 |  0.870240 |            0.002925 |
| Random Forest           |         0.800923 |        0.797104 |    0.513534 | 0.779639 |   0.619207 |  0.869992 |            0.003819 |
| Tuned Random Forest     |         0.799658 |        0.795649 |    0.511222 | 0.779496 |   0.617479 |  0.869517 |            0.004008 |

**Selected final model:** `Decision Tree - Gini` based on the strongest F1-score in this project run.

## 🔍 Key Business Findings

- Overall churn rate is **21.16%**.
- Highest churn geography is **Berlin-Germany**.
- Highest churn age group is **51-60**.
- Top model features include: Age, NumOfProducts, IsActiveMember, Balance, Geography_Berlin-Germany.
- The final model correctly identifies 5,330 churners and misses 1,654 churners in the test set at the 0.50 threshold.

## 📚 Complete Question Bank

`reports/all_216_question_answers.md` contains the original 216 project, interview, business, ethics, deployment, and portfolio questions with evidence-based answers from the current project run.

## 📈 Visual Outputs

- `images/confusion_matrix.png`
- `images/roc_curve.png`
- `images/feature_importance.png`
- `images/decision_tree_visual.png`

## 🚀 Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
streamlit run app/app.py
```

## 📁 Repository Structure

```text
bank-customer-churn-prediction/
├── app/
│   └── app.py
├── data/
│   └── processed/bank_churn_cleaned.csv
├── images/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── decision_tree_visual.png
├── models/
│   ├── churn_model.pkl
│   └── model_metadata.json
├── notebooks/
│   └── bank_customer_churn_prediction.ipynb
├── reports/
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── project_report.md
│   └── all_216_question_answers.md
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── README.md
├── Bank_churn.csv
├── requirements.txt
└── .gitignore
```

## ⚠️ Limitations

This project uses an educational Kaggle-style dataset. It should not be used directly for real banking decisions without fairness testing, compliance review, external validation, monitoring, and human oversight.

## 👤 Author

**Alok Agarwal**  
Data Analytics • Data Science • Machine Learning • SEO/Digital Marketing
