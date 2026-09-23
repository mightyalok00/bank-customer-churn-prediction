# Bank Customer Churn Prediction - Project Report

## Executive Summary
This project predicts whether a bank customer is likely to churn using Decision Tree and Random Forest classification models. The cleaned modeling dataset has **165,031 rows and 12 columns** after identity removal, combined-field parsing, validation, and duplicate removal. The churn rate is **21.16%**, so the problem is a binary classification task focused on identifying customers who may leave.

The best model by F1-score is **Decision Tree - Gini**. The project includes Gini Impurity, Entropy comparison, Random Forest feature importance, confusion matrix, ROC curve, saved model pipeline, and a Streamlit app.

## Key Dataset Findings
- Raw dataset shape: **165,034 rows × 14 columns**.
- Cleaned dataset shape: **165,031 rows × 12 columns**.
- Stayed customers: **130,110**.
- Churned customers: **34,921**.
- Overall churn rate: **21.16%**.
- Highest churn geography: **Berlin-Germany (37.91%)**.
- Highest churn age group: **51-60 (60.84%)**.

## Model Comparison
| model                   |   train_accuracy |   test_accuracy |   precision |   recall |   f1_score |   roc_auc |   overfit_gap_accuracy |
|:------------------------|-----------------:|----------------:|------------:|---------:|-----------:|----------:|-----------------------:|
| Decision Tree - Gini    |         0.806262 |        0.803587 |    0.524658 | 0.763173 |   0.621828 |  0.869526 |            0.002675 |
| Decision Tree - Entropy |         0.805301 |        0.802375 |    0.522658 | 0.761312 |   0.619805 |  0.870240 |            0.002925 |
| Random Forest           |         0.800923 |        0.797104 |    0.513534 | 0.779639 |   0.619207 |  0.869992 |            0.003819 |
| Tuned Random Forest     |         0.799658 |        0.795649 |    0.511222 | 0.779496 |   0.617479 |  0.869517 |            0.004008 |

## Top Feature Importances
| feature                  |   importance |
|:-------------------------|-------------:|
| Age                      |  0.433841    |
| NumOfProducts            |  0.363451    |
| IsActiveMember           |  0.0959159   |
| Balance                  |  0.0731180   |
| Geography_Berlin-Germany |  0.0234237   |
| Gender_Female            |  0.00517621  |
| Gender_Male              |  0.00505320  |
| EstimatedSalary          |  1.94478e-05 |
| HasCrCard                |  2.28473e-06 |
| Tenure                   |  0           |

## Business Recommendations
1. Prioritize customers with high predicted churn probability for proactive relationship-manager contact.
2. Use engagement campaigns for inactive customers because inactivity is directly actionable.
3. Create focused offers for valuable high-risk customers instead of spending retention budget on every customer.
4. Track retention KPIs after intervention: churn rate, saved customers, offer acceptance, campaign ROI, and false-positive cost.

## Limitations
This is a Kaggle/educational dataset. It does not include full real banking behavior such as transaction frequency, complaint history, branch interactions, digital banking usage, recent product changes, or retention-campaign history. The model should support human decisions, not fully automated customer treatment.
