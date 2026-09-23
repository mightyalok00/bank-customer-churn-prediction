# Bank Customer Churn Prediction - Project Report

## Executive Summary
This project predicts whether a bank customer is likely to churn using Decision Tree and Random Forest classification models. The cleaned modeling dataset has **165,031 rows and 12 columns** after identity removal, combined-field parsing, validation, and duplicate removal. The churn rate is **21.16%**, so the problem is a binary classification task focused on identifying customers who may leave.

The best model by holdout F1-score is **Decision Tree - Gini**. The project includes Gini Impurity, Entropy comparison, Random Forest feature importance, confusion matrix, ROC and precision-recall curves, probability calibration, five-fold cross-validation, threshold analysis, a saved model pipeline, and a tested Streamlit app.

## Key Dataset Findings
- Raw dataset shape: **165,034 rows × 14 columns**.
- Cleaned dataset shape: **165,031 rows × 12 columns**.
- Stayed customers: **130,110**.
- Churned customers: **34,921**.
- Overall churn rate: **21.16%**.
- Highest churn geography: **Berlin-Germany (37.91%)**.
- Highest churn age group: **51-60 (60.84%)**.

## Model Comparison
| Model | Test accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier score |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Decision Tree - Gini | 0.8036 | 0.5247 | 0.7632 | 0.6218 | 0.8695 | 0.6747 | 0.1434 |
| Decision Tree - Entropy | 0.8024 | 0.5227 | 0.7613 | 0.6198 | 0.8702 | 0.6819 | 0.1427 |
| Random Forest | 0.7971 | 0.5135 | 0.7796 | 0.6192 | 0.8700 | 0.6712 | 0.1468 |
| Tuned Random Forest | 0.7956 | 0.5112 | 0.7795 | 0.6175 | 0.8695 | 0.6701 | 0.1470 |

## Five-Fold Cross-Validation

The selected Gini tree achieved mean accuracy **0.8029 ± 0.0063**, F1 **0.6214 ± 0.0054**, ROC-AUC **0.8719 ± 0.0030**, and PR-AUC **0.6815 ± 0.0041**. These small fold-to-fold deviations support the stability of the holdout result.

## Decision Threshold

At the default 0.50 threshold, recall is **76.32%**, precision is **52.47%**, and F1 is **62.18%**. `reports/threshold_analysis.csv` shows alternatives from 0.30 to 0.70 so a bank can choose a threshold using retention capacity and false-positive cost rather than accuracy alone.

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
