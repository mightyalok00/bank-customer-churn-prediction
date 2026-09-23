# Portfolio 9/10 Upgrade Report

This report strengthens the sections that matter most for teacher review, interview explanation, and GitHub portfolio scoring. It expands the model-tuning, metric-selection, business-recommendation, error-cost, limitations, and ethics areas of the Bank Customer Churn Prediction project.

## 1. Random Forest Tuning Expansion

Random Forest tuning matters because a baseline forest can still underfit or overfit depending on tree depth, leaf size, number of trees, feature sampling, and class imbalance handling.

### Tuned hyperparameters

| Hyperparameter | Business/ML meaning | Practical impact |
|---|---|---|
| `n_estimators` | Number of trees | More trees usually stabilize predictions but increase compute time. |
| `max_depth` | Maximum tree depth | Controls complexity; lower depth reduces overfitting but can miss patterns. |
| `min_samples_leaf` | Minimum samples in a final leaf | Larger leaves smooth predictions and reduce noisy customer segments. |
| `max_features` | Number of features considered per split | Adds diversity between trees and can reduce correlation across trees. |
| `class_weight` | Weighting for churn vs stay classes | Helps the model pay attention to the minority churn class. |

### Result interpretation

The tuned Random Forest did not outperform the selected Gini Decision Tree on F1-score in the current run. This does not mean Random Forest is a bad algorithm. It means that for this dataset, feature set, and current tuning grid, the simpler constrained Gini tree achieved the strongest balance between precision and recall.

The correct portfolio conclusion is:

> Random Forest was tested as a stronger ensemble method, but model selection was based on measured validation results rather than algorithm popularity. The final Gini Decision Tree was selected because it produced the best F1-score while remaining interpretable.

This is a strong answer in interviews because it shows honest model selection instead of blindly choosing a more complex model.

## 2. Most Important Metric: Why F1-Score Was Used

Accuracy alone is not enough in churn prediction because the majority of customers stayed. A model can look accurate by predicting most customers as stayers, but that does not help the bank find churners.

### Metric meaning for churn

| Metric | What it means | Business interpretation |
|---|---|---|
| Accuracy | Overall correct predictions | Can be misleading when most customers do not churn. |
| Precision | Of customers flagged as churn, how many really churned | Important when retention offers are expensive. |
| Recall | Of actual churners, how many the model found | Important when missing churners is costly. |
| F1-score | Balance between precision and recall | Useful when both missed churners and wasted offers matter. |
| ROC-AUC | Ranking quality across thresholds | Useful for risk scoring and prioritization. |
| PR-AUC | Precision-recall performance under imbalance | Useful for minority-class performance review. |

### Final metric decision

F1-score is a reasonable primary selection metric because the bank wants to catch churners without creating too many unnecessary retention contacts. If the bank later calculates exact revenue loss and campaign costs, the best metric should become expected business value rather than only F1-score.

## 3. Business Recommendations Expansion

The model should support a retention workflow, not replace business judgment.

### Recommended action 1: High-value/high-risk review

Customers with high churn probability and high potential value should be routed to relationship managers. These customers may justify personalized attention because losing them can affect deposits, future interest income, fees, and cross-sell opportunities.

Expected benefit: protect customer lifetime value.

Risk: costly false positives if the model flags customers who would not have churned.

### Recommended action 2: Inactive-member re-engagement

Inactive members show higher churn risk, so the bank can use app notifications, email journeys, service check-ins, and feature education to rebuild engagement.

Expected benefit: low-cost outreach can reactivate customers before expensive intervention is needed.

Risk: too many contacts may create fatigue or annoyance.

### Recommended action 3: Product-count strategy

The analysis shows churn differs strongly by product count. One-product customers may need relevant cross-sell or bundle offers, while three/four-product customers may require service-friction investigation before selling more products.

Expected benefit: retention action becomes more specific instead of generic.

Risk: pushing more products without understanding customer dissatisfaction can worsen churn.

## 4. False Positive and False Negative Business Cost

### False negative

A false negative happens when the model predicts that a customer will stay, but the customer actually churns.

Business impact:

- missed chance for retention
- lost future deposits, fees, and cross-sell opportunities
- possible service issue remains undiscovered
- higher future acquisition cost to replace the customer

### False positive

A false positive happens when the model predicts churn, but the customer would have stayed anyway.

Business impact:

- wasted retention budget
- unnecessary discount or offer
- possible customer annoyance from irrelevant contact
- relationship manager time spent on lower-priority cases

### Which is worse?

There is no universal answer. If customers have high lifetime value, false negatives may be more costly. If retention offers are expensive, false positives may become more costly. A mature bank should estimate expected value:

```text
Expected intervention value = churn probability × customer value × expected save rate - retention cost
```

The final decision threshold should be selected using budget, capacity, expected value, and fairness constraints.

## 5. Limitations Expansion

This project is strong for portfolio learning, but it is not production-ready banking software.

### Dataset limitations

- It is an educational Kaggle-style dataset.
- It lacks transaction history.
- It lacks customer complaint and service-interaction data.
- It lacks campaign exposure and treatment-response history.
- It does not include time-based customer behavior.
- It does not include profitability or customer lifetime value.

### Modeling limitations

- The model finds associations, not proven causes.
- Feature importance does not prove business causation.
- The model must be tested on future unseen time periods before real use.
- Thresholds should be chosen based on cost-benefit analysis, not only default 0.50.
- Real deployment needs drift monitoring and periodic validation.

## 6. Ethics and Fairness Expansion

Banking models can affect customer treatment, offers, service priority, and access to benefits. Because this project uses features such as age, gender, and geography, fairness review is important before production use.

### Main fairness risks

| Risk | Explanation |
|---|---|
| Disparate treatment | Some groups may receive different levels of attention or offers. |
| Proxy discrimination | Geography or behavior variables may indirectly represent sensitive groups. |
| Unequal model error | False positives or false negatives may be higher for some demographic groups. |
| Automation risk | Fully automated decisions can create unfair or unexplained customer outcomes. |

### Responsible-use recommendation

The model should be used as a decision-support tool. It should not automatically approve, deny, penalize, or downgrade customers. Human review, compliance checks, explainability, and fairness monitoring are required before real banking deployment.

## 7. Strong Interview Answer

> I built a churn-classification project using a bank customer dataset. I cleaned the data, removed identity fields, extracted geography and gender, handled missing values through a pipeline, and compared Gini Decision Tree, Entropy Decision Tree, Random Forest, and Tuned Random Forest. I selected the Gini Decision Tree because it achieved the strongest F1-score in this project run while remaining interpretable. The model found that age, number of products, activity status, balance, and Berlin geography were important churn signals. I also analyzed false positives, false negatives, threshold trade-offs, business recommendations, and fairness limitations. The project includes a Streamlit app so a relationship manager can score customers, review churn probability, and decide which customers need retention follow-up.

## 8. Final Portfolio Assessment

After these README and report upgrades, the project is stronger because it now shows:

- technical ML workflow
- Gini and Entropy understanding
- classification metric interpretation
- Random Forest comparison
- threshold/business trade-off thinking
- retention strategy
- fairness and ethics awareness
- deployment through Streamlit
- automated checks and reproducibility

This moves the project closer to a **9/10 junior data science portfolio project** because it demonstrates not only coding, but also business reasoning and responsible model interpretation.
