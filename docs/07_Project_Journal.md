# Project Journal

## Day 1 – Phase 1: Business Understanding

### What I Learned
- Customer churn is...
- Businesses care about churn because...
- Machine Learning is used because...
- False Positives and False Negatives have different business impacts.

### Challenges
- Initially, I thought ML projects start with coding, but I learned that understanding the business problem comes first.

### Key Takeaways
- Every ML project begins with understanding the business objective.
- Accuracy alone does not define success.
- Machine Learning exists to solve business problems.

(i think i missed day-2 journal)

## Day 3 – Feature Analysis

### What I Learned

- EDA is not just about plotting graphs; it is about extracting meaningful business insights.
- Contract duration has a strong influence on customer churn.
- Fiber Optic customers unexpectedly have the highest churn rate.
- Data analysis should challenge assumptions rather than confirm them.
- Every business recommendation should balance customer satisfaction with company profitability.

### Business Thinking

While analyzing Fiber Optic customers, I realized that simply reducing prices may not always be the best solution. Improving customer support could reduce churn while maintaining revenue, making it a more sustainable strategy.


# Model Training and Hyperparameter Tuning

## Date

(Enter Submission Date)

---

## Tasks Completed

- Completed data preprocessing.
- Performed feature engineering.
- Split the dataset into training and testing sets.
- Trained multiple machine learning models.
- Compared model performance using various evaluation metrics.
- Performed hyperparameter tuning using RandomizedSearchCV.
- Saved trained models and preprocessing scaler.
- Generated model comparison reports.

---

## Models Implemented

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

---

## Hyperparameter Tuning

Random Forest and XGBoost were tuned using RandomizedSearchCV with 5-fold Cross Validation.

Important techniques used:

- StandardScaler
- Class Weight Balancing
- Scale Pos Weight
- Cross Validation

---

## Performance Summary

### Baseline Best Model

Logistic Regression

Accuracy: **80.62%**

---

### Tuned Best Model

Random Forest

Accuracy: **75.73%**

Recall: **79.68%**

F1 Score: **63.54%**

ROC-AUC: **76.99%**

---

## Key Learnings

- Accuracy alone is not sufficient for imbalanced classification problems.
- Recall and F1-score are important metrics for churn prediction.
- Hyperparameter tuning can significantly improve recall even if overall accuracy decreases.
- Business objectives should guide model selection rather than accuracy alone.
