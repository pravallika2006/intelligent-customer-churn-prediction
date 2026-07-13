# Model Comparison

## Objective

The objective of this phase was to train multiple machine learning models, compare their performance, and select the best model for customer churn prediction.

---

# Models Trained

The following machine learning algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. XGBoost Classifier

Later, Random Forest and XGBoost were optimized using RandomizedSearchCV for hyperparameter tuning.

---

# Baseline Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|--------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.8062 | 0.6593 | 0.5588 | 0.6049 | 0.7272 |
| Decision Tree | 0.7253 | 0.4825 | 0.4786 | 0.4805 | 0.6466 |
| Random Forest | 0.7899 | 0.6291 | 0.5080 | 0.5621 | 0.6999 |
| XGBoost | 0.7764 | 0.5891 | 0.5214 | 0.5532 | 0.6950 |

---

# Hyperparameter Tuning

Hyperparameter tuning was performed using **RandomizedSearchCV** with **5-fold Cross Validation**.

The following models were tuned:

- Random Forest
- XGBoost

Techniques used:

- RandomizedSearchCV
- Class balancing
- Standard Scaling
- Cross Validation

---

# Tuned Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|--------|----------|-----------|--------|----------|---------|
| Random Forest (Tuned) | 0.7573 | 0.5284 | 0.7968 | 0.6354 | 0.7699 |
| XGBoost (Tuned) | 0.7509 | 0.5204 | 0.7861 | 0.6262 | 0.7621 |

---

# Final Model Selection

Although Logistic Regression achieved the highest accuracy (80.62%), customer churn prediction is an imbalanced classification problem where identifying churn customers is more important than maximizing overall accuracy.

The tuned Random Forest model achieved:

- Highest Recall (79.68%)
- Highest F1 Score (63.54%)
- Highest ROC-AUC (76.99%)

Since the business objective is to identify customers who are likely to churn, the Tuned Random Forest model was selected as the final model.

---


## Selected Model

Although Logistic Regression achieved the highest overall accuracy (80.62%), the Tuned Random Forest model was selected for deployment because customer churn prediction is a recall-focused business problem.

The tuned Random Forest correctly identified nearly 80% of customers who were likely to churn, making it significantly more effective for customer retention strategies.

Therefore, the Tuned Random Forest was chosen as the final production model.
