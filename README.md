# Intelligent Customer Churn Prediction & Decision Support System

This project is being developed as part of an AIML Internship.

## Final Model

The project evaluates multiple machine learning algorithms for customer churn prediction.

Models trained:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Hyperparameter tuning was performed on:

- Random Forest
- XGBoost

Although Logistic Regression achieved the highest accuracy (80.62%), the Tuned Random Forest was selected for deployment because it achieved a much higher Recall (79.68%) on the churn class.

In churn prediction, recall is more important than overall accuracy because identifying customers who are likely to leave allows businesses to take proactive retention actions.

Final deployed model:

- ✅ Tuned Random Forest