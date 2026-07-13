# Project Requirements

## Project Title

**Intelligent Customer Churn Prediction System**

---

## Problem Statement

Telecommunication companies lose customers due to churn, which directly impacts revenue and customer retention. The objective of this project is to build a Machine Learning system capable of predicting whether a customer is likely to churn based on historical customer data.

---

## Business Objective

The primary business goals are:

- Predict customer churn before it occurs.
- Identify customers at high risk of leaving.
- Understand the major factors influencing churn.
- Support business teams in planning effective retention strategies.

---

## Functional Requirements

The system should:

- Accept customer information through a user-friendly interface.
- Preprocess the input data using the same pipeline used during training.
- Predict whether the customer will churn.
- Display the churn probability.
- Classify the customer into Low, Medium, or High risk.
- Provide business insights based on customer attributes.
- Recommend retention strategies.
- Explain the prediction using SHAP.

---

## Non-Functional Requirements

- Easy-to-use web interface.
- Fast prediction response.
- Reliable and consistent predictions.
- Maintainable and modular code structure.
- Explainable Machine Learning model.

---

## Software Requirements

- Python 3.x
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Plotly
- Matplotlib
- SHAP
- Joblib

---

## Hardware Requirements

Minimum:

- Dual-Core Processor
- 4 GB RAM
- 500 MB Free Storage

Recommended:

- Quad-Core Processor
- 8 GB RAM or higher

---

## Expected Outputs

The application should display:

- Customer churn prediction
- Churn probability
- Risk level
- Business insights
- Recommended retention actions
- SHAP-based feature explanation

---

## Success Criteria

The project is considered successful if it:

- Predicts churn accurately using the trained Random Forest model.
- Provides meaningful explanations for predictions.
- Assists businesses in identifying customers likely to churn.
- Offers actionable recommendations to improve customer retention.