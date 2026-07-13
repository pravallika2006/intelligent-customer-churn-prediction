# 📊 Intelligent Customer Churn Prediction System

A Machine Learning web application that predicts whether a telecom customer is likely to churn and provides explainable insights to support customer retention decisions.

Developed as part of an AIML Internship using Python, Scikit-Learn, SHAP, and Streamlit.

---

## Features

- Predict customer churn
- Display churn probability
- Classify customer risk (Low / Medium / High)
- Business insights based on customer profile
- Recommended retention actions
- SHAP explainability for model predictions
- Interactive Streamlit dashboard

---

## Dataset

**IBM Telco Customer Churn Dataset**

The dataset contains customer demographics, subscribed services, billing information, tenure, and churn status.

---

## Machine Learning Models Evaluated

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Hyperparameter tuning was performed on:

- Random Forest
- XGBoost

---

## Final Model Selection

Although **Logistic Regression** achieved the highest overall accuracy (**80.62%**), the **Tuned Random Forest** model was selected for deployment because it achieved a significantly higher **Recall (79.68%)** for churn prediction.

In customer churn prediction, identifying customers who are likely to leave is more important than maximizing overall accuracy. A higher recall helps businesses take proactive retention actions before customers churn.

**Final deployed model:** ✅ Tuned Random Forest

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Plotly
- Matplotlib
- SHAP
- Joblib

---

## Project Structure

```
intelligent-customer-churn-prediction/
│
├── app.py
├── data/
├── docs/
├── images/
├── models/
├── notebooks/
├── reports/
├── src/
├── tests/
├── README.md
└── requirements.txt
```

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/pravallika2006/intelligent-customer-churn-prediction.git
```

Move into the project folder:

```bash
cd intelligent-customer-churn-prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Future Improvements

- Deploy the application to Streamlit Cloud
- Improve prediction accuracy with advanced feature engineering
- Add customer segmentation
- Integrate with real-time business databases

---

## Author

**Devagurthi Teja Pravallika**

Developed during AIML Internship (2026).
