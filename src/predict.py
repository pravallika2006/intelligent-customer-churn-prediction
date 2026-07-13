import pandas as pd
import joblib

# ==========================================================
# Load Saved Objects
# ==========================================================

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# ==========================================================
# Prediction Function
# ==========================================================

def predict_customer(customer_data):

    # Create dataframe from dictionary
    df = pd.DataFrame([customer_data])

    # One-hot encode
    df = pd.get_dummies(df, drop_first=True)

    # Add missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Remove extra columns
    df = df[feature_columns]

    # Scale numeric columns
    numeric_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Prediction
    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return prediction, probability


# ==========================================================
# Example Customer
# ==========================================================

sample_customer = {

    "gender": "Female",

    "SeniorCitizen": 0,

    "Partner": "Yes",

    "Dependents": "No",

    "tenure": 12,

    "PhoneService": "Yes",

    "MultipleLines": "No",

    "InternetService": "Fiber optic",

    "OnlineSecurity": "No",

    "OnlineBackup": "Yes",

    "DeviceProtection": "Yes",

    "TechSupport": "No",

    "StreamingTV": "Yes",

    "StreamingMovies": "Yes",

    "Contract": "Month-to-month",

    "PaperlessBilling": "Yes",

    "PaymentMethod": "Electronic check",

    "MonthlyCharges": 95.60,

    "TotalCharges": 1147.20

}

prediction, probability = predict_customer(sample_customer)

print("=" * 60)

print("Prediction")

print("=" * 60)

print(
    "Customer will Churn"
    if prediction == 1
    else "Customer will Stay"
)

print(f"Churn Probability : {probability:.2%}")