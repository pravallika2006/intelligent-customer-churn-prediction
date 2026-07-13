import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# ==========================================================
# Load Processed Dataset
# ==========================================================

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

print("Datasets Loaded Successfully!")
print("-" * 60)

# ==========================================================
# Scale Numerical Features
# ==========================================================

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

scaler = StandardScaler()

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Save scaler for deployment
joblib.dump(scaler, "models/scaler.pkl")

# ==========================================================
# Models
# ==========================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )

}

# ==========================================================
# Train Models
# ==========================================================

results = []

best_model = None
best_model_name = ""
best_accuracy = 0

for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Training : {name}")
    print("=" * 60)

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc = roc_auc_score(y_test, predictions)

    print(classification_report(y_test, predictions))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc
    })

    # Save Best Model
    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# ==========================================================
# Model Comparison
# ==========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n")
print("=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(results_df.round(4))

# ==========================================================
# Save Results
# ==========================================================

results_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("\n" + "=" * 80)
print(f"Best Model : {best_model_name}")
print(f"Best Accuracy : {best_accuracy:.4f}")

print("\nBest model saved to : models/best_model.pkl")
print("Scaler saved to     : models/scaler.pkl")
print("Comparison saved to : reports/model_comparison.csv")

print("\nTraining Completed Successfully!")