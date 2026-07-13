import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

# =====================================================
# Load Test Data
# =====================================================

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

print("Test dataset loaded successfully!")

# =====================================================
# Load Scaler
# =====================================================

scaler = joblib.load("models/scaler.pkl")

numeric_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

X_test[numeric_cols] = scaler.transform(
    X_test[numeric_cols]
)

# =====================================================
# Load Best Model
# =====================================================

model = joblib.load("models/best_model.pkl")

print("Best model loaded successfully!")

# =====================================================
# Predictions
# =====================================================

y_pred = model.predict(X_test)

if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test)[:, 1]
else:
    y_prob = y_pred

# =====================================================
# Metrics
# =====================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("FINAL MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc:.4f}")

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# =====================================================
# Confusion Matrix
# =====================================================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.savefig("reports/confusion_matrix.png", dpi=300)
plt.show()

# =====================================================
# ROC Curve
# =====================================================

RocCurveDisplay.from_predictions(
    y_test,
    y_prob
)

plt.title("ROC Curve")
plt.savefig("reports/roc_curve.png", dpi=300)
plt.show()

print("\nEvaluation completed successfully!")