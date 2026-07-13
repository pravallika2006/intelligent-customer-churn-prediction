import pandas as pd
import joblib

from scipy.stats import randint, uniform

from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# ============================================================
# LOAD DATA
# ============================================================

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

print("=" * 70)
print("Datasets Loaded Successfully!")
print("=" * 70)

# ============================================================
# SCALE NUMERICAL FEATURES
# ============================================================

numeric_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

scaler = StandardScaler()

X_train[numeric_cols] = scaler.fit_transform(
    X_train[numeric_cols]
)

X_test[numeric_cols] = scaler.transform(
    X_test[numeric_cols]
)

joblib.dump(
    scaler,
    "models/scaler_tuned.pkl"
)

# ============================================================
# HANDLE CLASS IMBALANCE
# ============================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print(f"\nScale Pos Weight : {scale_pos_weight:.2f}")

# ============================================================
# RANDOM FOREST PARAMETER SPACE
# ============================================================

rf_params = {

    "n_estimators": randint(100, 500),

    "max_depth": randint(5, 30),

    "min_samples_split": randint(2, 15),

    "min_samples_leaf": randint(1, 10),

    "class_weight": ["balanced"]

}

print("\n" + "=" * 70)
print("TUNING RANDOM FOREST")
print("=" * 70)

rf = RandomForestClassifier(
    random_state=42
)

rf_search = RandomizedSearchCV(

    estimator=rf,

    param_distributions=rf_params,

    n_iter=20,

    cv=5,

    scoring="f1",

    verbose=2,

    random_state=42,

    n_jobs=-1

)

rf_search.fit(X_train, y_train)

best_rf = rf_search.best_estimator_

print("\nBest Random Forest Parameters\n")

for key, value in rf_search.best_params_.items():
    print(f"{key} : {value}")

rf_predictions = best_rf.predict(X_test)

print("\nClassification Report\n")

print(classification_report(y_test, rf_predictions))

rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)
rf_roc = roc_auc_score(y_test, rf_predictions)

print(f"Accuracy : {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall   : {rf_recall:.4f}")
print(f"F1 Score : {rf_f1:.4f}")
print(f"ROC AUC  : {rf_roc:.4f}")

joblib.dump(
    best_rf,
    "models/best_model.pkl"
)

print("Best model updated successfully!")

rf_results = pd.DataFrame(rf_search.cv_results_)

rf_results.to_csv(
    "reports/random_forest_tuning.csv",
    index=False
)

print("\nRandom Forest model saved successfully!")

# ============================================================
# XGBOOST PARAMETER SPACE
# ============================================================

xgb_params = {

    "n_estimators": randint(100, 500),

    "learning_rate": uniform(0.01, 0.29),

    "max_depth": randint(3, 10),

    "subsample": uniform(0.7, 0.3),

    "colsample_bytree": uniform(0.7, 0.3),

    "scale_pos_weight": [scale_pos_weight]

}

print("\n" + "=" * 70)
print("TUNING XGBOOST")
print("=" * 70)

xgb = XGBClassifier(

    random_state=42,

    eval_metric="logloss"

)

xgb_search = RandomizedSearchCV(

    estimator=xgb,

    param_distributions=xgb_params,

    n_iter=20,

    cv=5,

    scoring="f1",

    verbose=2,

    random_state=42,

    n_jobs=-1

)

xgb_search.fit(X_train, y_train)

best_xgb = xgb_search.best_estimator_

print("\nBest XGBoost Parameters\n")

for key, value in xgb_search.best_params_.items():
    print(f"{key} : {value}")

xgb_predictions = best_xgb.predict(X_test)

print("\nClassification Report\n")

print(classification_report(y_test, xgb_predictions))

xgb_accuracy = accuracy_score(y_test, xgb_predictions)
xgb_precision = precision_score(y_test, xgb_predictions)
xgb_recall = recall_score(y_test, xgb_predictions)
xgb_f1 = f1_score(y_test, xgb_predictions)
xgb_roc = roc_auc_score(y_test, xgb_predictions)

print(f"Accuracy : {xgb_accuracy:.4f}")
print(f"Precision: {xgb_precision:.4f}")
print(f"Recall   : {xgb_recall:.4f}")
print(f"F1 Score : {xgb_f1:.4f}")
print(f"ROC AUC  : {xgb_roc:.4f}")

joblib.dump(
    best_xgb,
    "models/xgboost_tuned.pkl"
)

xgb_results = pd.DataFrame(xgb_search.cv_results_)

xgb_results.to_csv(
    "reports/xgboost_tuning.csv",
    index=False
)

print("\nXGBoost model saved successfully!")

# ============================================================
# FINAL COMPARISON
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "Random Forest Tuned",
        "XGBoost Tuned"
    ],

    "Accuracy": [
        rf_accuracy,
        xgb_accuracy
    ],

    "Precision": [
        rf_precision,
        xgb_precision
    ],

    "Recall": [
        rf_recall,
        xgb_recall
    ],

    "F1 Score": [
        rf_f1,
        xgb_f1
    ],

    "ROC AUC": [
        rf_roc,
        xgb_roc
    ]

})

comparison = comparison.sort_values(
    by="F1 Score",
    ascending=False
)

comparison.to_csv(
    "reports/tuned_models_comparison.csv",
    index=False
)

print("\n")
print("=" * 70)
print("TUNED MODEL COMPARISON")
print("=" * 70)

print(comparison)

print("\n")
print("=" * 70)
print("HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY!")
print("=" * 70)