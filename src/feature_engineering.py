import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("data/processed/telco_churn_clean.csv")

print("Dataset Loaded!")
print(df.shape)

# ==========================================================
# Remove Customer ID
# ==========================================================

df.drop("customerID", axis=1, inplace=True)

# ==========================================================
# Encode Target Variable
# ==========================================================

df["Churn"] = LabelEncoder().fit_transform(df["Churn"])

print(df["Churn"].value_counts())

# ==========================================================
# One-Hot Encode Categorical Features
# ==========================================================

df = pd.get_dummies(df, drop_first=True)

# ==========================================================
# Split Features & Target
# ==========================================================

X = df.drop("Churn", axis=1)
y = df["Churn"]

# ==========================================================
# Save Feature Columns
# ==========================================================

joblib.dump(
    X.columns.tolist(),
    "models/feature_columns.pkl"
)

print("Feature columns saved successfully!")

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# ==========================================================
# Save Processed Data
# ==========================================================

X_train.to_csv(
    "data/processed/X_train.csv",
    index=False
)

X_test.to_csv(
    "data/processed/X_test.csv",
    index=False
)

y_train.to_csv(
    "data/processed/y_train.csv",
    index=False
)

y_test.to_csv(
    "data/processed/y_test.csv",
    index=False
)

print("\nFeature engineering completed successfully!")