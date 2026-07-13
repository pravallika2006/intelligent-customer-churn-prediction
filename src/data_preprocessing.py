import pandas as pd

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset loaded successfully!")
print(df.shape)

# -----------------------------
# Convert TotalCharges to numeric
# -----------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values with 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# -----------------------------
# Remove Customer ID
# -----------------------------
df.drop(columns=["customerID"], inplace=True)

print("\nColumns after preprocessing:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Save Clean Dataset
# -----------------------------
output_path = "data/processed/telco_churn_clean.csv"

df.to_csv(output_path, index=False)

print(f"\nClean dataset saved to:\n{output_path}")
print("Preprocessing completed successfully!")