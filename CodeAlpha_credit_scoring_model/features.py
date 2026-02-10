# features.py
import pandas as pd

df = pd.read_csv("credit_data.csv")

# New engineered features
df["debt_to_income_ratio"] = df["debts"] / (df["income"] + 1)
df["payment_score_weighted"] = df["payment_history"] * (1 - df["debt_to_income_ratio"])

print("Features after engineering:")
print(df.head())

df.to_csv("credit_data_engineered.csv", index=False)
print("Saved: credit_data_engineered.csv")