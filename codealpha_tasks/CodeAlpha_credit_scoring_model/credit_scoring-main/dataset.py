# dataset.py
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

data = {
    "income":          np.random.randint(20000, 120000, n),
    "debts":           np.random.randint(0, 50000, n),
    "payment_history": np.random.randint(0, 100, n),   # 0 = worst, 100 = perfect
    "age":             np.random.randint(18, 70, n),
    "num_late_payments": np.random.randint(0, 20, n),
}

df = pd.DataFrame(data)

# Target: 1 = creditworthy, 0 = not creditworthy
# Simple rule: high income + good payment history + low debts = creditworthy
score = (
    df["income"] / 120000 * 0.4 +
    df["payment_history"] / 100 * 0.4 +
    (1 - df["debts"] / 50000) * 0.2
)
df["creditworthy"] = (score > 0.5).astype(int)

df.to_csv("credit_data.csv", index=False)
print("Dataset saved: credit_data.csv")
print(df.head())