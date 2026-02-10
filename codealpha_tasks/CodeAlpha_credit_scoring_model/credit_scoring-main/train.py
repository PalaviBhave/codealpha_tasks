# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("credit_data_engineered.csv")

X = df.drop("creditworthy", axis=1)
y = df["creditworthy"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(random_state=42),
}

trained = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
    trained[name] = model
    print(f"{name} — trained ✓")

# Save test data for evaluation
import joblib, os
os.makedirs("saved", exist_ok=True)
joblib.dump(trained,        "saved/models.pkl")
joblib.dump(scaler,         "saved/scaler.pkl")
joblib.dump((X_test, X_test_scaled, y_test), "saved/test_data.pkl")
print("All models saved to saved/")