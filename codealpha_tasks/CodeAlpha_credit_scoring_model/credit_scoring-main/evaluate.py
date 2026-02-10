# evaluate.py
import joblib
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

models       = joblib.load("saved/models.pkl")
scaler       = joblib.load("saved/scaler.pkl")
X_test, X_test_scaled, y_test = joblib.load("saved/test_data.pkl")

print("=" * 55)
print(f"{'Model':<25} {'Precision':>9} {'Recall':>7} {'F1':>6} {'ROC-AUC':>8}")
print("=" * 55)

for name, model in models.items():
    X = X_test_scaled if name == "Logistic Regression" else X_test
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    p   = precision_score(y_test, y_pred)
    r   = recall_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"{name:<25} {p:>9.3f} {r:>7.3f} {f1:>6.3f} {auc:>8.3f}")

print("=" * 55)

# Detailed report for best model (Random Forest)
print("\n--- Detailed Report: Random Forest ---")
X = X_test
y_pred = models["Random Forest"].predict(X)
print(classification_report(y_test, y_pred))

# Confusion Matrix
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
ax.set_title("Random Forest — Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("Confusion matrix saved: confusion_matrix.png")