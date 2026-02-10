"""
Handwritten Character Recognition using Neural Network on MNIST Dataset
WINDOWS COMPATIBLE VERSION
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

np.random.seed(42)

# Create output directory
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created '{output_dir}' directory\n")

print("=" * 70)
print("HANDWRITTEN CHARACTER RECOGNITION USING NEURAL NETWORK")
print("=" * 70)

print("\n[1] Loading MNIST Dataset...")
print("Downloading 70,000 images...\n")

mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data, mnist.target
X = np.array(X)
y = np.array(y).astype(int)

print(f"Total samples: {X.shape[0]:,}")
print(f"Features: {X.shape[1]} (28x28 pixels)")
print(f"Classes: {np.unique(y)}")

print("\n[2] Visualizing Sample Images...")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Sample Handwritten Digits from MNIST', fontsize=14, fontweight='bold')
for i, ax in enumerate(axes.flat):
    digit_image = X[i].reshape(28, 28)
    ax.imshow(digit_image, cmap='gray')
    ax.set_title(f'Label: {y[i]}', fontsize=11, fontweight='bold')
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sample_digits.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/sample_digits.png")
plt.close()

print("\n[3] Preprocessing Data...")
sample_size = 10000
indices = np.random.choice(len(X), sample_size, replace=False)
X_subset = X[indices]
y_subset = y[indices]

X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=0.2, random_state=42, stratify=y_subset
)

X_train = X_train / 255.0
X_test = X_test / 255.0

print(f"Training samples: {X_train.shape[0]:,}")
print(f"Test samples: {X_test.shape[0]:,}")

print("\n[4] Building Neural Network...")
print("Architecture:")
print("  Input: 784 neurons")
print("  Hidden 1: 128 neurons (ReLU)")
print("  Hidden 2: 64 neurons (ReLU)")
print("  Output: 10 neurons (softmax)")

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    batch_size=32,
    learning_rate_init=0.001,
    max_iter=20,
    random_state=42,
    verbose=True,
    early_stopping=True,
    validation_fraction=0.1
)

print("\n[5] Training the Model...\n")
model.fit(X_train, y_train)
print("\n✓ Training complete!")

print("\n[6] Evaluating Performance...")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y_test, y_test_pred, digits=4))

print("\n[7] Generating Visualizations...")
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(model.loss_curve_, linewidth=2, color='#2E86AB')
ax.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/training_loss.png")
plt.close()

print("\n[8] Creating Confusion Matrix...")
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/confusion_matrix.png")
plt.close()

print("\n[9] Sample Predictions...")
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
fig.suptitle('Sample Predictions with Confidence', fontsize=16, fontweight='bold')
sample_indices = np.random.choice(len(X_test), 15, replace=False)

for idx, ax in enumerate(axes.flat):
    i = sample_indices[idx]
    digit_image = X_test[i].reshape(28, 28)
    ax.imshow(digit_image, cmap='gray')
    true_label = y_test[i]
    pred_label = y_test_pred[i]
    probabilities = model.predict_proba(X_test[i].reshape(1, -1))[0]
    confidence = probabilities[pred_label] * 100
    color = 'green' if true_label == pred_label else 'red'
    ax.set_title(f'True: {true_label} | Pred: {pred_label}\nConf: {confidence:.1f}%', fontsize=10, color=color, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sample_predictions.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/sample_predictions.png")
plt.close()

print("\n[10] Per-Digit Accuracy...")
digit_accuracies = []
for digit in range(10):
    digit_mask = y_test == digit
    digit_accuracy = accuracy_score(y_test[digit_mask], y_test_pred[digit_mask])
    digit_accuracies.append(digit_accuracy * 100)

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
bars = ax.bar(range(10), digit_accuracies, color='#2E86AB', alpha=0.8, edgecolor='black')
ax.set_title('Accuracy per Digit', fontsize=14, fontweight='bold')
ax.set_xlabel('Digit', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_xticks(range(10))
ax.set_ylim([0, 105])
ax.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'per_digit_accuracy.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/per_digit_accuracy.png")
plt.close()

print("\nPer-Digit Accuracy:")
print("-" * 40)
for digit in range(10):
    print(f"Digit {digit}: {digit_accuracies[digit]:.2f}%")

print("\n[11] Saving Model...")
model_path = os.path.join(output_dir, 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"✓ Saved: {model_path}")

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"Training Samples: {len(X_train):,}")
print(f"Test Samples: {len(X_test):,}")
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Iterations: {model.n_iter_}")
print("=" * 70)

print("\n✓ ALL DONE!")
print(f"\nFiles saved in '{output_dir}/' folder:")
print("  1. sample_digits.png")
print("  2. training_loss.png")
print("  3. confusion_matrix.png")
print("  4. sample_predictions.png")
print("  5. per_digit_accuracy.png")
print("  6. model.pkl")
print("\nPress Enter to exit...")
input()