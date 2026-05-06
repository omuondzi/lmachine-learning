"""
MNIST Confusion Matrix – Isolated
===================================
Trains a single classifier on MNIST and displays its confusion matrix.
"""

# ─── IMPORTS ──────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# ─── 1. LOAD MNIST ────────────────────────────────────────────────────────────
print("Loading MNIST …")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")

X = mnist.data.astype(np.float32)   # shape (70000, 784)
y = mnist.target.astype(int)        # labels 0 – 9

# ─── 2. USE A SMALL SUBSET FOR SPEED ─────────────────────────────────────────
# Increase SUBSET_SIZE for higher accuracy (max 70000)
SUBSET_SIZE = 10_000
rng = np.random.default_rng(seed=42)
idx = rng.choice(len(X), size=SUBSET_SIZE, replace=False)
X, y = X[idx], y[idx]

# ─── 3. TRAIN / TEST SPLIT ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ─── 4. SCALE FEATURES ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ─── 5. TRAIN CLASSIFIER ──────────────────────────────────────────────────────
print("Training SVM …")
model = SVC(kernel="rbf", C=5.0, gamma="scale", random_state=42)
model.fit(X_train, y_train)

# ─── 6. PREDICT ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc * 100:.2f} %")

# ─── 7. CONFUSION MATRIX ──────────────────────────────────────────────────────
# confusion_matrix[i][j] = number of samples of true class i predicted as class j
# Diagonal = correct predictions; off-diagonal = misclassifications

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix (raw counts):")
print(cm)

# ─── 8. PLOT CONFUSION MATRIX ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=range(10)        # digits 0 – 9
)

disp.plot(
    ax=ax,
    cmap="Blues",                   # colour intensity = count
    colorbar=True,
    xticks_rotation="horizontal",
)

ax.set_title(
    f"Confusion Matrix – SVM (RBF Kernel)\nAccuracy: {acc * 100:.2f} %",
    fontsize=13,
    weight="bold",
    pad=15,
)
ax.set_xlabel("Predicted Digit", fontsize=11)
ax.set_ylabel("True Digit",      fontsize=11)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=130, bbox_inches="tight")
plt.show()
print("Saved → confusion_matrix.png")