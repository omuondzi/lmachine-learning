
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")         

# Scikit-learn utilities
from sklearn.datasets import fetch_openml         
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier  

# Evaluation helpers
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


print("=" * 60)
print("  MNIST Digit Classification  (0 – 9)")
print("=" * 60)

print("\n[1] Loading MNIST dataset from OpenML …")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")

X = mnist.data.astype(np.float32)   
y = mnist.target.astype(int)        

print(f"    Total samples : {X.shape[0]}")
print(f"    Features/sample: {X.shape[1]}  (28 × 28 pixels)")
print(f"    Classes       : {np.unique(y).tolist()}")


print("\n[2] Plotting sample images …")

fig, axes = plt.subplots(2, 10, figsize=(15, 3.5))
fig.suptitle("MNIST – Sample Images for Each Digit (0 – 9)",
             fontsize=13, weight="bold")

for digit in range(10):
    # find two example images for each digit
    indices = np.where(y == digit)[0][:2]
    for row, idx in enumerate(indices):
        ax = axes[row, digit]
        ax.imshow(X[idx].reshape(28, 28), cmap="gray_r", interpolation="nearest")
        ax.set_title(str(digit), fontsize=10)
        ax.axis("off")

plt.tight_layout()
plt.savefig("mnist_samples.png", dpi=120, bbox_inches="tight")
plt.show()
print("    Saved → mnist_samples.png")


print("\n[3] Pre-processing …")


SUBSET_SIZE = 20_000
rng = np.random.default_rng(seed=42)
subset_idx = rng.choice(len(X), size=SUBSET_SIZE, replace=False)
X_sub, y_sub = X[subset_idx], y[subset_idx]

print(f"    Using {SUBSET_SIZE:,} samples for demonstration speed.")

X_train, X_test, y_train, y_test = train_test_split(
    X_sub, y_sub, test_size=0.20, random_state=42, stratify=y_sub
)
print(f"    Train size: {X_train.shape[0]:,}  |  Test size: {X_test.shape[0]:,}")


scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)  
X_test_sc  = scaler.transform(X_test)        


print("\n[4] Defining classifiers …")

classifiers = [
    (
        "Logistic Regression",
        LogisticRegression(max_iter=300, solver="saga", C=0.5, random_state=42),
        True,   
    ),
    (
        "Random Forest",
        RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
        False,  
    ),
    (
        "Support Vector Machine",
        SVC(kernel="rbf", C=5.0, gamma="scale", random_state=42),
        True,
    ),
    (
        "k-Nearest Neighbours (k=5)",
        KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        True,
    ),
    (
        "Neural Network (MLP)",
        MLPClassifier(
            hidden_layer_sizes=(256, 128),  
            activation="relu",
            max_iter=30,
            learning_rate_init=0.001,
            random_state=42,
            verbose=False,
        ),
        True,
    ),
]


print("\n[5] Training & evaluating each classifier …\n")

results = {}  

for name, model, scaled in classifiers:
    print(f"  ► {name}")

    Xtr = X_train_sc if scaled else X_train
    Xte = X_test_sc  if scaled else X_test

    model.fit(Xtr, y_train)

    y_pred = model.predict(Xte)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"    Accuracy : {acc * 100:.2f} %")

    print(classification_report(y_test, y_pred,
                                target_names=[str(d) for d in range(10)],
                                zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=range(10))
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix – {name}\n(Accuracy: {acc*100:.2f} %)",
                 weight="bold")
    plt.tight_layout()
    fname = name.replace(" ", "_").replace("(", "").replace(")", "") + "_cm.png"
    plt.savefig(fname, dpi=110, bbox_inches="tight")
    plt.show()
    print(f"    Saved → {fname}\n")


print("[6] Plotting accuracy comparison …")

names  = list(results.keys())
accs   = [results[n] * 100 for n in names]
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(names, accs, color=colors, edgecolor="white", height=0.55)

for bar, acc in zip(bars, accs):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{acc:.2f} %", va="center", fontsize=10, weight="bold")

ax.set_xlim(0, 105)
ax.set_xlabel("Test Accuracy (%)", fontsize=11)
ax.set_title("MNIST Digit Classification – Algorithm Comparison",
             fontsize=13, weight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=120, bbox_inches="tight")
plt.show()
print("    Saved → accuracy_comparison.png")


print("\n[7] Inspecting misclassified samples (best model) …")

best_name = max(results, key=results.get)
print(f"    Best model: {best_name}  ({results[best_name]*100:.2f} %)")

best_clf   = next(m for n, m, _ in classifiers if n == best_name)
best_scaled = next(s for n, _, s in classifiers if n == best_name)
Xte_best   = X_test_sc if best_scaled else X_test
y_pred_best = best_clf.predict(Xte_best)

wrong_idx = np.where(y_pred_best != y_test)[0][:20]  

if len(wrong_idx) > 0:
    cols = 10
    rows = (len(wrong_idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 2))
    axes = np.array(axes).reshape(rows, cols)  

    for i, idx in enumerate(wrong_idx):
        ax = axes[i // cols, i % cols]
        ax.imshow(X_test[idx].reshape(28, 28), cmap="Reds", interpolation="nearest")
        ax.set_title(f"True:{y_test[idx]}\nPred:{y_pred_best[idx]}",
                     fontsize=7, color="darkred")
        ax.axis("off")

    for j in range(i + 1, rows * cols):
        axes[j // cols, j % cols].axis("off")

    fig.suptitle(f"Misclassified Digits – {best_name}", fontsize=12, weight="bold")
    plt.tight_layout()
    plt.savefig("misclassified.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("    Saved → misclassified.png")
else:
    print("    No misclassifications found on this test subset!")


print("\n" + "=" * 60)
print("  FINAL ACCURACY SUMMARY")
print("=" * 60)
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    bar = "█" * int(acc * 40)
    print(f"  {name:<30}  {acc*100:6.2f} %  {bar}")
print("=" * 60)
print("\nDone! All plots saved to the working directory.")