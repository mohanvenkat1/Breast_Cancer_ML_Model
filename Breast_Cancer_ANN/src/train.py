"""
==============================================================================
  Breast Cancer ANN – Training Script
  File   : src/train.py
  Author : College Project
  Purpose: End-to-end training pipeline — loads data, preprocesses, builds,
           trains, evaluates, and saves the ANN model and scaler.
==============================================================================
"""

# ─────────────────────────── Standard & 3rd-party Imports ───────────────────
import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets         import load_breast_cancer
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)
import joblib

import tensorflow as tf
from tensorflow.keras.models   import Sequential
from tensorflow.keras.layers   import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ─────────────────────────── Reproducibility seed ───────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─────────────────────────── Paths ──────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

MODEL_PATH  = os.path.join(MODEL_DIR, "breast_cancer_ann.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
CSV_PATH    = os.path.join(DATASET_DIR, "breast_cancer_wdbc.csv")

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — DATASET LOADING AND EXPLORATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STAGE 1 — DATASET LOADING AND EXPLORATION")
print("="*65)

# Load via scikit-learn (Wisconsin Diagnostic Breast Cancer dataset)
raw = load_breast_cancer()

# Build a readable DataFrame (target: 0=malignant, 1=benign in sklearn)
# We REMAP: 1=Malignant, 0=Benign  so the positive class = cancer
df = pd.DataFrame(raw.data, columns=raw.feature_names)
# sklearn encodes 0=malignant, 1=benign — remap for clinical intuition
df["target"] = np.where(raw.target == 0, 1, 0)   # 1=Malignant, 0=Benign
df["diagnosis"] = df["target"].map({1: "Malignant", 0: "Benign"})

# Save a CSV copy to dataset/
df.to_csv(CSV_PATH, index=False)
print(f"\n✔  Dataset saved to : {CSV_PATH}")

print("\n── First 10 records ──────────────────────────────────────────")
print(df.head(10).to_string())

print(f"\n── Dataset shape ─────────────────────────────────────────────")
print(f"   Rows : {df.shape[0]}   Columns : {df.shape[1]}")

print(f"\n── Feature names (30 numerical features) ─────────────────────")
for i, name in enumerate(raw.feature_names, 1):
    print(f"   {i:2d}. {name}")

print("\n── Descriptive statistics ────────────────────────────────────")
print(df.describe().to_string())

print("\n── Missing values ────────────────────────────────────────────")
mv = df.isnull().sum()
print(mv[mv > 0] if mv.sum() > 0 else "   No missing values found.")

print("\n── Duplicate rows ────────────────────────────────────────────")
dup = df.duplicated().sum()
print(f"   Duplicates : {dup}")

print("\n── Class distribution ────────────────────────────────────────")
counts = df["diagnosis"].value_counts()
print(counts.to_string())

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — DATA PRE-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STAGE 2 — DATA PRE-PROCESSING")
print("="*65)

FEATURES = list(raw.feature_names)   # 30 numerical features
X = df[FEATURES].values              # (569, 30)  numpy array
y = df["target"].values              # (569,)     1=Malignant, 0=Benign

# --- Stratified 80/20 train-test split -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
print(f"\n   Training samples : {len(X_train)}")
print(f"   Test    samples  : {len(X_test)}")

# --- StandardScaler: fit ONLY on training data (avoid leakage) -----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)       # transform only

# Save scaler for inference
joblib.dump(scaler, SCALER_PATH)
print(f"\n✔  Scaler saved to : {SCALER_PATH}")

# --- Create validation split from training data --------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train,
    test_size=0.15, stratify=y_train, random_state=SEED
)
print(f"\n   Train (after val split) : {len(X_tr)}")
print(f"   Validation              : {len(X_val)}")
print(f"   Test                    : {len(X_test)}")

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3 — ANN MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STAGE 3 — ANN MODEL ARCHITECTURE")
print("="*65)

def build_ann(input_dim: int = 30) -> Sequential:
    """Build the Artificial Neural Network.

    Architecture:
        Input  → Dense(64, ReLU) → Dropout(0.3)
               → Dense(32, ReLU) → Dropout(0.2)
               → Dense(1, Sigmoid)
    """
    model = Sequential([
        # --- Hidden Layer 1 -------------------------------------------
        Dense(64, activation="relu", input_shape=(input_dim,),
              name="dense_1"),
        Dropout(0.3, name="dropout_1"),

        # --- Hidden Layer 2 -------------------------------------------
        Dense(32, activation="relu", name="dense_2"),
        Dropout(0.2, name="dropout_2"),

        # --- Output Layer (binary classification) ---------------------
        Dense(1, activation="sigmoid", name="output"),
    ], name="BreastCancer_ANN")

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    return model

model = build_ann(input_dim=X_tr.shape[1])
model.summary()

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4 — MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STAGE 4 — MODEL TRAINING")
print("="*65)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=0
    )
]

history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

print(f"\n✔  Best model saved to : {MODEL_PATH}")

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 5 — MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STAGE 5 — MODEL EVALUATION ON TEST DATA")
print("="*65)

# Load best checkpoint for final evaluation
best_model = tf.keras.models.load_model(MODEL_PATH)

y_pred_prob  = best_model.predict(X_test_scaled).flatten()
y_pred_class = (y_pred_prob >= 0.5).astype(int)

acc    = accuracy_score(y_test, y_pred_class)
prec   = precision_score(y_test, y_pred_class)
rec    = recall_score(y_test, y_pred_class)
f1     = f1_score(y_test, y_pred_class)
auc    = roc_auc_score(y_test, y_pred_prob)
cm     = confusion_matrix(y_test, y_pred_class)
report = classification_report(y_test, y_pred_class,
                                target_names=["Benign", "Malignant"])

print(f"\n   Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"   Precision : {prec:.4f}")
print(f"   Recall    : {rec:.4f}")
print(f"   F1-score  : {f1:.4f}")
print(f"   AUC Score : {auc:.4f}")
print(f"\n── Confusion Matrix ──")
print(cm)
print(f"\n── Classification Report ──")
print(report)

tn, fp, fn, tp = cm.ravel()
print(f"\n   True  Positives (Malignant correctly detected) : {tp}")
print(f"   False Negatives (Malignant missed)              : {fn}")
print(f"   Malignant Recall (Sensitivity)                  : {rec:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 6 — VISUALIZATIONS  (saved to models/ folder)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STAGE 6 — GENERATING VISUALIZATIONS")
print("="*65)

PLOTS_DIR = os.path.join(BASE_DIR, "models")

# ── 1. Class Distribution ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Breast Cancer Dataset — Class Distribution", fontsize=14, fontweight="bold")

counts.plot(kind="bar", ax=axes[0], color=["steelblue", "tomato"],
            edgecolor="black", width=0.5)
axes[0].set_title("Class Count")
axes[0].set_xlabel("Diagnosis")
axes[0].set_ylabel("Count")
axes[0].set_xticklabels(counts.index, rotation=0)
for bar in axes[0].patches:
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 2,
                 int(bar.get_height()), ha="center", va="bottom", fontweight="bold")

axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=["steelblue", "tomato"], startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2))
axes[1].set_title("Class Proportion")

plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "01_class_distribution.png"), dpi=150)
plt.close(fig)
print("   ✔ 01_class_distribution.png")

# ── 2. Training vs Validation Accuracy ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history.history["accuracy"],     label="Train Accuracy",      lw=2, color="royalblue")
ax.plot(history.history["val_accuracy"], label="Validation Accuracy", lw=2, color="darkorange", linestyle="--")
ax.set_title("Training vs Validation Accuracy", fontsize=13, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "02_accuracy_curve.png"), dpi=150)
plt.close(fig)
print("   ✔ 02_accuracy_curve.png")

# ── 3. Training vs Validation Loss ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history.history["loss"],     label="Train Loss",      lw=2, color="royalblue")
ax.plot(history.history["val_loss"], label="Validation Loss", lw=2, color="darkorange", linestyle="--")
ax.set_title("Training vs Validation Loss", fontsize=13, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Binary Cross-Entropy Loss")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "03_loss_curve.png"), dpi=150)
plt.close(fig)
print("   ✔ 03_loss_curve.png")

# ── 4. Confusion Matrix Heatmap ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
            ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("Actual Label",    fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "04_confusion_matrix.png"), dpi=150)
plt.close(fig)
print("   ✔ 04_confusion_matrix.png")

# ── 5. ROC Curve ──────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, lw=2, color="darkorange",
        label=f"ANN  (AUC = {auc:.4f})")
ax.plot([0, 1], [0, 1], lw=1.5, color="navy", linestyle="--",
        label="Random Classifier (AUC = 0.5)")
ax.set_title("ROC Curve — Breast Cancer ANN", fontsize=13, fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "05_roc_curve.png"), dpi=150)
plt.close(fig)
print("   ✔ 05_roc_curve.png")

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 7 — SAMPLE PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  STAGE 7 — SAMPLE PREDICTIONS (first 15 test records)")
print("="*65)

label_map = {0: "Benign", 1: "Malignant"}
print(f"\n{'#':>3}  {'Actual':<12}  {'Predicted':<12}  {'Prob(Malignant)':>16}  {'Status'}")
print("-" * 60)
for i in range(min(15, len(y_test))):
    actual    = label_map[y_test[i]]
    predicted = label_map[y_pred_class[i]]
    prob      = y_pred_prob[i]
    status    = "✓ Correct" if y_test[i] == y_pred_class[i] else "✗ Wrong"
    print(f"{i+1:>3}  {actual:<12}  {predicted:<12}  {prob:>16.4f}  {status}")

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  TRAINING COMPLETE — FINAL SUMMARY")
print("="*65)
print(f"\n   Model   : {MODEL_PATH}")
print(f"   Scaler  : {SCALER_PATH}")
print(f"\n   Test Accuracy  : {acc*100:.2f}%")
print(f"   Test AUC       : {auc:.4f}")
print(f"   Malignant Recall: {rec:.4f}  (sensitivity / true positive rate)")
print(f"\n   Plots saved in : {PLOTS_DIR}/")
print("\n" + "="*65)

# Save final metrics to a text file (used by the report generator)
metrics_path = os.path.join(PLOTS_DIR, "metrics.txt")
with open(metrics_path, "w") as mf:
    mf.write(f"accuracy={acc:.4f}\n")
    mf.write(f"precision={prec:.4f}\n")
    mf.write(f"recall={rec:.4f}\n")
    mf.write(f"f1={f1:.4f}\n")
    mf.write(f"auc={auc:.4f}\n")
    mf.write(f"tn={tn}\nfp={fp}\nfn={fn}\ntp={tp}\n")
    mf.write(f"train_samples={len(X_tr)}\n")
    mf.write(f"val_samples={len(X_val)}\n")
    mf.write(f"test_samples={len(X_test)}\n")
    mf.write(f"epochs_run={len(history.history['loss'])}\n")
    mf.write(f"report_text={report}\n")
print(f"✔  Metrics saved to : {metrics_path}")
