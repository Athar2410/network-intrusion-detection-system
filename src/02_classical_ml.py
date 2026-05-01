"""
=============================================================
 NIDS Project — Phase 2: Classical ML Models
 Trains: Decision Tree, Random Forest, XGBoost
 Evaluates: Accuracy, Precision, Recall, F1, ROC-AUC
 Run AFTER 01_eda_preprocessing.py
=============================================================
"""

import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tabulate import tabulate

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,roc_curve
)
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[!] XGBoost not installed. Run: pip install xgboost")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ─── LOAD PREPROCESSED DATA ───────────────────────────────────
def load_data():
    X_train    = np.load(os.path.join(OUTPUT_DIR, "X_train.npy"))
    X_test     = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
    y_train    = np.load(os.path.join(OUTPUT_DIR, "y_train_bin.npy"))
    y_test     = np.load(os.path.join(OUTPUT_DIR, "y_test_bin.npy"))
    y_train_mc = np.load(os.path.join(OUTPUT_DIR, "y_train_cat.npy"), allow_pickle=True)
    y_test_mc  = np.load(os.path.join(OUTPUT_DIR, "y_test_cat.npy"),  allow_pickle=True)
    print(f"[✓] Loaded — X_train:{X_train.shape} | X_test:{X_test.shape}")
    return X_train, X_test, y_train, y_test, y_train_mc, y_test_mc


# ─── TRAIN & EVALUATE (BINARY) ────────────────────────────────
def train_evaluate_binary(X_train, X_test, y_train, y_test):
    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    ratio = neg / pos
    print(f"[i] Class ratio (Normal/Attack): {ratio:.2f}")

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=20, min_samples_split=5,
            class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None,
            class_weight='balanced', n_jobs=-1, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            scale_pos_weight=ratio,
            eval_metric='logloss',
            n_jobs=-1, random_state=42)

    results = []
    trained = {}

    for name, model in models.items():                          # ← LOOP STARTS
        print(f"\n[→] Training {name} ...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        if hasattr(model, "predict_proba"):                     # ← INSIDE LOOP
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            j_scores = tpr - fpr
            best_threshold = thresholds[np.argmax(j_scores)]
            print(f"[i] Optimal threshold for {name}: {best_threshold:.4f}")
            y_pred = (y_prob >= best_threshold).astype(int)
        else:                                                    # ← INSIDE LOOP
            y_prob = None
            y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)                   # ← INSIDE LOOP
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else None

        results.append({                                        # ← INSIDE LOOP
            "Model":      name,
            "Accuracy":   f"{acc*100:.2f}%",
            "Precision":  f"{prec*100:.2f}%",
            "Recall":     f"{rec*100:.2f}%",
            "F1-Score":   f"{f1*100:.2f}%",
            "ROC-AUC":    f"{auc:.4f}" if auc else "N/A",
            "Train Time": f"{train_time:.1f}s"
        })
        trained[name] = (model, y_pred)                        # ← INSIDE LOOP

        safe_name = name.replace(" ", "_").lower()             # ← INSIDE LOOP
        joblib.dump(model, os.path.join(MODELS_DIR, f"{safe_name}_binary.pkl"))
        print(f"[✓] Saved model: models/{safe_name}_binary.pkl")
                                                                # ← LOOP ENDS HERE
    print("\n\n========== BINARY CLASSIFICATION RESULTS ==========")
    print(tabulate(results, headers="keys", tablefmt="rounded_outline"))
    return trained, results


# ─── MULTI-CLASS (5-class: Normal/DoS/Probe/R2L/U2R) ──────────
def train_evaluate_multiclass(X_train, X_test, y_train_mc, y_test_mc):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_mc)
    y_test_enc  = le.transform(y_test_mc)
    classes     = le.classes_
    joblib.dump(le, os.path.join(OUTPUT_DIR, "category_label_encoder.pkl"))

    print(f"\n[i] Multi-class labels: {list(classes)}")

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    print("\n[→] Training Random Forest (multi-class) ...")
    t0 = time.time()
    rf.fit(X_train, y_train_enc)
    print(f"[✓] Done in {time.time()-t0:.1f}s")

    y_pred = rf.predict(X_test)
    print("\n========== MULTI-CLASS CLASSIFICATION REPORT ==========")
    print(classification_report(y_test_enc, y_pred,
                                target_names=classes, zero_division=0))

    joblib.dump(rf, os.path.join(MODELS_DIR, "rf_multiclass.pkl"))
    return rf, y_pred, y_test_enc, classes


# ─── PLOT CONFUSION MATRICES ──────────────────────────────────
def plot_confusion_matrices(trained_bin, y_test_bin,
                             y_pred_mc, y_test_mc_enc, classes):
    fig, axes = plt.subplots(1, len(trained_bin) + 1,
                              figsize=(6 * (len(trained_bin) + 1), 5))
    if len(trained_bin) + 1 == 1:
        axes = [axes]

    for ax, (name, (_, y_pred)) in zip(axes, trained_bin.items()):
        cm = confusion_matrix(y_test_bin, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Attack'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(f"{name}\n(Binary)", fontweight='bold')

    # Multi-class confusion matrix
    ax_mc = axes[-1]
    cm_mc = confusion_matrix(y_test_mc_enc, y_pred_mc)
    sns.heatmap(cm_mc, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax_mc,
                linewidths=0.4, cbar=False)
    ax_mc.set_title("Random Forest\n(5-Class)", fontweight='bold')
    ax_mc.set_xlabel("Predicted")
    ax_mc.set_ylabel("Actual")
    ax_mc.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: {out}")


# ─── PLOT FEATURE IMPORTANCES ─────────────────────────────────
def plot_feature_importances(rf_model):
    feat_path = os.path.join(OUTPUT_DIR, "feature_cols.txt")
    with open(feat_path) as f:
        feature_names = f.read().splitlines()

    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]

    plt.figure(figsize=(10, 6))
    plt.barh(
        [feature_names[i] for i in indices][::-1],
        importances[indices][::-1],
        color='#01696f'
    )
    plt.xlabel("Importance Score")
    plt.title("Top 20 Feature Importances (Random Forest)", fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "feature_importances.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: {out}")


# ─── PLOT MODEL COMPARISON BAR CHART ─────────────────────────
def plot_model_comparison(results):
    df = pd.DataFrame(results)
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_names = df["Model"].tolist()

    # Convert percentage strings to floats
    values = {m: [float(df.loc[df["Model"] == n, m].values[0].strip('%'))
                  for n in model_names]
              for m in metrics}

    x = np.arange(len(model_names))
    width = 0.2
    colors = ['#01696f', '#4f98a3', '#da7101', '#6daa45']

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * width, values[metric], width,
                      label=metric, color=color, edgecolor='white')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%",
                    ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(85, 102)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Performance Comparison (Binary Classification)",
                 fontweight='bold', fontsize=13)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: {out}")


# ─── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, y_train_mc, y_test_mc = load_data()

    trained_bin, results = train_evaluate_binary(
        X_train, X_test, y_train, y_test)

    rf_mc, y_pred_mc, y_test_mc_enc, classes = train_evaluate_multiclass(
        X_train, X_test, y_train_mc, y_test_mc)

    plot_confusion_matrices(trained_bin, y_test, y_pred_mc, y_test_mc_enc, classes)
    plot_feature_importances(rf_mc)
    plot_model_comparison(results)

    print("\n[✓] Phase 2 complete! Models saved to models/")
    print("    Next → Phase 3: Deep Learning (LSTM/DNN)")
