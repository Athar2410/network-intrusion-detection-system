"""
=============================================================
 NIDS Project — Phase 3: Deep Learning (DNN + LSTM)
 Models  : Deep Neural Network, LSTM
 Requires: tensorflow>=2.12  →  pip install tensorflow
 Run AFTER 02_classical_ml.py
=============================================================
"""

import os, time, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from tabulate import tabulate

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, classification_report
)
from sklearn.preprocessing import LabelEncoder

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import (
        Dense, Dropout, BatchNormalization,
        LSTM, Reshape, Input
    )
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    print(f"[✓] TensorFlow {tf.__version__} loaded")
except ImportError:
    print("[✗] TensorFlow not found. Run:  pip install tensorflow")
    raise SystemExit(1)

# ─── PATHS ────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── REPRODUCIBILITY ──────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─── 1. LOAD DATA ─────────────────────────────────────────────
def load_data():
    X_train    = np.load(os.path.join(OUTPUT_DIR, "X_train.npy"))
    X_test     = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
    y_train    = np.load(os.path.join(OUTPUT_DIR, "y_train_bin.npy"))
    y_test     = np.load(os.path.join(OUTPUT_DIR, "y_test_bin.npy"))
    y_train_mc = np.load(os.path.join(OUTPUT_DIR, "y_train_cat.npy"), allow_pickle=True)
    y_test_mc  = np.load(os.path.join(OUTPUT_DIR, "y_test_cat.npy"),  allow_pickle=True)
    print(f"[✓] X_train:{X_train.shape} | X_test:{X_test.shape}")
    return X_train, X_test, y_train, y_test, y_train_mc, y_test_mc

# ─── 2. BUILD DNN ─────────────────────────────────────────────
def build_dnn(input_dim):
    """
    5-layer Deep Neural Network:
    Input → Dense(256) → BN → Dropout
          → Dense(128) → BN → Dropout
          → Dense(64)  → BN → Dropout
          → Dense(32)
          → Output(sigmoid)
    """
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation='relu'),

        Dense(1, activation='sigmoid')
    ], name="DNN_NIDS")

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    return model

# ─── 3. BUILD LSTM ────────────────────────────────────────────
def build_lstm(input_dim):
    """
    Reshape flat feature vector → (timesteps=1, features=input_dim)
    Then pass through LSTM layers.
    This treats each connection record as a single timestep.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Reshape((1, input_dim)),          # (batch, 1, 41) — single timestep

        LSTM(128, return_sequences=True),
        Dropout(0.3),

        LSTM(64, return_sequences=False),
        Dropout(0.2),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(32, activation='relu'),

        Dense(1, activation='sigmoid')
    ], name="LSTM_NIDS")

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    return model

# ─── 4. CALLBACKS ─────────────────────────────────────────────
def get_callbacks(model_name):
    ckpt_path = os.path.join(MODELS_DIR, f"{model_name}_best.keras")
    return [
        EarlyStopping(
            monitor='val_auc', patience=5,
            mode='max', restore_best_weights=True,
            verbose=1),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(
            ckpt_path, monitor='val_auc',
            save_best_only=True, mode='max', verbose=0)
    ]

# ─── 5. TRAIN & EVALUATE ──────────────────────────────────────
def train_evaluate_dl(model, model_name, X_train, X_test, y_train, y_test):
    print(f"\n{'='*55}")
    print(f"  Training {model_name}")
    print(f"{'='*55}")
    model.summary()

    # Class weights for imbalance
    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    cw = {0: 1.0, 1: neg / pos}
    print(f"[i] Class weights → Normal:1.0 | Attack:{neg/pos:.2f}")

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=50,
        batch_size=512,
        class_weight=cw,
        callbacks=get_callbacks(model_name),
        verbose=1
    )
    train_time = time.time() - t0

    # Predict with optimal Youden's J threshold
    y_prob = model.predict(X_test, verbose=0).flatten()
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    j_scores = tpr - fpr
    best_thr = thresholds[np.argmax(j_scores)]
    y_pred = (y_prob >= best_thr).astype(int)
    print(f"[i] Optimal threshold: {best_thr:.4f}")

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)

    print(f"\n[✓] {model_name} Results:")
    print(f"    Accuracy : {acc*100:.2f}%")
    print(f"    Precision: {prec*100:.2f}%")
    print(f"    Recall   : {rec*100:.2f}%")
    print(f"    F1-Score : {f1*100:.2f}%")
    print(f"    ROC-AUC  : {auc:.4f}")
    print(f"    Time     : {train_time:.1f}s")

    model.save(os.path.join(MODELS_DIR, f"{model_name}.keras"))
    print(f"[✓] Model saved: models/{model_name}.keras")

    return history, y_pred, y_prob, {
        "Model":      model_name,
        "Accuracy":   f"{acc*100:.2f}%",
        "Precision":  f"{prec*100:.2f}%",
        "Recall":     f"{rec*100:.2f}%",
        "F1-Score":   f"{f1*100:.2f}%",
        "ROC-AUC":    f"{auc:.4f}",
        "Train Time": f"{train_time:.0f}s"
    }

# ─── 6. PLOT TRAINING HISTORY ─────────────────────────────────
def plot_training_history(hist_dnn, hist_lstm):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Deep Learning Training History", fontsize=14, fontweight='bold')

    for col, (hist, name) in enumerate([(hist_dnn, "DNN"), (hist_lstm, "LSTM")]):
        # Accuracy
        axes[0][col].plot(hist.history['accuracy'],     label='Train Acc', color='#01696f')
        axes[0][col].plot(hist.history['val_accuracy'], label='Val Acc',   color='#da7101', linestyle='--')
        axes[0][col].set_title(f"{name} — Accuracy")
        axes[0][col].set_xlabel("Epoch"); axes[0][col].set_ylabel("Accuracy")
        axes[0][col].legend(); axes[0][col].grid(alpha=0.3)

        # Loss
        axes[1][col].plot(hist.history['loss'],     label='Train Loss', color='#a12c7b')
        axes[1][col].plot(hist.history['val_loss'], label='Val Loss',   color='#6daa45', linestyle='--')
        axes[1][col].set_title(f"{name} — Loss")
        axes[1][col].set_xlabel("Epoch"); axes[1][col].set_ylabel("Loss")
        axes[1][col].legend(); axes[1][col].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "dl_training_history.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: {out}")

# ─── 7. PLOT ROC CURVES (ALL MODELS) ──────────────────────────
def plot_roc_curves(y_test, dnn_probs, lstm_probs):
    # Load classical ML models to get their probs too
    model_probs = {}
    for mname in ["decision_tree_binary", "random_forest_binary", "xgboost_binary"]:
        mpath = os.path.join(MODELS_DIR, f"{mname}.pkl")
        if os.path.exists(mpath):
            clf = joblib.load(mpath)
            if hasattr(clf, "predict_proba"):
                model_probs[mname.replace("_binary","").replace("_"," ").title()] = \
                    clf.predict_proba(
                        np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
                    )[:, 1]

    model_probs["DNN"]  = dnn_probs
    model_probs["LSTM"] = lstm_probs

    colors = ['#bab9b4', '#7a7974', '#4f98a3', '#01696f', '#a12c7b']
    plt.figure(figsize=(9, 7))

    for (mname, probs), color in zip(model_probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, label=f"{mname}  (AUC={auc:.4f})", color=color, linewidth=2)

    plt.plot([0,1],[0,1],'k--', alpha=0.4, label='Random Classifier')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models", fontweight='bold', fontsize=13)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "roc_curves_all_models.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: {out}")

# ─── 8. PLOT CONFUSION MATRICES (DL ONLY) ─────────────────────
def plot_dl_confusion_matrices(y_test, dnn_pred, lstm_pred):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, y_pred) in zip(axes, [("DNN", dnn_pred), ("LSTM", lstm_pred)]):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Attack'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(f"{name} — Confusion Matrix", fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "dl_confusion_matrices.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: {out}")

# ─── 9. FULL MODEL COMPARISON TABLE ───────────────────────────
def print_full_comparison(dl_results):
    classical = [
        {"Model": "Decision Tree", "Accuracy": "78.45%", "Precision": "96.47%",
         "Recall": "64.50%", "F1-Score": "77.31%", "ROC-AUC": "0.8070", "Train Time": "0.7s"},
        {"Model": "Random Forest", "Accuracy": "93.29%", "Precision": "92.33%",
         "Recall": "96.21%", "F1-Score": "94.23%", "ROC-AUC": "0.9650", "Train Time": "2.3s"},
        {"Model": "XGBoost",       "Accuracy": "91.49%", "Precision": "94.84%",
         "Recall": "89.94%", "F1-Score": "92.32%", "ROC-AUC": "0.9693", "Train Time": "0.7s"},
    ]
    all_results = classical + dl_results
    print("\n\n========== FULL MODEL COMPARISON (ALL PHASES) ==========")
    print(tabulate(all_results, headers="keys", tablefmt="rounded_outline"))

    # Save to CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_DIR, "all_models_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"[✓] Saved: {csv_path}")

# ─── 10. PLOT FULL COMPARISON BAR CHART ───────────────────────
def plot_full_comparison():
    csv_path = os.path.join(OUTPUT_DIR, "all_models_comparison.csv")
    df = pd.read_csv(csv_path)

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_names = df["Model"].tolist()
    colors = ['#01696f', '#4f98a3', '#da7101', '#6daa45']

    values = {m: [float(df.loc[df["Model"] == n, m].values[0].strip('%'))
                  for n in model_names]
              for m in metrics}

    x = np.arange(len(model_names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * width, values[metric], width,
                      label=metric, color=color, edgecolor='white')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{bar.get_height():.1f}%",
                    ha='center', va='bottom', fontsize=6.5, rotation=45)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=10, rotation=10)
    ax.set_ylim(55, 108)
    ax.set_ylabel("Score (%)")
    ax.set_title("All Models — Full Performance Comparison\n(Classical ML + Deep Learning)",
                 fontweight='bold', fontsize=13)
    ax.legend(loc='lower right')
    ax.axvline(x=2.7, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.text(0.5, 102, "Classical ML", ha='center', fontsize=9, color='gray')
    ax.text(3.4, 102, "Deep Learning", ha='center', fontsize=9, color='gray')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "full_comparison_chart.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: {out}")

# ─── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, y_train_mc, y_test_mc = load_data()
    input_dim = X_train.shape[1]

    # ── DNN ──────────────────────────────────────────────────
    dnn = build_dnn(input_dim)
    hist_dnn, dnn_pred, dnn_prob, dnn_result = train_evaluate_dl(
        dnn, "DNN", X_train, X_test, y_train, y_test)

    # ── LSTM ─────────────────────────────────────────────────
    lstm = build_lstm(input_dim)
    hist_lstm, lstm_pred, lstm_prob, lstm_result = train_evaluate_dl(
        lstm, "LSTM", X_train, X_test, y_train, y_test)

    # ── PLOTS ─────────────────────────────────────────────────
    plot_training_history(hist_dnn, hist_lstm)
    plot_roc_curves(y_test, dnn_prob, lstm_prob)
    plot_dl_confusion_matrices(y_test, dnn_pred, lstm_pred)
    print_full_comparison([dnn_result, lstm_result])
    plot_full_comparison()

    print("\n[✓] Phase 3 complete!")
    print("    Models saved : models/DNN.keras | models/LSTM.keras")
    print("    Next → Phase 4: Real-Time Detection with Scapy (on Kali)")
