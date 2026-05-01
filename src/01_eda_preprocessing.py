"""
=============================================================
 NIDS Project — Phase 1: EDA & Preprocessing
 Dataset  : NSL-KDD (auto-downloaded from GitHub mirror)
 Run with : python 01_eda_preprocessing.py
            or paste cells into Google Colab
=============================================================
"""

import os, urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

# ─── CONFIG ──────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
TEST_URL  = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
    'is_host_login','is_guest_login','count','srv_count','serror_rate',
    'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
    'dst_host_srv_rerror_rate','label','difficulty_level'
]

ATTACK_MAP = {
    'normal': 'Normal',
    # DoS
    'back':'DoS','land':'DoS','neptune':'DoS','pod':'DoS','smurf':'DoS',
    'teardrop':'DoS','apache2':'DoS','udpstorm':'DoS','processtable':'DoS',
    'mailbomb':'DoS','worm':'DoS',
    # Probe
    'ipsweep':'Probe','nmap':'Probe','portsweep':'Probe','satan':'Probe',
    'mscan':'Probe','saint':'Probe',
    # R2L
    'ftp_write':'R2L','guess_passwd':'R2L','imap':'R2L','multihop':'R2L',
    'phf':'R2L','spy':'R2L','warezclient':'R2L','warezmaster':'R2L',
    'sendmail':'R2L','named':'R2L','snmpgetattack':'R2L','snmpguess':'R2L',
    'xlock':'R2L','xsnoop':'R2L','httptunnel':'R2L',
    # U2R
    'buffer_overflow':'U2R','loadmodule':'U2R','perl':'U2R','rootkit':'U2R',
    'xterm':'U2R','ps':'U2R','sqlattack':'U2R',
}

CATEGORICAL = ['protocol_type', 'service', 'flag']

# ─── 1. DOWNLOAD DATASET ─────────────────────────────────────
def download_data():
    for name, url in [("KDDTrain+.txt", TRAIN_URL), ("KDDTest+.txt", TEST_URL)]:
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            print(f"[↓] Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
            print(f"[✓] Saved to {path}")
        else:
            print(f"[✓] {name} already exists.")

# ─── 2. LOAD DATASET ─────────────────────────────────────────
def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "KDDTrain+.txt"),
                        header=None, names=COLUMNS)
    test  = pd.read_csv(os.path.join(DATA_DIR, "KDDTest+.txt"),
                        header=None, names=COLUMNS)
    print(f"[i] Train shape: {train.shape} | Test shape: {test.shape}")
    return train, test

# ─── 3. BASIC EDA ─────────────────────────────────────────────
def run_eda(df):
    print("\n========== EDA SUMMARY ==========")
    print(f"Shape          : {df.shape}")
    print(f"Missing values : {df.isnull().sum().sum()}")
    print(f"Duplicates     : {df.duplicated().sum()}")
    print("\nLabel distribution (raw):")
    print(df['label'].value_counts().head(20))
    print("\nDtypes sample:")
    print(df.dtypes.value_counts())

# ─── 4. MAP ATTACKS → 5-CLASS CATEGORY ───────────────────────
def map_attacks(df):
    df = df.copy()
    df['attack_category'] = df['label'].str.strip().map(
        lambda x: ATTACK_MAP.get(x, 'Unknown')
    )
    df['binary_label'] = df['attack_category'].apply(
        lambda x: 0 if x == 'Normal' else 1
    )
    print("\n[i] Attack category distribution:")
    print(df['attack_category'].value_counts())
    return df

# ─── 5. PLOT CLASS DISTRIBUTION ───────────────────────────────
def plot_distributions(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 5-class
    counts = df['attack_category'].value_counts()
    colors = ['#01696f','#4f98a3','#da7101','#a12c7b','#6daa45']
    axes[0].bar(counts.index, counts.values, color=colors[:len(counts)])
    axes[0].set_title("5-Class Attack Category Distribution", fontweight='bold')
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("Count")
    for i, (idx, val) in enumerate(counts.items()):
        axes[0].text(i, val + 300, f'{val:,}', ha='center', fontsize=9)

    # Binary
    bin_counts = df['binary_label'].value_counts()
    labels = ['Normal (0)', 'Attack (1)']
    axes[1].pie(bin_counts.values, labels=labels, autopct='%1.1f%%',
                colors=['#01696f', '#a12c7b'], startangle=140,
                wedgeprops={'edgecolor':'white','linewidth':2})
    axes[1].set_title("Binary Label Distribution", fontweight='bold')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "class_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: {out}")

# ─── 6. PROTOCOL & SERVICE PLOTS ──────────────────────────────
def plot_protocol_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    proto = df.groupby(['protocol_type','attack_category']).size().unstack(fill_value=0)
    proto.plot(kind='bar', ax=axes[0], colormap='Set2', edgecolor='white')
    axes[0].set_title("Attack Categories per Protocol", fontweight='bold')
    axes[0].set_xlabel("Protocol")
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].legend(loc='upper right', fontsize=8)

    top_services = df['service'].value_counts().head(15)
    axes[1].barh(top_services.index, top_services.values, color='#4f98a3')
    axes[1].set_title("Top 15 Network Services", fontweight='bold')
    axes[1].set_xlabel("Count")
    axes[1].invert_yaxis()

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "protocol_service_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: {out}")

# ─── 7. CORRELATION HEATMAP (numeric features) ────────────────
def plot_correlation(df):
    numeric_df = df.select_dtypes(include=[np.number]).drop(
        columns=['difficulty_level','binary_label'], errors='ignore'
    )
    corr = numeric_df.corr()
    plt.figure(figsize=(18, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='RdYlGn', center=0,
                linewidths=0.3, annot=False, square=True,
                cbar_kws={'shrink': 0.6})
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved: {out}")

# ─── 8. PREPROCESSING ─────────────────────────────────────────
def preprocess(train_df, test_df):
    print("\n[i] Preprocessing ...")

    # Drop difficulty_level (not a feature)
    train_df = train_df.drop(columns=['difficulty_level'])
    test_df  = test_df.drop(columns=['difficulty_level'])

    # Encode categoricals with LabelEncoder
    le_dict = {}
    for col in CATEGORICAL:
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]], axis=0)
        le.fit(combined)
        train_df[col] = le.transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])
        le_dict[col]  = le

    # Save label encoders
    joblib.dump(le_dict, os.path.join(OUTPUT_DIR, "label_encoders.pkl"))

    # Feature matrix
    feature_cols = [c for c in train_df.columns
                    if c not in ['label','attack_category','binary_label']]

    X_train = train_df[feature_cols].values.astype(np.float32)
    X_test  = test_df[feature_cols].values.astype(np.float32)

    # MinMax scaling
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

    y_train_bin = train_df['binary_label'].values
    y_test_bin  = test_df['binary_label'].values
    y_train_cat = train_df['attack_category'].values
    y_test_cat  = test_df['attack_category'].values

    print(f"[✓] X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"[✓] Feature columns: {feature_cols}")

    # Save preprocessed data
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train_bin.npy"), y_train_bin)
    np.save(os.path.join(OUTPUT_DIR, "y_test_bin.npy"),  y_test_bin)
    np.save(os.path.join(OUTPUT_DIR, "y_train_cat.npy"), y_train_cat)
    np.save(os.path.join(OUTPUT_DIR, "y_test_cat.npy"),  y_test_cat)

    # Save feature names for later use
    with open(os.path.join(OUTPUT_DIR, "feature_cols.txt"), "w") as f:
        f.write("\n".join(feature_cols))

    print("[✓] Preprocessed arrays saved to output/")
    return X_train, X_test, y_train_bin, y_test_bin, y_train_cat, y_test_cat


# ─── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    download_data()
    train_df, test_df = load_data()

    run_eda(train_df)

    train_df = map_attacks(train_df)
    test_df  = map_attacks(test_df)

    plot_distributions(train_df)
    plot_protocol_analysis(train_df)
    plot_correlation(train_df)

    X_train, X_test, y_train_bin, y_test_bin, y_train_cat, y_test_cat = \
        preprocess(train_df, test_df)

    print("\n[✓] Phase 1 complete! Run 02_classical_ml.py next.")
