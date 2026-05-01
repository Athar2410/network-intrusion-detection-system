# 🛡️ Network Intrusion Detection System (NIDS)
### ML + AI + Live Detection on Kali Linux

A full end-to-end Network Intrusion Detection System that combines classical machine learning, deep learning, and real-time packet analysis to detect network attacks with high accuracy.

---

## 🗂️ Project Structure

```
nids_project/
├── data/                          ← NSL-KDD dataset
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── src/
│   ├── 01_eda_preprocessing.py    ← Phase 1: EDA + feature engineering
│   ├── 02_classical_ml.py         ← Phase 2: DT, RF, XGBoost
│   ├── 03_deep_learning.py        ← Phase 3: DNN + LSTM
│   ├── feature_extractor.py       ← Phase 4: Packet → 41 NSL-KDD features
│   └── 04_realtime_detector.py    ← Phase 4: Live Scapy sniffer
├── dashboard/
│   ├── app.py                     ← Phase 5: Streamlit live dashboard
│   └── soc.py                     ← Phase 6: SOC investigation + PDF export
├── models/                        ← Saved trained models (.pkl, .keras)
├── output/                        ← Charts, preprocessed arrays, CSVs
├── kali_setup.sh                  ← Kali one-time setup script
└── requirements.txt
```

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree | 78.45% | 96.47% | 64.50% | 77.31% | 0.8070 |
| **Random Forest** | **93.29%** | **92.33%** | **96.21%** | **94.23%** | **0.9650** |
| XGBoost | 91.49% | 94.84% | 89.94% | 92.32% | 0.9693 |
| DNN | 87.70% | 92.74% | 85.05% | 88.72% | 0.9426 |
| LSTM | 88.99% | 89.97% | 90.77% | 90.37% | 0.9447 |

> **Deployment model:** Random Forest — highest recall (96.21%) minimises missed attacks.

---

## 🚀 Quick Start

### Phase 1–3: ML Training (Windows)
```bash
pip install -r requirements.txt
python src/01_eda_preprocessing.py
python src/02_classical_ml.py
python src/03_deep_learning.py
```

### Phase 4: Live Detection (Kali Linux)
```bash
bash kali_setup.sh
sudo python3 src/04_realtime_detector.py --iface eth0 --log /tmp/alerts.json
```

### Phase 5–6: Dashboard (Kali Linux)
```bash
python3 -m venv ~/nids_venv && source ~/nids_venv/bin/activate
pip install streamlit plotly fpdf2
streamlit run dashboard/app.py       # Live dashboard
streamlit run dashboard/soc.py       # SOC investigation + PDF export
```

---

## 🔬 Dataset

**NSL-KDD** — the standard benchmark for NIDS research.

- Training set: 125,973 records | Test set: 22,544 records
- 41 features per connection record
- 5 categories: Normal, DoS, Probe, R2L, U2R

---

## 🏗️ Architecture

```
Raw Network Traffic
       │
       ▼
Scapy Packet Sniffer (eth0)
       │
       ▼
Feature Extractor (41 NSL-KDD features)
       │
  ┌────▼────────────────────┐
  │  MinMaxScaler + Encoder  │  ← trained on NSL-KDD
  └────┬────────────────────┘
       │
       ▼
Random Forest Classifier
  ├── Binary: Normal / Attack
  └── Multi-class: DoS / Probe / R2L / U2R
       │
       ▼
NDJSON Alert Log (/tmp/alerts.json)
       │
       ▼
Streamlit Dashboard (localhost:8501)
```

---

## ⚔️ Attack Categories Detected

| Category | Description | Examples |
|---|---|---|
| **DoS** | Denial of Service — flood target to exhaust resources | SYN Flood, ICMP Flood, UDP Flood |
| **Probe** | Reconnaissance — scan network to find vulnerabilities | Nmap SYN Scan, Port Sweep |
| **R2L** | Remote to Local — unauthorised access from remote machine | FTP brute force, SSH guessing |
| **U2R** | User to Root — privilege escalation on local machine | Buffer overflow, rootkit |

---

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| ML Models | scikit-learn, XGBoost |
| Deep Learning | TensorFlow / Keras |
| Packet Capture | Scapy |
| Dashboard | Streamlit, Plotly |
| PDF Reports | fpdf2 |
| Environment | Windows (training) + Kali Linux VMware (detection) |

---

## 📄 License
MIT License — free to use for educational and research purposes.
