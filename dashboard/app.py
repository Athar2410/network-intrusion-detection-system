"""
NIDS Project — Phase 5: Streamlit Live Dashboard
Reads NDJSON alerts from Phase 4 and shows live metrics/charts.
"""

import json
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="NIDS Live Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR    = Path(__file__).resolve().parents[1]
COMPARE_CSV = BASE_DIR / "output" / "all_models_comparison.csv"

ATTACK_COLORS = {
    "Normal":  "#2E8B57",
    "DoS":     "#D7263D",
    "Probe":   "#F4A261",
    "R2L":     "#7B2CBF",
    "U2R":     "#6A040F",
    "Unknown": "#6C757D",
}

with st.sidebar:
    st.title("🛡️ NIDS Dashboard")
    log_path     = st.text_input("Alerts log path", "/tmp/alerts.json")
    refresh_sec  = st.slider("Refresh every (seconds)", 2, 30, 5)
    rows_to_show = st.slider("Recent alerts to show", 10, 100, 25)
    st.caption("Keep Phase 4 running in another terminal to see live data.")

def read_ndjson(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    rows = []
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")
    return df

def load_model_results() -> pd.DataFrame:
    if COMPARE_CSV.exists():
        df = pd.read_csv(COMPARE_CSV)
        for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace("%","",regex=False).astype(float)
        if "ROC-AUC" in df.columns:
            df["ROC-AUC"] = df["ROC-AUC"].astype(float)
        return df
    return pd.DataFrame([
        {"Model":"Decision Tree","Accuracy":78.45,"Precision":96.47,"Recall":64.50,"F1-Score":77.31,"ROC-AUC":0.8070},
        {"Model":"Random Forest","Accuracy":93.29,"Precision":92.33,"Recall":96.21,"F1-Score":94.23,"ROC-AUC":0.9650},
        {"Model":"XGBoost",      "Accuracy":91.49,"Precision":94.84,"Recall":89.94,"F1-Score":92.32,"ROC-AUC":0.9693},
        {"Model":"DNN",          "Accuracy":87.70,"Precision":92.74,"Recall":85.05,"F1-Score":88.72,"ROC-AUC":0.9426},
        {"Model":"LSTM",         "Accuracy":88.99,"Precision":89.97,"Recall":90.77,"F1-Score":90.37,"ROC-AUC":0.9447},
    ])

alerts    = read_ndjson(log_path)
models_df = load_model_results()

st.title("🛡️ Network Intrusion Detection Dashboard")
st.caption(f"Reading: `{log_path}` · Last updated: {datetime.now().strftime('%H:%M:%S')}")
st.divider()

if alerts.empty:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total alerts",   "0")
    c2.metric("Top category",   "N/A")
    c3.metric("Avg confidence", "N/A")
    c4.metric("Last seen",      "N/A")
    st.warning(
        "⚠️ No alerts found yet.\n\n"
        "**Make sure:**\n"
        "1. Phase 4 is running: `sudo python3 src/04_realtime_detector.py --iface eth0 --log /tmp/alerts.json`\n"
        "2. Generate traffic: `sudo nmap -sS -T4 192.168.237.2`"
    )
else:
    total    = len(alerts)
    top_cat  = alerts["category"].value_counts().idxmax() if "category" in alerts.columns else "N/A"
    avg_conf = alerts["confidence"].mean() * 100 if "confidence" in alerts.columns else 0
    last_ts  = alerts["timestamp"].max() if "timestamp" in alerts.columns else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total alerts",   f"{total:,}")
    c2.metric("Top category",   top_cat)
    c3.metric("Avg confidence", f"{avg_conf:.1f}%")
    c4.metric("Last seen",      last_ts.strftime("%H:%M:%S") if pd.notnull(last_ts) else "N/A")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Attack Distribution")
        counts = alerts["category"].value_counts().reset_index()
        counts.columns = ["category", "count"]
        fig_pie = px.pie(counts, names="category", values="count",
                         color="category", color_discrete_map=ATTACK_COLORS, hole=0.42)
        fig_pie.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    with right:
        st.subheader("Alerts Over Time")
        if "timestamp" in alerts.columns and alerts["timestamp"].notna().sum() > 1:
            tl = alerts.copy()
            tl["minute"] = tl["timestamp"].dt.floor("min")
            grouped = tl.groupby(["minute","category"]).size().reset_index(name="count")
            fig_line = px.line(grouped, x="minute", y="count", color="category",
                               color_discrete_map=ATTACK_COLORS, markers=True)
            fig_line.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=380)
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Not enough data for timeline yet.")

    st.divider()
    st.subheader(f"Recent {rows_to_show} Alerts")
    recent = alerts.tail(rows_to_show).sort_values("timestamp", ascending=False).copy()
    show_cols = [c for c in ["timestamp","src_ip","src_port","dst_ip","dst_port",
                              "protocol","service","category","confidence"]
                 if c in recent.columns]
    if "confidence" in recent.columns:
        recent["confidence"] = (recent["confidence"] * 100).round(1).astype(str) + "%"
    st.dataframe(recent[show_cols], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Alerts by Category")
    bar_df = alerts["category"].value_counts().reset_index()
    bar_df.columns = ["category", "count"]
    fig_bar_cat = px.bar(bar_df, x="category", y="count", color="category",
                         color_discrete_map=ATTACK_COLORS, text="count")
    fig_bar_cat.update_traces(textposition="outside")
    fig_bar_cat.update_layout(showlegend=False, height=350, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig_bar_cat, use_container_width=True)

st.divider()
st.subheader("📊 Model Performance (All Phases)")
metric = st.selectbox("Compare by metric",
                      ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"], index=2)
fig_model = px.bar(models_df, x="Model", y=metric, color="Model",
                   text=metric, title=f"All Models — {metric}")
fig_model.update_traces(textposition="outside")
fig_model.update_layout(showlegend=False, height=420, margin=dict(l=10,r=10,t=50,b=10))
st.plotly_chart(fig_model, use_container_width=True)
st.dataframe(models_df, use_container_width=True, hide_index=True)

# Auto-refresh LAST — after all content has rendered
time.sleep(refresh_sec)
st.rerun()
