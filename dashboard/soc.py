"""
NIDS Project — Phase 6: SOC Investigation Dashboard
Incident timeline, attack details, and PDF incident report export.
"""

import json
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="NIDS SOC Console",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEFAULT_LOG = "/tmp/alerts.json"
ATTACK_COLORS = {
    "Normal":  "#2E8B57",
    "DoS":     "#D7263D",
    "Probe":   "#F4A261",
    "R2L":     "#7B2CBF",
    "U2R":     "#6A040F",
    "Unknown": "#6C757D",
}

SEVERITY = {
    "DoS":     ("CRITICAL", "#D7263D"),
    "U2R":     ("CRITICAL", "#D7263D"),
    "R2L":     ("HIGH",     "#F4A261"),
    "Probe":   ("MEDIUM",   "#FFD166"),
    "Normal":  ("INFO",     "#2E8B57"),
    "Unknown": ("LOW",      "#6C757D"),
}

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 SOC Console")
    log_path     = st.text_input("Alerts log", DEFAULT_LOG)
    refresh_sec  = st.slider("Auto-refresh (s)", 2, 30, 5)
    category_filter = st.multiselect(
        "Filter by category",
        ["DoS", "Probe", "R2L", "U2R", "Normal"],
        default=["DoS", "Probe", "R2L", "U2R"]
    )
    min_conf = st.slider("Min confidence (%)", 0, 100, 0)
    st.divider()
    st.caption("Phase 6 — SOC Investigation + PDF Export")

# ── Helpers ───────────────────────────────────────────────────
def read_ndjson(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    rows = []
    with open(p) as f:
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
    if "confidence" in df.columns:
        df["confidence"] = df["confidence"].astype(float)
    return df


def make_pdf(df: pd.DataFrame, summary: dict) -> bytes:
    try:
        from fpdf import FPDF
    except ImportError:
        return b""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_fill_color(30, 30, 30)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 14, "NIDS INCIDENT REPORT", new_x="LMARGIN", new_y="NEXT", fill=True, align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8,
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}    |    Log: {summary.get('log_path','')}",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    # Summary
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "EXECUTIVE SUMMARY", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)

    items = [
        ("Total alerts",   str(summary.get("total", 0))),
        ("Attack alerts",  str(summary.get("attacks", 0))),
        ("Normal traffic", str(summary.get("normal", 0))),
        ("Attack rate",    f"{summary.get('attack_rate', 0):.1f}%"),
        ("Top category",   summary.get("top_cat", "N/A")),
        ("Avg confidence", f"{summary.get('avg_conf', 0):.1f}%"),
        ("Period start",   summary.get("start", "N/A")),
        ("Period end",     summary.get("end", "N/A")),
    ]
    for label, val in items:
        pdf.cell(60, 7, label + ":", border=0)
        pdf.cell(0, 7, val, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Category breakdown
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "ALERTS BY CATEGORY", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 10)
    col_w = [50, 30, 40, 40]
    for header, w in zip(["Category", "Count", "Severity", "Avg Conf (%)"], col_w):
        pdf.cell(w, 8, header, border=1, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 10)
    for _, row in summary.get("cat_table", pd.DataFrame()).iterrows():
        cat = str(row.get("category", ""))
        sev_label, _ = SEVERITY.get(cat, ("LOW", "#6C757D"))
        pdf.cell(50, 7, cat, border=1)
        pdf.cell(30, 7, str(row.get("count", 0)), border=1, align="C")
        pdf.cell(40, 7, sev_label, border=1, align="C")
        pdf.cell(40, 7, f"{row.get('avg_conf', 0):.1f}", border=1, align="C")
        pdf.ln()
    pdf.ln(4)

    # Recent alerts table
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f"RECENT ALERTS (last {min(25, len(df))})", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 8)
    headers = ["Timestamp", "Src IP", "Dst IP", "Proto", "Category", "Conf%"]
    widths  = [42, 35, 35, 20, 28, 22]
    for h, w in zip(headers, widths):
        pdf.cell(w, 7, h, border=1, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    recent = df[df["category"] != "Normal"].tail(25) if "category" in df.columns else df.tail(25)
    for _, row in recent.iterrows():
        ts = row.get("timestamp")
        ts_s = ts.strftime("%m-%d %H:%M:%S") if pd.notnull(ts) else "N/A"
        vals = [
            ts_s,
            str(row.get("src_ip", ""))[:15],
            str(row.get("dst_ip", ""))[:15],
            str(row.get("protocol", "")).upper()[:5],
            str(row.get("category", "")),
            f"{float(row.get('confidence', 0))*100:.1f}",
        ]
        for v, w in zip(vals, widths):
            pdf.cell(w, 6, v, border=1)
        pdf.ln()
    pdf.ln(4)

    # Recommendations
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "RECOMMENDED ACTIONS", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    recs = {
        "DoS":   "Block source IPs at firewall. Enable rate limiting on exposed ports. Alert NOC team.",
        "Probe": "Investigate scanning source IP. Check for follow-up intrusion attempts. Update IDS signatures.",
        "R2L":   "Audit remote access logs. Check for successful logins from flagged IPs. Reset credentials.",
        "U2R":   "Isolate affected host immediately. Run rootkit scan. Escalate to Tier 2 SOC analyst.",
    }
    top_cat = summary.get("top_cat", "")
    rec_text = recs.get(top_cat, "Continue monitoring. Review logs for anomalous patterns.")
    pdf.multi_cell(0, 7, f"Primary threat ({top_cat}): {rec_text}")
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "Report auto-generated by NIDS Phase 6 — Atharva Amle", align="C")

    return bytes(pdf.output())


# ── Load + filter data ────────────────────────────────────────
all_alerts = read_ndjson(log_path)

if not all_alerts.empty:
    alerts = all_alerts.copy()
    if category_filter and "category" in alerts.columns:
        alerts = alerts[alerts["category"].isin(category_filter)]
    if "confidence" in alerts.columns:
        alerts = alerts[alerts["confidence"] >= min_conf / 100]
else:
    alerts = all_alerts

# ── Page header ───────────────────────────────────────────────
st.title("🔍 SOC Investigation Console")
st.caption(f"Log: `{log_path}` · Last updated: {datetime.now().strftime('%H:%M:%S')}")
st.divider()

if all_alerts.empty:
    st.warning("No alerts found yet. Start Phase 4 detector first.")
else:
    # ── Summary metrics ───────────────────────────────────────
    total       = len(all_alerts)
    attacks     = len(all_alerts[all_alerts["category"] != "Normal"]) if "category" in all_alerts.columns else 0
    normal      = total - attacks
    attack_rate = (attacks / total * 100) if total > 0 else 0
    top_cat     = all_alerts["category"].value_counts().idxmax() if "category" in all_alerts.columns else "N/A"
    avg_conf    = all_alerts["confidence"].mean() * 100 if "confidence" in all_alerts.columns else 0
    top_sev_label, top_sev_color = SEVERITY.get(top_cat, ("LOW", "#6C757D"))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total alerts", f"{total:,}")
    c2.metric("Attacks",      f"{attacks:,}")
    c3.metric("Normal",       f"{normal:,}")
    c4.metric("Attack rate",  f"{attack_rate:.1f}%")
    c5.metric("Top threat",   f"{top_cat} ({top_sev_label})")

    st.divider()

    # ── Severity badge ────────────────────────────────────────
    st.markdown(
        f"<div style='background:{top_sev_color};color:white;padding:0.6rem 1.2rem;"
        f"border-radius:8px;display:inline-block;font-weight:700;font-size:1rem;margin-bottom:1rem'>"
        f"🚨 Current Threat Level: {top_sev_label} — Primary Category: {top_cat}</div>",
        unsafe_allow_html=True
    )

    # ── Charts ────────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("Attack Timeline")
        if "timestamp" in alerts.columns and alerts["timestamp"].notna().sum() > 1:
            tl = alerts.copy()
            tl["minute"] = tl["timestamp"].dt.floor("min")
            grouped = tl.groupby(["minute", "category"]).size().reset_index(name="count")
            fig = px.area(grouped, x="minute", y="count", color="category",
                          color_discrete_map=ATTACK_COLORS)
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for timeline.")

    with right:
        st.subheader("Top Source IPs")
        if "src_ip" in alerts.columns:
            top_ips = (alerts[alerts["category"] != "Normal"]["src_ip"]
                       .value_counts().head(10).reset_index())
            top_ips.columns = ["src_ip", "count"]
            fig_ip = px.bar(top_ips, x="count", y="src_ip", orientation="h",
                            color_discrete_sequence=["#D7263D"])
            fig_ip.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                                 yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_ip, use_container_width=True)
        else:
            st.info("No IP data available.")

    # ── Category breakdown table ──────────────────────────────
    st.subheader("Category Breakdown")
    if "category" in all_alerts.columns:
        cat_table = (all_alerts.groupby("category")
                     .agg(count=("category", "count"),
                          avg_conf=("confidence", lambda x: x.mean() * 100))
                     .reset_index().sort_values("count", ascending=False))
        cat_table["severity"] = cat_table["category"].map(lambda c: SEVERITY.get(c, ("LOW", ""))[0])
        cat_table["avg_conf"] = cat_table["avg_conf"].round(1)
        st.dataframe(cat_table, use_container_width=True, hide_index=True)

    # ── Alert detail table ────────────────────────────────────
    st.subheader("🔎 Alert Investigation Table")
    if not alerts.empty:
        show_cols = [c for c in ["timestamp", "src_ip", "src_port", "dst_ip", "dst_port",
                                  "protocol", "service", "category", "confidence"]
                     if c in alerts.columns]
        display = alerts.sort_values("timestamp", ascending=False).copy()
        if "confidence" in display.columns:
            display["confidence"] = (display["confidence"] * 100).round(1).astype(str) + "%"
        st.dataframe(display[show_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No alerts match current filters.")

    # ── PDF export ────────────────────────────────────────────
    st.divider()
    st.subheader("📄 Export Incident Report")

    if "category" in all_alerts.columns:
        cat_table_pdf = (all_alerts.groupby("category")
                         .agg(count=("category", "count"),
                              avg_conf=("confidence", lambda x: x.mean() * 100))
                         .reset_index())
    else:
        cat_table_pdf = pd.DataFrame()

    ts_start = all_alerts["timestamp"].min() if "timestamp" in all_alerts.columns else None
    ts_end   = all_alerts["timestamp"].max() if "timestamp" in all_alerts.columns else None

    summary = {
        "log_path":    log_path,
        "total":       total,
        "attacks":     attacks,
        "normal":      normal,
        "attack_rate": attack_rate,
        "top_cat":     top_cat,
        "avg_conf":    avg_conf,
        "cat_table":   cat_table_pdf,
        "start":       ts_start.strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(ts_start) else "N/A",
        "end":         ts_end.strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(ts_end) else "N/A",
    }

    col_btn, col_info = st.columns([1, 3])

    with col_btn:
        if st.button("📥 Generate PDF Report", type="primary"):
            try:
                pdf_bytes = make_pdf(all_alerts, summary)
                if pdf_bytes:
                    fname = f"NIDS_Incident_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name=fname,
                        mime="application/pdf"
                    )
                    st.success("PDF ready — click Download!")
                else:
                    st.error("fpdf2 not installed. Run: pip install fpdf2")
            except Exception as e:
                st.error(f"PDF error: {e}")

    with col_info:
        st.info(
            f"Report will include:\n"
            f"- Executive summary ({total} alerts, {attack_rate:.1f}% attack rate)\n"
            f"- Category breakdown table\n"
            f"- Recent 25 attack alerts\n"
            f"- Recommended SOC actions for {top_cat}"
        )

# ── Auto-refresh — MUST be last ───────────────────────────────
time.sleep(refresh_sec)
st.rerun()