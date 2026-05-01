"""
=============================================================
 NIDS Project — Phase 4: Real-Time Network Intrusion Detector
 Platform : Kali Linux (run as root for packet capture)
 Requires : scapy, joblib, numpy, scikit-learn
            pip install scapy joblib numpy scikit-learn

 Usage:
   sudo python3 04_realtime_detector.py
   sudo python3 04_realtime_detector.py --iface eth0
   sudo python3 04_realtime_detector.py --iface eth0 --log alerts.json
=============================================================
"""

import os, sys, time, json, argparse, signal
from collections import deque
from datetime import datetime

import numpy as np
import joblib

# Add src/ to path so feature_extractor can be imported
sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import (
    ConnectionTracker, FeatureExtractor, FEATURE_ORDER
)

# ─── PATHS (relative to this script's location → src/) ────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ─── ANSI COLORS ──────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ─── ATTACK CATEGORY LABELS (multiclass model) ────────────────
ATTACK_COLORS = {
    'Normal': GREEN,
    'DoS':    RED,
    'Probe':  YELLOW,
    'R2L':    "\033[95m",
    'U2R':    "\033[31m",
    'Unknown':"\033[90m",
}

# ─── MODEL LOADER ─────────────────────────────────────────────
class ModelLoader:
    def __init__(self):
        print(f"{CYAN}[→] Loading models and preprocessors...{RESET}")

        # Binary classifier (Normal vs Attack)
        self.binary_model = joblib.load(
            os.path.join(MODELS_DIR, "random_forest_binary.pkl"))

        # Multi-class classifier (attack type)
        self.multiclass_model = joblib.load(
            os.path.join(MODELS_DIR, "rf_multiclass.pkl"))

        # Scaler (MinMaxScaler from Phase 1)
        self.scaler = joblib.load(
            os.path.join(OUTPUT_DIR, "scaler.pkl"))

        # Categorical label encoders (protocol_type, service, flag)
        self.le_dict = joblib.load(
            os.path.join(OUTPUT_DIR, "label_encoders.pkl"))

        # Category label encoder (Normal/DoS/Probe/R2L/U2R)
        self.cat_le = joblib.load(
            os.path.join(OUTPUT_DIR, "category_label_encoder.pkl"))

        print(f"{GREEN}[✓] Models loaded successfully{RESET}")
        print(f"    Binary model  : {type(self.binary_model).__name__}")
        print(f"    Multi-class   : {type(self.multiclass_model).__name__}")
        print(f"    Categories    : {list(self.cat_le.classes_)}\n")

    def preprocess(self, raw_features: list) -> np.ndarray:
        """
        Apply the same encoding + scaling from Phase 1 preprocessing.
        raw_features: list of values in FEATURE_ORDER
        """
        feat_dict = dict(zip(FEATURE_ORDER, raw_features))

        # Encode categorical features using saved LabelEncoders
        for col in ['protocol_type', 'service', 'flag']:
            le = self.le_dict[col]
            val = feat_dict[col]
            # Handle unseen labels gracefully
            if val in le.classes_:
                feat_dict[col] = le.transform([val])[0]
            else:
                feat_dict[col] = 0  # fallback for unknown

        # Rebuild vector in correct order
        vec = np.array([feat_dict[k] for k in FEATURE_ORDER],
                       dtype=np.float32).reshape(1, -1)

        # Apply MinMax scaling
        vec = self.scaler.transform(vec)
        return vec

    def predict(self, raw_features: list):
        """
        Returns: (is_attack: bool, category: str, confidence: float)
        """
        vec = self.preprocess(raw_features)

        # Binary prediction with probability
        prob      = self.binary_model.predict_proba(vec)[0][1]
        is_attack = prob >= 0.35

        if is_attack:
            cat_enc  = self.multiclass_model.predict(vec)[0]
            category = self.cat_le.inverse_transform([cat_enc])[0]
        else:
            category = 'Normal'

        return is_attack, category, round(float(prob), 4)


# ─── ALERT LOGGER ─────────────────────────────────────────────
class AlertLogger:
    def __init__(self, log_path: str = None):
        self.log_path  = log_path
        self.total     = 0
        self.attacks   = 0
        self.cat_counts = {}
        self.recent    = deque(maxlen=10)  # last 10 alerts for summary

    def log(self, conn, category: str, confidence: float, is_attack: bool):
        self.total += 1
        if is_attack:
            self.attacks += 1
        self.cat_counts[category] = self.cat_counts.get(category, 0) + 1

        ts    = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        color = ATTACK_COLORS.get(category, RESET)

        # ── Console output ─────────────────────────────────────
        if is_attack:
            print(
                f"{color}{BOLD}[ALERT]{RESET} {ts} | "
                f"{conn.src_ip}:{conn.src_port} → "
                f"{conn.dst_ip}:{conn.dst_port} | "
                f"Proto:{conn.protocol.upper()} | "
                f"Svc:{conn.service} | "
                f"{color}{BOLD}{category}{RESET} | "
                f"Conf:{confidence*100:.1f}% | "
                f"Bytes:{conn.src_bytes}↑"
            )
        else:
            if self.total % 20 == 0:   # Print Normal every 20 connections
                print(
                    f"{GREEN}[ OK  ]{RESET} {ts} | "
                    f"{conn.src_ip}:{conn.src_port} → "
                    f"{conn.dst_ip}:{conn.dst_port} | "
                    f"Normal | Conf:{(1-confidence)*100:.1f}%"
                )

        # ── JSON log file ──────────────────────────────────────
        if self.log_path and is_attack:
            entry = {
                "timestamp":  datetime.now().isoformat(),
                "src_ip":     conn.src_ip,
                "dst_ip":     conn.dst_ip,
                "src_port":   conn.src_port,
                "dst_port":   conn.dst_port,
                "protocol":   conn.protocol,
                "service":    conn.service,
                "flag":       conn.flag,
                "category":   category,
                "confidence": confidence,
                "src_bytes":  conn.src_bytes,
                "duration":   round(conn.duration, 3),
            }
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(entry) + "\n")

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"{BOLD}  DETECTION SUMMARY{RESET}")
        print(f"{'='*60}")
        print(f"  Total connections analysed : {self.total}")
        print(f"  Attacks detected           : {RED}{self.attacks}{RESET}")
        print(f"  Normal traffic             : {GREEN}{self.total - self.attacks}{RESET}")
        if self.total:
            print(f"  Attack rate                : {self.attacks/self.total*100:.1f}%")
        print(f"\n  Breakdown by category:")
        for cat, cnt in sorted(self.cat_counts.items(),
                                key=lambda x: -x[1]):
            color = ATTACK_COLORS.get(cat, RESET)
            bar   = '█' * min(cnt, 40)
            print(f"    {color}{cat:<12}{RESET}  {bar} {cnt}")
        print(f"{'='*60}")


# ─── MAIN DETECTOR ────────────────────────────────────────────
def run_detector(iface: str, log_path: str, packet_count: int):
    try:
        from scapy.all import sniff, conf
        conf.verb = 0
    except ImportError:
        print("[✗] Scapy not found. Run: pip install scapy")
        sys.exit(1)

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  🛡️  NIDS — Real-Time Intrusion Detector{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"  Interface : {CYAN}{iface}{RESET}")
    print(f"  Log file  : {log_path if log_path else 'disabled'}")
    print(f"  Press Ctrl+C to stop and view summary\n")

    model   = ModelLoader()
    tracker = ConnectionTracker()
    extractor = FeatureExtractor(tracker)
    logger  = AlertLogger(log_path)

    def handle_packet(pkt):
        # Force-flush connections older than TIMEOUT_SEC every 100 packets
        if handle_packet.count % 100 == 0:
            now = time.time()
            expired = [k for k, c in tracker.active.items()
                   if (now - c.last_time) > 3]
            for k in expired:
                conn = tracker.active.pop(k)
                conn.completed = True
                tracker._add_to_windows(conn)
                try:
                    raw_features = extractor.to_vector(conn)
                    is_attack, category, confidence = model.predict(raw_features)
                    logger.log(conn, category, confidence, is_attack)
                except Exception:
                    pass
        handle_packet.count += 1

        conn = tracker.process_packet(pkt)
        if conn is None:
            return
        try:
            raw_features = extractor.to_vector(conn)
            is_attack, category, confidence = model.predict(raw_features)
            logger.log(conn, category, confidence, is_attack)
        except Exception:
            pass

    handle_packet.count = 0   # ← Add this line RIGHT AFTER the function definition

    def graceful_exit(sig, frame):
        print(f"\n{YELLOW}[!] Stopping detector...{RESET}")
        logger.print_summary()
        sys.exit(0)

    signal.signal(signal.SIGINT,  graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    print(f"{GREEN}[✓] Sniffing on {iface} — waiting for traffic...{RESET}\n")
    sniff(
        iface=iface,
        prn=handle_packet,
        store=False,
        count=packet_count,  # 0 = infinite
        filter="ip"          # Only IP traffic
    )

    logger.print_summary()


# ─── ARGUMENT PARSER ──────────────────────────────────────────
if __name__ == "__main__":
    if os.geteuid() != 0:
        print(f"{RED}[✗] Must run as root: sudo python3 04_realtime_detector.py{RESET}")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="NIDS Real-Time Detector — Phase 4")
    parser.add_argument(
        "--iface", default="eth0",
        help="Network interface to sniff on (default: eth0). "
             "Use 'ip a' to find yours.")
    parser.add_argument(
        "--log", default=None,
        help="Path to save JSON alert log (optional). "
             "Example: --log /tmp/nids_alerts.json")
    parser.add_argument(
        "--count", type=int, default=0,
        help="Number of packets to capture (0 = infinite)")

    args = parser.parse_args()
    run_detector(args.iface, args.log, args.count)
