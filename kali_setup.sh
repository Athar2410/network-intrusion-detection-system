#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  NIDS Phase 4 — Kali Linux Setup Script
#  Run with: bash kali_setup.sh
# ─────────────────────────────────────────────────────────────

echo "========================================"
echo "  NIDS Phase 4 — Kali Setup"
echo "========================================"

# Update and install system dependencies
sudo apt update -y
sudo apt install -y python3-pip python3-venv python3-scapy \
                    nmap hping3 tcpdump net-tools

# Install Python packages
pip3 install scapy joblib numpy scikit-learn

echo ""
echo "[✓] Setup complete!"
echo ""
echo "Usage:"
echo "  1. Find your network interface:"
echo "     ip a"
echo ""
echo "  2. Run the detector:"
echo "     sudo python3 src/04_realtime_detector.py --iface eth0"
echo ""
echo "  3. Run with JSON alert logging:"
echo "     sudo python3 src/04_realtime_detector.py --iface eth0 --log /tmp/alerts.json"
echo ""
echo "  4. Test with attack simulation (separate terminal):"
echo "     # SYN Flood (DoS)"
echo "     sudo hping3 -S --flood -V -p 80 <target_ip>"
echo ""
echo "     # Port Scan (Probe)"
echo "     nmap -sS -T4 <target_ip>"
echo ""
echo "     # ICMP Flood"
echo "     sudo hping3 --icmp --flood <target_ip>"
