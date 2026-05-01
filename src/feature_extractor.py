"""
=============================================================
 NIDS Project — Feature Extractor
 Converts raw Scapy packets → 41 NSL-KDD-style features
 per completed network connection/flow.
=============================================================
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ─── SERVICE PORT MAP ─────────────────────────────────────────
# Maps dst_port → NSL-KDD service name
PORT_SERVICE = {
    20: 'ftp_data', 21: 'ftp', 22: 'ssh', 23: 'telnet',
    25: 'smtp', 53: 'domain', 67: 'other', 68: 'other',
    69: 'tftp_u', 79: 'finger', 80: 'http', 110: 'pop_3',
    111: 'sunrpc', 119: 'nntp', 123: 'ntp_u', 135: 'other',
    137: 'netbios_ns', 138: 'netbios_dgm', 139: 'netbios_ssn',
    143: 'imap4', 161: 'snmp', 179: 'bgp', 194: 'IRC',
    389: 'ldap', 443: 'http_443', 445: 'netbios_ssn',
    465: 'smtp', 513: 'login', 514: 'shell', 515: 'printer',
    587: 'smtp', 631: 'other', 993: 'imap4', 995: 'pop_3',
    1080: 'other', 1433: 'sql_net', 1521: 'sql_net',
    3306: 'sql_net', 3389: 'http', 5432: 'sql_net',
    6667: 'IRC', 8080: 'http', 8443: 'http_443',
}

# TCP flag string → NSL-KDD flag label
TCP_FLAG_MAP = {
    'SF':   'SF',    # Normal: SYN + FIN both seen
    'S0':   'S0',    # SYN sent, no response
    'S1':   'S1',    # SYN+ACK seen
    'S2':   'S2',
    'S3':   'S3',
    'REJ':  'REJ',   # Connection rejected (RST)
    'RSTO': 'RSTO',  # RST from originator
    'RSTR': 'RSTR',  # RST from responder
    'SH':   'SH',    # SYN+FIN
    'OTH':  'OTH',   # Other/unknown
}

FEATURE_ORDER = [
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
    'dst_host_srv_rerror_rate'
]


# ─── CONNECTION RECORD ────────────────────────────────────────
@dataclass
class ConnectionRecord:
    src_ip:   str
    dst_ip:   str
    src_port: int
    dst_port: int
    protocol: str           # 'tcp' | 'udp' | 'icmp'
    service:  str = 'other'

    start_time:  float = field(default_factory=time.time)
    last_time:   float = field(default_factory=time.time)

    src_bytes:      int = 0
    dst_bytes:      int = 0
    wrong_fragment: int = 0
    urgent:         int = 0

    # TCP flag tracking
    syn_seen:  bool = False
    fin_seen:  bool = False
    rst_seen:  bool = False
    ack_seen:  bool = False
    syn_ack_seen: bool = False

    completed: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.last_time - self.start_time)

    @property
    def land(self) -> int:
        return int(self.src_ip == self.dst_ip and self.src_port == self.dst_port)

    @property
    def flag(self) -> str:
        if self.rst_seen:
            return 'RSTO' if self.syn_seen else 'REJ'
        if self.syn_seen and self.fin_seen:
            return 'SF'
        if self.syn_seen and self.syn_ack_seen and not self.fin_seen:
            return 'S1'
        if self.syn_seen and not self.syn_ack_seen:
            return 'S0'
        return 'OTH'

    def flow_key(self) -> Tuple:
        return (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol)


# ─── CONNECTION TRACKER ───────────────────────────────────────
class ConnectionTracker:
    TIMEOUT_SEC = 3  # Close idle connections after 30s

    def __init__(self):
        self.active: Dict[Tuple, ConnectionRecord] = {}
        # Sliding windows for statistical features
        self.completed_2s:  deque = deque()   # last 2 seconds
        self.completed_100: deque = deque(maxlen=100)  # last 100 connections

    def process_packet(self, pkt) -> Optional[ConnectionRecord]:
        """
        Feed a Scapy packet in. Returns a completed ConnectionRecord
        when a flow closes, else None.
        """
        try:
            from scapy.layers.inet import IP, TCP, UDP, ICMP
            from scapy.layers.inet6 import IPv6

            # Only handle IP packets
            if IP not in pkt and IPv6 not in pkt:
                return None

            ip_layer = pkt[IP] if IP in pkt else pkt[IPv6]
            src_ip   = ip_layer.src
            dst_ip   = ip_layer.dst

            # Determine protocol and ports
            if TCP in pkt:
                proto    = 'tcp'
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
                payload  = len(pkt[TCP].payload)
                service  = PORT_SERVICE.get(dst_port,
                           PORT_SERVICE.get(src_port, 'other'))
            elif UDP in pkt:
                proto    = 'udp'
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport
                payload  = len(pkt[UDP].payload)
                service  = PORT_SERVICE.get(dst_port,
                           PORT_SERVICE.get(src_port, 'other'))
            elif ICMP in pkt:
                proto    = 'icmp'
                src_port = 0
                dst_port = 0
                payload  = len(pkt[ICMP].payload)
                service  = 'eco_i'
            else:
                return None

            key = (src_ip, dst_ip, src_port, dst_port, proto)
            now = time.time()

            # Create or update connection record
            if key not in self.active:
                self.active[key] = ConnectionRecord(
                    src_ip=src_ip, dst_ip=dst_ip,
                    src_port=src_port, dst_port=dst_port,
                    protocol=proto, service=service,
                    start_time=now, last_time=now
                )

            conn = self.active[key]
            conn.last_time = now

            # Byte counting (forward = src→dst)
            conn.src_bytes += payload

            # Fragment tracking
            if IP in pkt and pkt[IP].frag > 0:
                conn.wrong_fragment += 1

            # TCP flag tracking
            if TCP in pkt:
                flags = pkt[TCP].flags
                if flags & 0x02:  conn.syn_seen     = True  # SYN
                if flags & 0x01:  conn.fin_seen     = True  # FIN
                if flags & 0x04:  conn.rst_seen     = True  # RST
                if flags & 0x10:  conn.ack_seen     = True  # ACK
                if flags & 0x20:  conn.urgent      += 1     # URG
                if flags & 0x12:  conn.syn_ack_seen = True  # SYN+ACK

            # Check if connection should be completed
            completed = None
            should_close = (
                (TCP in pkt and (pkt[TCP].flags & 0x01 or pkt[TCP].flags & 0x04))
                or proto == 'icmp'
                or (now - conn.start_time) > self.TIMEOUT_SEC
            )

            if should_close:
                conn.completed = True
                completed = conn
                del self.active[key]
                self._add_to_windows(completed)

            # Expire old active connections
            self._expire_old(now)

            return completed

        except Exception:
            return None

    def _add_to_windows(self, conn: ConnectionRecord):
        now = time.time()
        self.completed_2s.append((now, conn))
        self.completed_100.append(conn)
        # Clean up >2s old entries
        while self.completed_2s and (now - self.completed_2s[0][0]) > 2.0:
            self.completed_2s.popleft()

    def _expire_old(self, now: float):
        expired = [k for k, c in self.active.items()
                   if (now - c.last_time) > self.TIMEOUT_SEC]
        for k in expired:
            conn = self.active.pop(k)
            conn.completed = True
            self._add_to_windows(conn)

    def get_stats_2s(self, conn: ConnectionRecord) -> Dict:
        """Compute 2-second sliding window statistics."""
        window = [c for _, c in self.completed_2s]
        same_host = [c for c in window if c.dst_ip == conn.dst_ip]
        same_srv  = [c for c in window if c.service == conn.service]

        def serror(lst):
            if not lst: return 0.0
            return sum(1 for c in lst if c.flag in ('S0','S1','S2','S3')) / len(lst)

        def rerror(lst):
            if not lst: return 0.0
            return sum(1 for c in lst if c.flag in ('REJ',)) / len(lst)

        n  = len(same_host) or 1
        ns = len(same_srv)  or 1

        return {
            'count':           len(same_host),
            'srv_count':       len(same_srv),
            'serror_rate':     serror(same_host),
            'srv_serror_rate': serror(same_srv),
            'rerror_rate':     rerror(same_host),
            'srv_rerror_rate': rerror(same_srv),
            'same_srv_rate':   sum(1 for c in same_host
                                   if c.service == conn.service) / n,
            'diff_srv_rate':   sum(1 for c in same_host
                                   if c.service != conn.service) / n,
            'srv_diff_host_rate': sum(1 for c in same_srv
                                      if c.dst_ip != conn.dst_ip) / ns,
        }

    def get_stats_100(self, conn: ConnectionRecord) -> Dict:
        """Compute last-100-connection statistics."""
        window    = list(self.completed_100)
        same_host = [c for c in window if c.dst_ip == conn.dst_ip]
        same_srv  = [c for c in same_host if c.service == conn.service]

        def serror(lst):
            if not lst: return 0.0
            return sum(1 for c in lst if c.flag in ('S0','S1','S2','S3')) / len(lst)

        def rerror(lst):
            if not lst: return 0.0
            return sum(1 for c in lst if c.flag in ('REJ',)) / len(lst)

        n  = len(same_host) or 1

        return {
            'dst_host_count':            len(same_host),
            'dst_host_srv_count':        len(same_srv),
            'dst_host_same_srv_rate':    len(same_srv) / n,
            'dst_host_diff_srv_rate':    sum(1 for c in same_host
                                            if c.service != conn.service) / n,
            'dst_host_same_src_port_rate': sum(1 for c in same_host
                                               if c.src_port == conn.src_port) / n,
            'dst_host_srv_diff_host_rate': sum(1 for c in same_srv
                                               if c.src_ip != conn.src_ip) / (len(same_srv) or 1),
            'dst_host_serror_rate':      serror(same_host),
            'dst_host_srv_serror_rate':  serror(same_srv),
            'dst_host_rerror_rate':      rerror(same_host),
            'dst_host_srv_rerror_rate':  rerror(same_srv),
        }


# ─── FEATURE EXTRACTOR ────────────────────────────────────────
class FeatureExtractor:
    def __init__(self, tracker: ConnectionTracker):
        self.tracker = tracker

    def extract(self, conn: ConnectionRecord) -> Dict:
        stats2s  = self.tracker.get_stats_2s(conn)
        stats100 = self.tracker.get_stats_100(conn)

        features = {
            # Basic
            'duration':          round(conn.duration, 3),
            'protocol_type':     conn.protocol,
            'service':           conn.service,
            'flag':              conn.flag,
            'src_bytes':         conn.src_bytes,
            'dst_bytes':         conn.dst_bytes,
            'land':              conn.land,
            'wrong_fragment':    conn.wrong_fragment,
            'urgent':            conn.urgent,
            # Content features (not extractable without DPI → set 0)
            'hot':               0,
            'num_failed_logins': 0,
            'logged_in':         0,
            'num_compromised':   0,
            'root_shell':        0,
            'su_attempted':      0,
            'num_root':          0,
            'num_file_creations':0,
            'num_shells':        0,
            'num_access_files':  0,
            'num_outbound_cmds': 0,
            'is_host_login':     0,
            'is_guest_login':    0,
        }
        features.update(stats2s)
        features.update(stats100)
        return features

    def to_vector(self, conn: ConnectionRecord) -> List:
        f = self.extract(conn)
        return [f[k] for k in FEATURE_ORDER]
