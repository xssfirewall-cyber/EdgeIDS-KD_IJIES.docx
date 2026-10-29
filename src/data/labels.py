"""Canonical label maps. Every loader and every metric must go through this module,
so a class index means the same thing everywhere."""

TON_IOT = ["normal", "backdoor", "ddos", "dos", "injection",
           "password", "scanning", "ransomware", "xss", "mitm"]

UNSW_NB15 = ["normal", "generic", "exploits", "fuzzers", "dos", "reconnaissance",
             "analysis", "backdoor", "shellcode", "worms"]

NBAIOT = ["benign",
          "gafgyt.combo", "gafgyt.junk", "gafgyt.scan", "gafgyt.tcp", "gafgyt.udp",
          "mirai.ack", "mirai.scan", "mirai.syn", "mirai.udp", "mirai.udpplain"]

CICIOT2023 = [
    "benigntraffic",
    # DDoS
    "ddos-icmp_flood", "ddos-udp_flood", "ddos-tcp_flood", "ddos-pshack_flood",
    "ddos-syn_flood", "ddos-rstfinflood", "ddos-synonymousip_flood",
    "ddos-icmp_fragmentation", "ddos-udp_fragmentation", "ddos-ack_fragmentation",
    "ddos-http_flood", "ddos-slowloris",
    # DoS
    "dos-udp_flood", "dos-tcp_flood", "dos-syn_flood", "dos-http_flood",
    # Mirai
    "mirai-greeth_flood", "mirai-udpplain", "mirai-greip_flood",
    # Recon
    "recon-hostdiscovery", "recon-osscan", "recon-portscan", "recon-pingsweep",
    "vulnerabilityscan",
    # Spoofing
    "mitm-arpspoofing", "dns_spoofing",
    # Web / brute force
    "dictionarybruteforce", "browserhijacking", "commandinjection", "sqlinjection",
    "xss", "backdoor_malware", "uploading_attack",
]

EDGE_IIOT = ["normal", "ddos_udp", "ddos_http", "ddos_tcp",
             "ransomware", "sql_injection", "uploading", "backdoor",
             "vulnerability_scanner", "port_scanning", "xss", "password", "mitm"]
# Without identifier columns, ddos_icmp (14,090 rows) collapses to 1 distinct
# packet and fingerprinting (1,001) to 5, all shared with other labels: a
# single ICMP packet cannot tell a flood from a ping. Excluded, not relabelled.
EDGE_IIOT_EXCLUDED = ["ddos_icmp", "fingerprinting"]

CLASSES = {"ton_iot": TON_IOT, "unsw_nb15": UNSW_NB15, "nbaiot": NBAIOT,
           "ciciot2023": CICIOT2023, "edge_iiot": EDGE_IIOT}

# Raw spellings that differ from the canonical name.
_ALIASES = {
    "backdoors": "backdoor",
    "dos ": "dos",
}


def canonical(name: str) -> str:
    n = str(name).strip().lower()
    return _ALIASES.get(n, n)


def to_index(dataset: str, names):
    """Map raw label strings to class indices; raises on anything unknown
    instead of silently dropping rows."""
    lookup = {c: i for i, c in enumerate(CLASSES[dataset])}
    out, unknown = [], set()
    for n in names:
        c = canonical(n)
        if c not in lookup:
            unknown.add(n)
        out.append(lookup.get(c, -1))
    if unknown:
        raise ValueError(f"{dataset}: unknown labels {sorted(unknown)}")
    return out
