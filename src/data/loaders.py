"""Raw dataset loaders.

Each loader returns (df, numeric_cols, categorical_cols) where df holds the
feature columns plus ``y`` (canonical class index) and ``group`` (device id for
N-BaIoT, 0 elsewhere). Identifier columns are dropped here, before any
de-duplication, because removing them creates new duplicate rows.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import config
from src.data import labels

# ---------------------------------------------------------------- TON_IoT
_TON_DROP = [
    "src_ip", "dst_ip", "src_port", "dst_port",          # host identifiers
    "dns_query", "ssl_subject", "ssl_issuer",            # free text
    "http_uri", "http_user_agent",
    "label",                                             # binary copy of target
]
_TON_CAT = [
    "proto", "service", "conn_state",
    "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
    "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established",
    "http_method", "http_version", "http_orig_mime_types", "http_resp_mime_types",
    "weird_name", "weird_addl", "weird_notice",
]


def load_ton_iot():
    df = pd.read_csv(config.TON_IOT_CSV, low_memory=False)
    df = df.drop(columns=_TON_DROP)
    y = labels.to_index("ton_iot", df.pop("type"))
    num = [c for c in df.columns if c not in _TON_CAT]
    for c in num:
        df[c] = pd.to_numeric(df[c].replace("-", np.nan), errors="coerce")
    for c in _TON_CAT:
        df[c] = df[c].astype(str)
    df["y"] = y
    df["group"] = 0
    return df, num, list(_TON_CAT)


# ------------------------------------------------------------- UNSW-NB15
_UNSW_CAT = ["proto", "service", "state"]


def load_unsw_nb15():
    # The official "training" and "testing" files overlap (identical feature
    # rows appear in both), so they are merged and re-split after de-duplication.
    parts = [pd.read_csv(config.UNSW_DIR / f)
             for f in ("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv")]
    df = pd.concat(parts, ignore_index=True).drop(columns=["id", "label"])
    y = labels.to_index("unsw_nb15", df.pop("attack_cat"))
    num = [c for c in df.columns if c not in _UNSW_CAT]
    for c in _UNSW_CAT:
        df[c] = df[c].astype(str)
    df["y"] = y
    df["group"] = 0
    return df, num, list(_UNSW_CAT)


# --------------------------------------------------------------- N-BaIoT
_NBAIOT_FILE = re.compile(r"^(\d)\.(benign|(?:gafgyt|mirai)\.\w+)\.csv$")


def load_nbaiot(max_per_file: int = config.NBAIOT_MAX_PER_FILE, seed: int = 0):
    """Loads every <device>.<class>.csv, de-duplicating and capping each file
    before concatenation so memory stays bounded."""
    frames = []
    for path in sorted(config.NBAIOT_DIR.glob("*.csv")):
        m = _NBAIOT_FILE.match(path.name)
        if not m:
            continue                      # data_summary.csv, features.csv, ...
        dev, cls = int(m.group(1)), m.group(2)
        # De-duplicate at float32, the precision the model sees: gafgyt.tcp/udp
        # floods yield ~85k rows differing only past float32 precision, which
        # collapse to ~100 distinct model inputs per device.
        part = pd.read_csv(path, dtype=np.float32).drop_duplicates()
        if len(part) > max_per_file:
            part = part.sample(max_per_file, random_state=seed)
        part["y"] = labels.to_index("nbaiot", [cls])[0]
        part["group"] = dev
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    num = [c for c in df.columns if c not in ("y", "group")]
    return df, num, []


# ------------------------------------------------------------ CICIoT2023
def load_ciciot2023(max_per_class: int = config.CICIOT2023_MAX_PER_CLASS, seed: int = 0):
    """Two passes over the 169 part files, so the 13 GB never sits in memory.

    Pass 1 counts labels. Pass 2 keeps each row of class c with probability
    min(1, 1.5 * cap / count_c), so the sample is spread uniformly over all
    files; the 1.5x margin absorbs float32 duplicates removed afterwards,
    before the final cap. Rare classes (count <= cap) are kept whole."""
    import pyarrow.csv as pc

    files = sorted(config.CICIOT2023_DIR.glob("*.csv"))
    counts: dict[str, int] = {}
    for f in files:
        col = pc.read_csv(f, convert_options=pc.ConvertOptions(include_columns=["label"]))
        for vc in col.column("label").value_counts().to_pylist():
            counts[vc["values"]] = counts.get(vc["values"], 0) + vc["counts"]
    keep_p = {k: min(1.0, 1.5 * max_per_class / v) for k, v in counts.items()}

    rng = np.random.default_rng(seed)
    frames = []
    for f in files:
        part = pd.read_csv(f)
        p = part["label"].map(keep_p).to_numpy()
        frames.append(part[rng.random(len(part)) < p])
    df = pd.concat(frames, ignore_index=True)

    feats = [c for c in df.columns if c != "label"]
    df[feats] = df[feats].astype(np.float32)
    df = df.drop_duplicates()     # float32, as for N-BaIoT
    df = (df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max_per_class), random_state=seed)))
    df = df.reset_index(drop=True)
    df["y"] = labels.to_index("ciciot2023", df.pop("label"))
    df["group"] = 0
    return df, feats, []


# ----------------------------------------------------------- Edge-IIoTset
_EDGE_DROP = [
    # identifiers and timestamps (dataset authors' list)
    "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
    "tcp.srcport", "tcp.dstport", "udp.port", "icmp.transmit_timestamp",
    # free-text payloads (dataset authors' list)
    "http.file_data", "http.request.full_uri", "http.request.uri.query",
    "tcp.options", "tcp.payload", "mqtt.msg",
    # per-packet / per-session identifiers: they name the capture, not the attack
    "tcp.checksum", "icmp.checksum", "tcp.ack_raw", "udp.stream", "icmp.seq_le",
    "Attack_label",                                     # binary copy of target
]
_EDGE_CAT = ["http.request.method", "http.referer", "http.request.version",
             "dns.qry.name.len", "mqtt.conack.flags", "mqtt.protoname", "mqtt.topic"]


def load_edge_iiot():
    df = pd.read_csv(config.EDGE_IIOT_CSV, low_memory=False).drop(columns=_EDGE_DROP)
    excluded = df["Attack_type"].map(labels.canonical).isin(labels.EDGE_IIOT_EXCLUDED)
    df = df[~excluded].reset_index(drop=True)
    y = labels.to_index("edge_iiot", df.pop("Attack_type"))
    num = [c for c in df.columns if c not in _EDGE_CAT]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in _EDGE_CAT:
        # "0" and "0.0" both mean "field absent"
        df[c] = df[c].astype(str).str.strip().replace({"0.0": "0"})
    num = [c for c in num if df[c].nunique(dropna=False) > 1]   # drop constant columns
    df = df[num + _EDGE_CAT].copy()
    df["y"] = y
    df["group"] = 0
    return df, num, list(_EDGE_CAT)


LOADERS = {"ton_iot": load_ton_iot, "unsw_nb15": load_unsw_nb15, "nbaiot": load_nbaiot,
           "ciciot2023": load_ciciot2023, "edge_iiot": load_edge_iiot}
