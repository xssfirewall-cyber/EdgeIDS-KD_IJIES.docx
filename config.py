"""Paths and global settings for EdgeIDS-KD.

Every dataset path defaults to data/raw/<dataset> inside the project and can
be overridden with an environment variable of the same name.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = Path(os.environ.get("RAW_DIR", ROOT / "data" / "raw"))


def _path(name, default):
    return Path(os.environ.get(name, default))


TON_IOT_CSV = _path("TON_IOT_CSV", RAW_DIR / "TON_IoT" / "train_test_network.csv")
UNSW_DIR = _path("UNSW_DIR", RAW_DIR / "UNSW-NB15")
EDGE_IIOT_CSV = _path("EDGE_IIOT_CSV", RAW_DIR / "Edge-IIoTset" / "ML-EdgeIIoT-dataset.csv")
NBAIOT_DIR = _path("NBAIOT_DIR", RAW_DIR / "N-BaIoT")
CICIOT2023_DIR = _path("CICIOT2023_DIR", RAW_DIR / "ciciot2023")

CACHE_DIR = ROOT / "data_cache"      # processed .npz splits
RESULTS_DIR = ROOT / "results"

SEEDS = [0, 1, 2, 3, 4]
SPLIT = {"train": 0.70, "val": 0.10, "test": 0.20}

# N-BaIoT has ~7M rows; cap rows per (device, class) file after de-duplication.
NBAIOT_MAX_PER_FILE = 10_000

# CICIoT2023 has 46.7M rows; cap each class after de-duplication. Classes
# smaller than the cap (all rare attacks) are kept whole.
CICIOT2023_MAX_PER_CLASS = 50_000
