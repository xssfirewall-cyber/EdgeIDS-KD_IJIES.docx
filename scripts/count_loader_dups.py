"""Counts the float32 duplicates that the N-BaIoT and CICIoT2023 loaders remove
BEFORE capping, i.e. before the counts in data_cache/<d>_report.json are taken.
Read-only: replays the loaders' sampling with the same seed and writes
data_cache/<d>_loader_dups.json; the cached splits are not touched.

    python scripts/count_loader_dups.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.data.loaders import _NBAIOT_FILE  # noqa: E402


def nbaiot():
    raw, unique = 0, 0
    for path in sorted(config.NBAIOT_DIR.glob("*.csv")):
        if not _NBAIOT_FILE.match(path.name):
            continue
        part = pd.read_csv(path, dtype=np.float32)
        raw += len(part)
        unique += len(part.drop_duplicates())
    return {"raw_rows_all_files": raw, "within_file_float32_duplicates": raw - unique,
            "scope": "all 89 device/class files, before the 10k-per-file cap"}


def ciciot2023():
    import pyarrow.csv as pc
    files = sorted(config.CICIOT2023_DIR.glob("*.csv"))
    counts: dict[str, int] = {}
    for f in files:
        col = pc.read_csv(f, convert_options=pc.ConvertOptions(include_columns=["label"]))
        for vc in col.column("label").value_counts().to_pylist():
            counts[vc["values"]] = counts.get(vc["values"], 0) + vc["counts"]
    cap = config.CICIOT2023_MAX_PER_CLASS
    keep_p = {k: min(1.0, 1.5 * cap / v) for k, v in counts.items()}
    rng = np.random.default_rng(0)                     # same seed and order as the loader
    frames = []
    for f in files:
        # pyarrow + immediate float32 cast keeps memory low; the values are
        # identical to the loader's (read as float64, then cast to float32).
        part = pc.read_csv(f).to_pandas()
        keep = rng.random(len(part)) < part["label"].map(keep_p).to_numpy()
        part = part[keep]
        feats = [c for c in part.columns if c != "label"]
        part[feats] = part[feats].astype(np.float32)
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    sampled = len(df)
    # One 64-bit hash per row instead of DataFrame.duplicated(), which
    # factorizes all 47 columns and exceeds the memory available here.
    dup = int(pd.Series(pd.util.hash_pandas_object(df, index=False)).duplicated().sum())
    return {"raw_rows_all_files": int(sum(counts.values())), "sampled_rows": sampled,
            "within_sample_float32_duplicates": dup,
            "scope": "uniform per-class sample (1.5x cap) before the 50k-per-class cap"}


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["nbaiot", "ciciot2023"]
    for name, fn in (("nbaiot", nbaiot), ("ciciot2023", ciciot2023)):
        if name not in wanted:
            continue
        r = fn()
        (config.CACHE_DIR / f"{name}_loader_dups.json").write_text(json.dumps(r, indent=2))
        print(name, r)
