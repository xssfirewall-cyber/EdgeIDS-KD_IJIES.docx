"""Builds leakage-checked, preprocessed splits for each dataset.

    python scripts/prepare_data.py                  # all datasets
    python scripts/prepare_data.py --dataset ton_iot

Writes data_cache/<dataset>.npz and data_cache/<dataset>_report.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.data import labels, loaders, preprocess, splits  # noqa: E402


def prepare(name: str) -> dict:
    t0 = time.time()
    df, num, cat = loaders.LOADERS[name]()
    feats = num + cat
    df, dedup = splits.deduplicate(df, feats)
    parts = splits.stratified_split(df, seed=0)
    checks = splits.leakage_checks(parts, feats, len(labels.CLASSES[name]))
    X, feat_names = preprocess.fit_transform(parts, num, cat)
    # Distinct raw rows can still map to identical model inputs after the
    # transform; report how many test inputs the model has literally seen.
    seen = {r.tobytes() for r in X["train"]}
    checks["test_inputs_identical_to_train"] = int(sum(r.tobytes() in seen for r in X["test"]))

    config.CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(
        config.CACHE_DIR / f"{name}.npz",
        **{f"X_{k}": X[k] for k in X},
        **{f"y_{k}": parts[k]["y"].to_numpy(np.int64) for k in parts},
        **{f"g_{k}": parts[k]["group"].to_numpy(np.int64) for k in parts},
        classes=np.array(labels.CLASSES[name]),
        features=np.array(feat_names),
    )
    report = {"dataset": name, "n_features": len(feat_names),
              "classes": labels.CLASSES[name], **dedup, **checks,
              "preprocessor_fit_on": "train",
              "seconds": round(time.time() - t0, 1)}
    (config.CACHE_DIR / f"{name}_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(loaders.LOADERS), nargs="*")
    names = ap.parse_args().dataset or list(loaders.LOADERS)
    for n in names:
        r = prepare(n)
        train = r["class_counts"]["train"]
        print(f"[{n}] raw={r['raw_rows']:,} dups={r['exact_duplicates_removed']:,} "
              f"conflicts={r['conflicting_label_rows_removed']:,} kept={r['rows_after']:,} "
              f"features={r['n_features']} overlap={r['overlap']} "
              f"test_seen_in_train={r['test_inputs_identical_to_train']} ({r['seconds']}s)")
        for c, k in zip(r["classes"], train):
            print(f"    {c:<16} train={k:>7,}")


if __name__ == "__main__":
    main()
