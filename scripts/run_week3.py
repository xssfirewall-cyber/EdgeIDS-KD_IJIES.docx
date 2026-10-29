"""Week 3 batch (E3): every student x objective x seed.

ce / wce go through scripts/train.py, la / kd / kd_wce / rap_kd through
scripts/distill.py. A dataset is skipped until both of its teachers have
finished, so every KD run on it uses the same (final) teacher choice.
Restartable: existing result JSONs are skipped.
"""
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts import distill  # noqa: E402
from scripts import train as train_script  # noqa: E402
from src import losses, models, train  # noqa: E402

SEEDS = (0, 1, 2)
METHODS = ("ce", "wce") + losses.METHODS


def teachers_ready(d: str) -> bool:
    return all((config.RESULTS_DIR / "runs" / d / f"{t}_ce_s0.json").exists() for t in models.TEACHERS)


def main():
    dev = torch.device("cuda")
    for d in train_script.DATASETS:
        if not teachers_ready(d):
            print(f"[{d}] teachers not finished - skipped for now", flush=True)
            continue
        data = train.load_split(d, dev)
        for s in models.STUDENTS:
            for m in METHODS:
                for seed in SEEDS:
                    if (config.RESULTS_DIR / "runs" / d / f"{s}_{m}_s{seed}.json").exists():
                        continue
                    try:
                        if m in ("ce", "wce"):
                            train_script.run(d, s, m, seed, quiet=True)
                        else:
                            distill.run(d, s, m, seed, data=data)
                    except Exception:
                        traceback.print_exc()
        del data
        distill._teacher_cache.clear()
        torch.cuda.empty_cache()
    print("WEEK3 DONE", flush=True)


if __name__ == "__main__":
    main()
