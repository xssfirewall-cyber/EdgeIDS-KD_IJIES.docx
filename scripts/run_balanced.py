"""Balanced-teacher experiment.

1. For each dataset, retrain the architecture of the selected standard
   teacher with logit adjustment ("la") -> the balanced teacher.
2. Distil into 3 student sizes with kd, kd_wce and bt_kd from that teacher
   (results tagged kd_bt, kd_wce_bt, bt_kd), 3 seeds each.
Same T/alpha as the main sweep (no new tuning). Restartable; exits on a CUDA
error so run_retry.ps1 can restart it.
"""
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts import distill  # noqa: E402
from scripts import train as train_script  # noqa: E402
from src import train  # noqa: E402

SEEDS = (0, 1, 2)
STUDENTS = ("mlp_h16", "mlp_h64", "mlp_h256")
METHODS = ("kd", "kd_wce", "bt_kd")


def guarded(fn, *args, **kw):
    try:
        fn(*args, **kw)
    except Exception as e:
        traceback.print_exc()
        if "CUDA" in str(e):
            print("CUDA ERROR - exiting for restart", flush=True)
            sys.exit(3)


def main():
    runs = config.RESULTS_DIR / "runs"
    for d in train_script.DATASETS:
        arch = distill.pick_teacher(d, "ce")["model"]
        if not (runs / d / f"{arch}_la_s0.json").exists():
            guarded(train_script.run, d, arch, "la", 0, quiet=True)
        data = train.load_split(d, torch.device("cuda"))
        for s in STUDENTS:
            for m in METHODS:
                label = m if m == "bt_kd" else f"{m}_bt"
                for seed in SEEDS:
                    if not (runs / d / f"{s}_{label}_s{seed}.json").exists():
                        guarded(distill.run, d, s, m, seed, data=data, teacher_loss="la")
        del data
        distill._teacher_cache.clear()
        torch.cuda.empty_cache()
        print(f"[{d}] DATASET DONE", flush=True)
    print("BALANCED DONE", flush=True)


if __name__ == "__main__":
    main()
