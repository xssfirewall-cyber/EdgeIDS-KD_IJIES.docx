"""Student-size sweep (E5) + week-3 re-run with the validation-selected KD settings.

For every dataset: MLP students of 7 widths (mlp_h8 .. mlp_h512) and cnn_s,
x 7 objectives (ce, wce, la, kd, kd_wce, rap_kd, la_kd) x 3 seeds.
Waits (up to 6 h) for a dataset's teachers instead of skipping it, so every
KD run on a dataset uses the final teacher choice. Restartable.
"""
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts import distill  # noqa: E402
from scripts import train as train_script  # noqa: E402
from scripts.run_week3 import teachers_ready  # noqa: E402
from src import losses, models, train  # noqa: E402

SEEDS = (0, 1, 2)
METHODS = ("ce", "wce") + losses.METHODS
STUDENTS = models.SWEEP + ("cnn_s",)


def main():
    dev = torch.device("cuda")
    for d in train_script.DATASETS:
        waited = 0
        while not teachers_ready(d):
            if waited == 0:
                print(f"[{d}] waiting for teachers ...", flush=True)
            if waited > 6 * 3600:
                print(f"[{d}] teachers never finished - skipped", flush=True)
                break
            time.sleep(300)
            waited += 300
        if not teachers_ready(d):
            continue
        data = train.load_split(d, dev)
        for s in STUDENTS:
            for m in METHODS:
                for seed in SEEDS:
                    if (config.RESULTS_DIR / "runs" / d / f"{s}_{m}_s{seed}.json").exists():
                        continue
                    try:
                        if m in ("ce", "wce"):
                            train_script.run(d, s, m, seed, quiet=True)
                        else:
                            distill.run(d, s, m, seed, data=data)
                    except Exception as e:
                        traceback.print_exc()
                        if "CUDA" in str(e):
                            # A CUDA error poisons the context for every later
                            # run in this process; exit so the wrapper restarts.
                            print("CUDA ERROR - exiting for restart", flush=True)
                            sys.exit(3)
        del data
        distill._teacher_cache.clear()
        torch.cuda.empty_cache()
        print(f"[{d}] DATASET DONE", flush=True)
    print("SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
