"""Week 2 batch: teachers (E1) and scratch students with CE / class-weighted CE (E2).

Skips runs whose result JSON already exists, so it can be restarted safely.
Students run first (minutes), teachers after (hours).
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.train import DATASETS, run  # noqa: E402

SEED = 0
jobs = ([(d, m, loss) for d in DATASETS for m in ("mlp_s", "cnn_s") for loss in ("ce", "wce")]
        + [(d, m, "ce") for d in DATASETS for m in ("cnn_bilstm_mha", "ft_transformer")])

for i, (d, m, loss) in enumerate(jobs, 1):
    out = config.RESULTS_DIR / "runs" / d / f"{m}_{loss}_s{SEED}.json"
    if out.exists():
        print(f"({i}/{len(jobs)}) skip {out.name} [{d}]", flush=True)
        continue
    print(f"({i}/{len(jobs)}) start {d} {m} {loss}", flush=True)
    try:
        run(d, m, loss, SEED, quiet=True)
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
print("WEEK2 DONE", flush=True)
