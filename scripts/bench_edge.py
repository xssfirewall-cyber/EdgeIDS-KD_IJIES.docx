"""Edge-emulation benchmark (E6) over the ONNX models written by run_quant.py.

Latency depends on the architecture and the precision, not on the training
method, so one checkpoint per architecture is timed (seed 0, CE for students;
both teachers). Each model runs in a fresh process (see bench_one.py).

    python scripts/bench_edge.py   -> results/edge_bench.csv
"""
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.bench_one import TIMED, WARMUP  # noqa: E402
from scripts.run_quant import ONNX_DIR, STUDENTS  # noqa: E402
from scripts.train import DATASETS  # noqa: E402

VARIANTS = ("fp32", "int8_bal")   # int8_rand has the same graph, so the same speed


def cpu_name() -> str:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or platform.processor()
    except Exception:
        return platform.processor()


def main():
    rows_dir = config.RESULTS_DIR / "bench_rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for d in DATASETS:
        X = np.load(config.CACHE_DIR / f"{d}.npz")["X_test"]
        rows_path = rows_dir / f"{d}.npy"
        np.save(rows_path, X[np.random.default_rng(0).choice(len(X), WARMUP + TIMED)])
        del X
        tags = [p.name.split(".")[0] for p in (ONNX_DIR / d).glob("*_ce_s0.fp32.onnx")]
        tags = [t for t in tags if t.split("_ce_")[0] in STUDENTS or "_ce_" in t]
        for tag in sorted(set(tags)):
            for v in VARIANTS:
                path = ONNX_DIR / d / f"{tag}.{v}.onnx"
                if not path.exists():
                    continue
                out = subprocess.run([sys.executable, "-W", "ignore",
                                      str(Path(__file__).with_name("bench_one.py")),
                                      str(path), str(rows_path)],
                                     capture_output=True, text=True)
                line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
                if not line.startswith("{"):
                    print(f"[{d}] {tag}.{v} FAILED\n{out.stderr[-800:]}", flush=True)
                    continue
                r = json.loads(line)
                records.append({"dataset": d, "tag": tag, "variant": v, **r})
                print(f"[{d}] {tag:<28} {v:<9} p50={r['latency_us_p50']:8.1f}us "
                      f"p99={r['latency_us_p99']:8.1f}us  rss={r['session_rss_mb']:.1f}MB "
                      f"file={r['file_kb']:.0f}KB", flush=True)
    df = pd.DataFrame(records)
    df["cpu"] = cpu_name()
    df["emulation"] = "1 pinned core, 1 ORT thread, batch 1"
    df.to_csv(config.RESULTS_DIR / "edge_bench.csv", index=False)
    print(f"wrote {len(df)} rows; cpu = {df['cpu'].iloc[0]}")


if __name__ == "__main__":
    main()
