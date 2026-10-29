"""Benchmarks one ONNX model under edge emulation; prints one JSON line.

Runs in its own process (spawned by bench_edge.py) so memory readings are not
contaminated by other models. Emulation: pinned to one CPU core, one ONNX
Runtime thread, batch size 1, 200 warm-up + 2,000 timed inferences.

    python scripts/bench_one.py <model.onnx> <rows.npy>

<rows.npy> holds the WARMUP + TIMED input rows, prepared by the parent, so
this process never loads a full dataset and its peak memory reflects the
runtime and the model only.
"""
import json
import os
import sys
import time
from pathlib import Path

import psutil

WARMUP, TIMED = 200, 2000

proc = psutil.Process()
# Core 2 avoids core 0 (most OS interrupts) on the laptop; a 1-OCPU ARM VM has
# a single core, so fall back to the last one that exists.
_allowed = proc.cpu_affinity()
CORE = _allowed[min(int(os.environ.get("BENCH_CORE", 2)), len(_allowed) - 1)]


def peak_mb() -> float:
    mi = proc.memory_info()
    if hasattr(mi, "peak_wset"):                    # Windows
        return mi.peak_wset / 2**20
    import resource                                 # Linux: ru_maxrss in KiB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

# Deliberately self-contained: importing src.edge would pull in PyTorch, which
# inflates the process's memory figures and can fail under a tight commit limit.


def session(path: Path) -> ort.InferenceSession:
    """Same settings as src.edge.session(threads=1)."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def main(model_path: str, rows_path: str):
    proc.cpu_affinity([CORE])     # here, not at import: bench_edge imports this module
    rows = np.load(rows_path).astype(np.float32)
    assert len(rows) >= WARMUP + TIMED
    rss_before = proc.memory_info().rss

    sess = session(Path(model_path))
    for i in range(WARMUP):
        sess.run(None, {"x": rows[i:i + 1]})
    lat = np.empty(TIMED)
    for i in range(TIMED):
        x = rows[WARMUP + i:WARMUP + i + 1]
        t0 = time.perf_counter_ns()
        sess.run(None, {"x": x})
        lat[i] = (time.perf_counter_ns() - t0) / 1000.0     # microseconds
    mi = proc.memory_info()
    print(json.dumps({
        "model": Path(model_path).name,
        "latency_us_p50": float(np.percentile(lat, 50)),
        "latency_us_p99": float(np.percentile(lat, 99)),
        "throughput_per_s": float(1e6 / lat.mean()),
        "session_rss_mb": (mi.rss - rss_before) / 2**20,
        "process_peak_mb": peak_mb(),
        "core": CORE,
        "file_kb": Path(model_path).stat().st_size / 1024,
    }))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
