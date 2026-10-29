"""INT8 with the input layer kept in FP32 (follow-up to run_quant.py).

Reuses the FP32 ONNX files from run_quant.py (no retraining) and adds the
variant int8_bal_fpin: balanced calibration, input layer excluded from
quantization. FP32, int8_bal and int8_bal_fpin are scored on BOTH validation
and test, because the fix was found by inspecting one model's test scores;
the validation columns let the choice be justified without the test split.

Writes results/quant/<dataset>_fpin.csv. Restartable per dataset; exits with
code 3 on allocation failures so run_retry.ps1 can restart it.
"""
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.run_quant import ONNX_DIR, evaluate  # noqa: E402
from scripts.summarize import rare_classes  # noqa: E402
from scripts.train import DATASETS  # noqa: E402
from src import edge  # noqa: E402


def main():
    for d in DATASETS:
        out_csv = config.RESULTS_DIR / "quant" / f"{d}_fpin.csv"
        if out_csv.exists():
            print(f"[{d}] done already", flush=True)
            continue
        npz = dict(np.load(config.CACHE_DIR / f"{d}.npz"))
        classes = [str(c) for c in npz["classes"]]
        rare = rare_classes(d)
        calib = edge.calibration_rows(npz["X_train"], npz["y_train"], "bal")
        tags = sorted(pd.read_csv(config.RESULTS_DIR / "quant" / f"{d}.csv")["tag"].unique())
        rows = []
        try:
            for tag in tags:
                fp32 = ONNX_DIR / d / f"{tag}.fp32.onnx"
                fpin = edge.quantize_int8(fp32, ONNX_DIR / d / f"{tag}.int8_bal_fpin.onnx",
                                          calib, fp32_input=True)
                variants = {"fp32": fp32, "int8_bal": ONNX_DIR / d / f"{tag}.int8_bal.onnx",
                            "int8_bal_fpin": fpin}
                for v, p in variants.items():
                    for split in ("val", "test"):
                        m = evaluate(edge.onnx_logits(p, npz[f"X_{split}"]), npz[f"y_{split}"],
                                     classes, rare)
                        rows.append({"dataset": d, "tag": tag, "variant": v, "split": split,
                                     "onnx_kb": p.stat().st_size / 1024, **m})
                print(f"[{d}] {tag} done", flush=True)
        except Exception as e:
            traceback.print_exc()
            if isinstance(e, MemoryError) or "allocate" in str(e) or "paging file" in str(e):
                print("RESOURCE ERROR - exiting for restart", flush=True)
                sys.exit(3)
            raise
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[{d}] DATASET DONE ({len(rows)} rows)", flush=True)
    print("FPIN DONE", flush=True)


if __name__ == "__main__":
    main()
