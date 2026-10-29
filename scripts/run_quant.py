"""INT8 post-training quantization (E4).

For each dataset x student size x method x seed: retrain the student (weights
only; results/runs is never overwritten), export to ONNX, quantize with a
random and a class-balanced calibration set, and evaluate FP32 / INT8 on the
test split. Also exports + quantizes each dataset's selected teachers.

Writes results/quant/<dataset>.csv and models/onnx/<dataset>/*.onnx.
Restartable; exits on a CUDA error so run_retry.ps1 can restart it.
"""
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts import distill  # noqa: E402
from scripts import train as train_script  # noqa: E402
from scripts.summarize import rare_classes  # noqa: E402
from src import edge, models, train  # noqa: E402

SEEDS = (0, 1, 2)
STUDENTS = ("mlp_h16", "mlp_h64", "mlp_h256")
# method -> (trainer, teacher_loss); tags match results/runs naming.
METHODS = {"ce": None, "wce": None, "kd_wce": "ce", "kd_wce_bt": "la"}
ONNX_DIR = config.ROOT / "models" / "onnx"


def ensure_student(d, s, m, seed, data) -> Path:
    ck = config.ROOT / "checkpoints" / d / "students" / f"{s}_{m}_s{seed}.pt"
    if not ck.exists():
        if METHODS[m] is None:
            train_script.run(d, s, m, seed, quiet=True, record=False)
        else:
            distill.run(d, s, "kd_wce", seed, data=data, teacher_loss=METHODS[m], record=False)
    return ck


def evaluate(logits, y, classes, rare):
    import torch as t
    r = train.metrics(t.from_numpy(y), t.from_numpy(logits), classes)
    pc = r["per_class"]
    return {"macro_f1": r["macro_f1"], "accuracy": r["accuracy"],
            "rare_recall": float(np.mean([pc[c]["recall"] for c in rare]))}


def process(d, arch, tag, state_path, npz, rare, rows):
    classes = [str(c) for c in npz["classes"]]
    n_features, n_classes = npz["X_train"].shape[1], len(classes)
    model = models.build(arch, n_features, n_classes)
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    out = ONNX_DIR / d
    fp32 = edge.export_onnx(model, n_features, out / f"{tag}.fp32.onnx")
    variants = {"fp32": fp32}
    for kind in ("rand", "bal"):
        calib = edge.calibration_rows(npz["X_train"], npz["y_train"], kind)
        variants[f"int8_{kind}"] = edge.quantize_int8(fp32, out / f"{tag}.int8_{kind}.onnx", calib)
    for v, p in variants.items():
        rows.append({"dataset": d, "tag": tag, "arch": arch, "variant": v,
                     "params": models.size_report(model)["params"], "onnx_kb": p.stat().st_size / 1024,
                     **evaluate(edge.onnx_logits(p, npz["X_test"]), npz["y_test"], classes, rare)})


def main():
    dev = torch.device("cuda")
    (config.RESULTS_DIR / "quant").mkdir(parents=True, exist_ok=True)
    for d in train_script.DATASETS:
        csv = config.RESULTS_DIR / "quant" / f"{d}.csv"
        if csv.exists():
            print(f"[{d}] done already", flush=True)
            continue
        npz = dict(np.load(config.CACHE_DIR / f"{d}.npz"))
        rare = rare_classes(d)
        data = train.load_split(d, dev)
        rows = []
        try:
            for tl in ("ce", "la"):
                t = distill.pick_teacher(d, tl)
                process(d, t["model"], t["tag"], t["checkpoint"], npz, rare, rows)
            for s in STUDENTS:
                for m in METHODS:
                    for seed in SEEDS:
                        ck = ensure_student(d, s, m, seed, data)
                        process(d, s, f"{s}_{m}_s{seed}", ck, npz, rare, rows)
        except Exception as e:
            traceback.print_exc()
            # CUDA errors and allocation failures come from the shared machine
            # (GPU and Windows commit limit), not from this code: retry them.
            if "CUDA" in str(e) or isinstance(e, MemoryError) or "allocate" in str(e):
                print("RESOURCE ERROR - exiting for restart", flush=True)
                sys.exit(3)
            raise
        pd.DataFrame(rows).to_csv(csv, index=False)
        del data
        distill._teacher_cache.clear()
        torch.cuda.empty_cache()
        print(f"[{d}] DATASET DONE ({len(rows)} rows)", flush=True)
    print("QUANT DONE", flush=True)


if __name__ == "__main__":
    main()
