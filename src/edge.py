"""ONNX export, INT8 post-training quantization, and ONNX evaluation (E4).

INT8 uses ONNX Runtime static quantization (QDQ format, per-channel int8
weights, uint8 activations). The calibration set decides the activation
ranges, so two variants are compared:
    int8_rand  2,000 random training rows (rare classes almost absent)
    int8_bal   up to 200 training rows per class (rare classes represented)
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

CALIB_RANDOM_N = 2000
CALIB_PER_CLASS = 200


def export_onnx(model: torch.nn.Module, n_features: int, path: Path) -> Path:
    model = model.eval().cpu().float()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, n_features), str(path), input_names=["x"],
                      output_names=["logits"], dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    return path


def calibration_rows(X: np.ndarray, y: np.ndarray, kind: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "rand":
        return X[rng.choice(len(X), min(CALIB_RANDOM_N, len(X)), replace=False)]
    if kind == "bal":
        idx = []
        for c in np.unique(y):
            cls = np.flatnonzero(y == c)
            idx.append(rng.choice(cls, min(CALIB_PER_CLASS, len(cls)), replace=False))
        return X[np.concatenate(idx)]
    raise ValueError(kind)


class _Reader(CalibrationDataReader):
    def __init__(self, rows: np.ndarray, batch: int = 64):
        self._it = iter([{"x": rows[i:i + batch].astype(np.float32)}
                         for i in range(0, len(rows), batch)])

    def get_next(self):
        return next(self._it, None)


_PASS_THROUGH = {"Unsqueeze", "Reshape", "Transpose", "Cast", "Identity", "Flatten", "Squeeze"}


def input_layer_nodes(model_path: Path) -> list[str]:
    """Names of the first compute nodes fed by the graph input, skipping
    shape-only ops (e.g. Unsqueeze before a Conv). Excluding them from
    quantization keeps the raw features in FP32: scaled IDS features have
    rare extreme values (TON_IoT: max 83 vs. 99.9th percentile 3.8), so a
    min-max uint8 input grid leaves typical values only ~16 levels."""
    import onnx
    g = onnx.load(str(model_path)).graph
    frontier, seen, first = {g.input[0].name}, set(), []
    while frontier:
        nxt = set()
        for n in g.node:
            if n.name in seen or not frontier.intersection(n.input):
                continue
            seen.add(n.name)
            if n.op_type in _PASS_THROUGH:
                nxt.update(n.output)
            else:
                first.append(n.name)
        frontier = nxt
    return first


def quantize_int8(fp32_path: Path, out_path: Path, calib_rows: np.ndarray,
                  fp32_input: bool = False) -> Path:
    # ONNX Runtime's quantizer writes and re-reads temporary models; on Windows
    # that fails silently for non-ASCII paths (the project path contains
    # Arabic), so all quantization work happens in an ASCII temp directory.
    with tempfile.TemporaryDirectory(prefix="edgeids_q_") as tmp:
        tmp = Path(tmp)
        src, pre, dst = tmp / "fp32.onnx", tmp / "pre.onnx", tmp / "int8.onnx"
        shutil.copyfile(fp32_path, src)
        quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
        exclude = input_layer_nodes(pre) if fp32_input else []
        quantize_static(str(pre), str(dst), _Reader(calib_rows), quant_format=QuantFormat.QDQ,
                        per_channel=True, activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QInt8, nodes_to_exclude=exclude)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dst), str(out_path))
    return out_path


def session(path: Path, threads: int = 1) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def onnx_logits(path: Path, X: np.ndarray, batch: int = 512) -> np.ndarray:
    # 512, not 4096: FT-Transformer attention at 4096 asks for one ~400 MB buffer.
    s = session(path, threads=4)
    return np.concatenate([s.run(None, {"x": X[i:i + batch].astype(np.float32)})[0]
                           for i in range(0, len(X), batch)])
