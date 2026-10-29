"""Hardware-independent deployment cost and microcontroller feasibility.

For every architecture and dataset (batch size 1, one flow):
  macs            multiply-accumulates of one forward pass (analytical, per layer)
  weights_fp32_kb all parameters in FP32
  weights_int8_kb int8 weights + one FP32 scale per output channel + int32 biases
                  for Linear/Conv layers; every other parameter stays FP32
  act_peak_kb     largest (input + output) activation of any single layer,
                  FP32 at batch 1: a lower bound on the RAM a layer-by-layer
                  runtime needs for activations
  ram_int8_kb     weights_int8_kb + act_peak_kb / 4 (INT8 activations): the
                  conservative "everything in on-chip SRAM" footprint

Device budgets are the on-chip SRAM of the two microcontrollers used by
De Bernardis et al. (Arduino Mega 2560: 8 KB; ESP32-C3: 400 KB).

    python scripts/cost_model.py   -> results/cost_model.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.train import DATASETS  # noqa: E402
from src import models  # noqa: E402

DEVICES_KB = {"Arduino Mega 2560": 8, "ESP32-C3": 400}
ARCHS = models.SWEEP + models.TEACHERS


def count(model: nn.Module, n_features: int) -> dict:
    macs, peak = 0, 0

    def hook(mod, inp, out):
        nonlocal macs, peak
        x = inp[0]
        o = out[0] if isinstance(out, tuple) else out
        peak = max(peak, x.numel() + o.numel())
        if isinstance(mod, nn.MultiheadAttention):
            L, d = x.shape[1], mod.embed_dim             # batch_first, batch 1
            # QKV + output projections, QK^T and AV. out_proj is counted here:
            # MultiheadAttention uses its weight directly, so its own forward
            # hook never fires (checked against a hand count).
            macs += 4 * L * d * d + 2 * L * L * d
        elif isinstance(mod, nn.Linear):
            macs += (x.numel() // x.shape[-1]) * mod.in_features * mod.out_features
        elif isinstance(mod, nn.Conv1d):
            k = mod.kernel_size[0]
            macs += o.shape[-1] * mod.out_channels * (mod.in_channels // mod.groups) * k
        elif isinstance(mod, nn.LSTM):
            L, h = x.shape[1], mod.hidden_size
            dirs = 2 if mod.bidirectional else 1
            for layer in range(mod.num_layers):
                inp_size = mod.input_size if layer == 0 else h * dirs
                macs += dirs * L * 4 * h * (inp_size + h)

    leaf = (nn.Linear, nn.Conv1d, nn.LSTM, nn.MultiheadAttention)
    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, leaf)]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, n_features))
    for h in handles:
        h.remove()
    if isinstance(model, models.FTTransformer):
        macs += n_features * model.weight.shape[1]       # feature tokenizer x*W + b

    w32, w8 = 0, 0
    quantizable = {id(p) for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv1d))
                   for p in m.parameters(recurse=False)}
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            w8 += m.weight.numel() + 4 * m.out_channels if isinstance(m, nn.Conv1d) else \
                  m.weight.numel() + 4 * m.out_features
            if m.bias is not None:
                w8 += 4 * m.bias.numel()
    for p in model.parameters():
        w32 += 4 * p.numel()
        if id(p) not in quantizable:
            w8 += 4 * p.numel()
    params = sum(p.numel() for p in model.parameters())
    return {"params": params, "macs": macs, "weights_fp32_kb": w32 / 1024,
            "weights_int8_kb": w8 / 1024, "act_peak_kb": 4 * peak / 1024,
            "ram_int8_kb": w8 / 1024 + peak / 1024}


def main():
    rows = []
    for d in DATASETS:
        z = np.load(config.CACHE_DIR / f"{d}.npz", mmap_mode="r")
        n_features, n_classes = len(z["features"]), len(z["classes"])
        for a in ARCHS:
            r = count(models.build(a, n_features, n_classes), n_features)
            for dev, kb in DEVICES_KB.items():
                r[f"fits_{dev}"] = r["ram_int8_kb"] <= kb
            rows.append({"dataset": d, "arch": a, **r})
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "cost_model.csv", index=False)
    pd.set_option("display.width", 220)
    show = df.groupby("arch", sort=False).agg(
        params=("params", "median"), macs=("macs", "median"),
        w8_min=("weights_int8_kb", "min"), w8_max=("weights_int8_kb", "max"),
        ram_min=("ram_int8_kb", "min"), ram_max=("ram_int8_kb", "max"),
        mega=("fits_Arduino Mega 2560", "sum"), esp=("fits_ESP32-C3", "sum"))
    print(show.round(1).to_string())


if __name__ == "__main__":
    main()
