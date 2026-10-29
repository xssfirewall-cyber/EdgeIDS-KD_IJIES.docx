"""Training and evaluation shared by every experiment."""
from __future__ import annotations

import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn

import config


def load_split(name: str, device) -> dict:
    d = np.load(config.CACHE_DIR / f"{name}.npz")
    out = {"classes": [str(c) for c in d["classes"]]}
    for k in ("train", "val", "test"):
        out[f"X_{k}"] = torch.from_numpy(d[f"X_{k}"]).to(device)
        out[f"y_{k}"] = torch.from_numpy(d[f"y_{k}"]).to(device)
    return out


def class_weights(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Inverse-frequency ("balanced") weights, normalised to mean 1 over classes."""
    counts = torch.bincount(y, minlength=n_classes).float().clamp(min=1)
    w = counts.sum() / (n_classes * counts)
    return w / w.mean()


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, batch: int = 2048) -> torch.Tensor:
    # Same bf16 autocast as training; a larger batch makes cuDNN's LSTM request
    # one ~2 GB block, which Windows (WDDM) refuses even when 6 GB are free.
    model.eval()
    with torch.autocast(X.device.type, dtype=torch.bfloat16, enabled=X.device.type == "cuda"):
        return torch.cat([model(X[i:i + batch]).float() for i in range(0, len(X), batch)])


def metrics(y_true, logits, classes) -> dict:
    y = y_true.cpu().numpy()
    p = logits.argmax(1).cpu().numpy()
    labels = list(range(len(classes)))
    prec, rec, f1, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    return {
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, labels=labels, average="macro", zero_division=0),
        "per_class": {c: {"precision": round(float(a), 4), "recall": round(float(b), 4),
                          "f1": round(float(f), 4), "support": int(s)}
                      for c, a, b, f, s in zip(classes, prec, rec, f1, sup)},
    }


def fit(model, data, loss_fn, *, epochs=50, batch=1024, lr=1e-3, weight_decay=1e-4,
        patience=8, seed=0, log=print) -> dict:
    """AdamW + cosine schedule; early stopping on validation macro-F1.
    ``loss_fn(logits, y, idx)`` gets the batch's training-set indices, so
    distillation losses can look up precomputed teacher logits."""
    torch.manual_seed(seed)
    dev = data["X_train"].device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator(device=dev).manual_seed(seed)
    n = len(data["X_train"])
    best, best_state, best_epoch, stale = -1.0, None, 0, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev, generator=gen)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = data["X_train"][idx], data["y_train"][idx]
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = model(xb)
            loss = loss_fn(logits.float(), yb, idx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        val_f1 = metrics(data["y_val"], predict(model, data["X_val"]), data["classes"])["macro_f1"]
        if val_f1 > best:
            best, best_epoch, stale = val_f1, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        log(f"  ep {ep:>3}  loss {total / n:.4f}  val macro-F1 {val_f1:.4f}"
            + ("  *" if stale == 0 else ""))
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "epochs_run": ep, "val_macro_f1": best,
            "train_seconds": round(time.time() - t0, 1)}
