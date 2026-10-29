"""Trains one model with plain or class-weighted cross-entropy (experiments E1, E2).

    python scripts/train.py --dataset ton_iot --model cnn_bilstm_mha
    python scripts/train.py --dataset ton_iot --model mlp_s --loss wce --seed 1

Writes results/runs/<dataset>/<model>_<loss>_s<seed>.json and, for teachers,
checkpoints/<dataset>/<model>_<loss>_s<seed>.pt (needed for distillation).
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import models, train  # noqa: E402

DATASETS = ["ton_iot", "unsw_nb15", "nbaiot", "ciciot2023", "edge_iiot"]


def run(dataset, model_name, loss="ce", seed=0, epochs=50, quiet=False, record=True):
    """record=False: save the weights only, never overwrite results/runs (a GPU
    retrain is not bit-identical to the original run)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = train.load_split(dataset, dev)
    n_classes, n_features = len(data["classes"]), data["X_train"].shape[1]
    torch.manual_seed(seed)
    model = models.build(model_name, n_features, n_classes).to(dev)

    if loss == "la":
        # Logit adjustment (Menon et al., 2021); used for the balanced teachers.
        from src.losses import make
        loss_fn = make("la", data["y_train"], n_classes)
    else:
        w = train.class_weights(data["y_train"], n_classes) if loss == "wce" else None
        loss_fn = lambda logits, y, _idx: F.cross_entropy(logits, y, weight=w)  # noqa: E731
    # Large teachers on the big datasets: bigger batches keep epochs short.
    batch = 2048 if len(data["X_train"]) > 300_000 else 1024
    # FT-Transformer attention grows with n_features^2 (116 tokens on N-BaIoT).
    if model_name == "ft_transformer" and n_features > 100:
        batch = 512

    tag = f"{model_name}_{loss}_s{seed}"
    print(f"[{dataset}] {tag}  params={models.size_report(model)['params']:,}")
    info = train.fit(model, data, loss_fn, epochs=epochs, batch=batch, seed=seed,
                     log=(lambda *_: None) if quiet else print)
    test = train.metrics(data["y_test"], train.predict(model, data["X_test"]), data["classes"])
    result = {"dataset": dataset, "model": model_name, "loss": loss, "seed": seed,
              **models.size_report(model), **info, "test": test}

    if record:
        out = config.RESULTS_DIR / "runs" / dataset
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{tag}.json").write_text(json.dumps(result, indent=2))
    ck = config.ROOT / "checkpoints" / dataset
    if model_name not in models.TEACHERS:
        ck = ck / "students"
    ck.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ck / f"{tag}.pt")
    print(f"[{dataset}] {tag}  test macro-F1={test['macro_f1']:.4f}  acc={test['accuracy']:.4f}  "
          f"best_ep={info['best_epoch']}  {info['train_seconds']}s")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--model", choices=list(models.MODELS), required=True)
    ap.add_argument("--loss", choices=["ce", "wce", "la"], default="ce")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    a = ap.parse_args()
    run(a.dataset, a.model, a.loss, a.seed, a.epochs)


if __name__ == "__main__":
    main()
