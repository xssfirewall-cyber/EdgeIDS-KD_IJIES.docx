"""Trains a student with a week-3 objective (experiment E3).

    python scripts/distill.py --dataset ton_iot --student mlp_s --method rap_kd --seed 0

The teacher is the checkpointed teacher with the best *validation* macro-F1
(never test). Writes results/runs/<dataset>/<student>_<method>_s<seed>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import losses, models, train  # noqa: E402
from scripts.train import DATASETS  # noqa: E402


def pick_teacher(dataset: str, teacher_loss: str = "ce") -> dict:
    """Best teacher by validation macro-F1 among checkpointed teachers trained
    with ``teacher_loss`` ("ce" = standard, "la" = balanced)."""
    best = None
    for f in (config.RESULTS_DIR / "runs" / dataset).glob("*.json"):
        r = json.loads(f.read_text())
        ck = config.ROOT / "checkpoints" / dataset / f"{f.stem}.pt"
        if r["model"] in models.TEACHERS and r["loss"] == teacher_loss and ck.exists():
            if best is None or r["val_macro_f1"] > best["val_macro_f1"]:
                best = {**r, "tag": f.stem, "checkpoint": ck}
    if best is None:
        raise FileNotFoundError(f"no trained teacher for {dataset}")
    return best


_teacher_cache: dict = {}


def teacher_logits(dataset: str, data: dict, teacher_loss: str = "ce") -> tuple[torch.Tensor, dict]:
    key = (dataset, teacher_loss)
    if key not in _teacher_cache:
        info = pick_teacher(dataset, teacher_loss)
        n_classes, n_features = len(data["classes"]), data["X_train"].shape[1]
        t = models.build(info["model"], n_features, n_classes).to(data["X_train"].device)
        t.load_state_dict(torch.load(info["checkpoint"], map_location=data["X_train"].device))
        logits = train.predict(t, data["X_train"])
        del t
        torch.cuda.empty_cache()
        _teacher_cache[key] = (logits, info)
    return _teacher_cache[key]


def run(dataset, student, method, seed=0, epochs=50, data=None, quiet=True, teacher_loss="ce",
        record=True):
    """teacher_loss="la" distils from the balanced teacher; results are then
    tagged <method>_bt so they never overwrite the standard-teacher runs.
    record=False saves the weights only (see scripts/train.py)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data or train.load_split(dataset, dev)
    n_classes, n_features = len(data["classes"]), data["X_train"].shape[1]

    t_logits, t_info = (teacher_logits(dataset, data, teacher_loss)
                        if method in losses.NEEDS_TEACHER else (None, None))
    label = method + ("_bt" if teacher_loss == "la" and method != "bt_kd" else "")
    loss_fn = losses.make(method, data["y_train"], n_classes, t_logits)
    torch.manual_seed(seed)
    model = models.build(student, n_features, n_classes).to(dev)
    batch = 2048 if len(data["X_train"]) > 300_000 else 1024

    tag = f"{student}_{label}_s{seed}"
    info = train.fit(model, data, loss_fn, epochs=epochs, batch=batch, seed=seed,
                     log=(lambda *_: None) if quiet else print)
    test = train.metrics(data["y_test"], train.predict(model, data["X_test"]), data["classes"])
    result = {"dataset": dataset, "model": student, "loss": label, "seed": seed,
              "teacher": t_info and t_info["tag"],
              "hyper": losses.hyper(method),
              **models.size_report(model), **info, "test": test}
    if record:
        out = config.RESULTS_DIR / "runs" / dataset
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{tag}.json").write_text(json.dumps(result, indent=2))
    ck = config.ROOT / "checkpoints" / dataset / "students"
    ck.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ck / f"{tag}.pt")
    print(f"[{dataset}] {tag}  teacher={result['teacher']}  test macro-F1={test['macro_f1']:.4f}  "
          f"best_ep={info['best_epoch']}  {info['train_seconds']}s", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--student", choices=list(models.MODELS), required=True)
    ap.add_argument("--method", choices=losses.METHODS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.dataset, a.student, a.method, a.seed, quiet=False)


if __name__ == "__main__":
    main()
