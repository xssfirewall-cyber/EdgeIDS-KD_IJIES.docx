"""Pipeline sanity check: LightGBM on each cached dataset.

Confirms the splits are learnable and shows per-class recall, so the
rare-class gap the paper targets is visible before any deep model is trained.
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def main():
    out = {}
    for name in ("ton_iot", "unsw_nb15", "nbaiot"):
        d = np.load(config.CACHE_DIR / f"{name}.npz")
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31,
                                 random_state=0, verbose=-1)
        clf.fit(d["X_train"], d["y_train"],
                eval_set=[(d["X_val"], d["y_val"])],
                callbacks=[lgb.early_stopping(20, verbose=False)])
        pred = clf.predict(d["X_test"])
        classes = list(d["classes"])
        rec = recall_score(d["y_test"], pred, average=None, labels=range(len(classes)), zero_division=0)
        out[name] = {"macro_f1": f1_score(d["y_test"], pred, average="macro"),
                     "per_class_recall": dict(zip(classes, rec.round(4).tolist()))}
        print(f"[{name}] macro-F1={out[name]['macro_f1']:.4f}")
        for c, r in sorted(zip(classes, rec), key=lambda t: t[1])[:4]:
            print(f"    lowest recall  {c:<16} {r:.3f}")
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "sanity_lightgbm.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
