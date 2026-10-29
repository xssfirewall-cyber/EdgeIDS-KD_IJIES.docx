"""Collects results/runs/**.json into one table per dataset.

Rare class (fixed before any results were seen): fewer than 2% of the
training rows. "rare recall" is the mean recall over those classes.

    python scripts/summarize.py            # prints markdown, writes results/summary.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

RARE_FRACTION = 0.02


def rare_classes(dataset: str) -> list[str]:
    r = json.loads((config.CACHE_DIR / f"{dataset}_report.json").read_text())
    train = r["class_counts"]["train"]
    return [c for c, n in zip(r["classes"], train) if n < RARE_FRACTION * sum(train)]


def main():
    df = collect()
    df.to_csv(config.RESULTS_DIR / "summary.csv", index=False)
    if "--agg" in sys.argv:
        aggregate(df)
        return
    for d, g in df.groupby("dataset", sort=False):
        print(f"\n### {d}  (rare classes: {g['n_rare'].iloc[0]})\n")
        cols = ["model", "loss", "params", "kb_fp32", "macro_f1", "accuracy", "rare_recall", "worst_class"]
        print(g[cols].sort_values("params").to_markdown(index=False, floatfmt=".4f"))


def collect() -> pd.DataFrame:
    """One row per run in results/runs."""
    rows = []
    for f in sorted((config.RESULTS_DIR / "runs").glob("*/*.json")):
        r = json.loads(f.read_text())
        pc = r["test"]["per_class"]
        rare = rare_classes(r["dataset"])
        worst = min(pc, key=lambda c: pc[c]["recall"])
        rows.append({
            "dataset": r["dataset"], "model": r["model"], "loss": r["loss"], "seed": r["seed"],
            "params": r["params"], "kb_fp32": r["size_kb_fp32"],
            "macro_f1": r["test"]["macro_f1"], "accuracy": r["test"]["accuracy"],
            "rare_recall": sum(pc[c]["recall"] for c in rare) / len(rare) if rare else float("nan"),
            "n_rare": len(rare),
            "worst_class": f"{worst} ({pc[worst]['recall']:.2f})",
            "best_epoch": r["best_epoch"], "train_s": r["train_seconds"],
            "val_macro_f1": r["val_macro_f1"],
        })
    return pd.DataFrame(rows)


METHOD_ORDER = ["ce", "wce", "la", "kd", "kd_wce", "rap_kd", "la_kd",
                "kd_bt", "kd_wce_bt", "bt_kd"]


def aggregate(df: pd.DataFrame):
    """Students only: mean +- std over seeds, one table per dataset."""
    s = df[df["model"].isin(["mlp_s", "cnn_s"])]
    g = s.groupby(["dataset", "model", "loss"])
    agg = g[["macro_f1", "accuracy", "rare_recall"]].agg(["mean", "std"])
    agg["n_seeds"] = g.size()
    agg.to_csv(config.RESULTS_DIR / "summary_students_agg.csv")
    for d in agg.index.get_level_values(0).unique():
        print(f"\n### {d}\n")
        print("| student | method | seeds | macro-F1 | accuracy | rare recall |")
        print("|---|---|---|---|---|---|")
        block = agg.loc[d]
        for (m, loss) in sorted(block.index, key=lambda k: (k[0], METHOD_ORDER.index(k[1]))):
            r = block.loc[(m, loss)]
            cell = lambda c: f"{r[(c, 'mean')]:.4f} ± {0 if pd.isna(r[(c, 'std')]) else r[(c, 'std')]:.4f}"  # noqa: E731
            print(f"| {m} | {loss} | {int(r['n_seeds'].iloc[0] if hasattr(r['n_seeds'], 'iloc') else r['n_seeds'])} "
                  f"| {cell('macro_f1')} | {cell('accuracy')} | {cell('rare_recall')} |")


if __name__ == "__main__":
    main()
