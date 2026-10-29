"""Figure for E5: macro-F1 and rare-class recall vs. student size.

One row per dataset, two panels (macro-F1 | rare recall), one line per
objective (mean over seeds, band = +-1 std), teacher as a dashed reference.

    python scripts/plot_sweep.py    -> results/figures/sweep.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.summarize import METHOD_ORDER, collect  # noqa: E402
from src import models  # noqa: E402

# Categorical slots 1-7 of the reference palette (validated: CVD and
# normal-vision separation pass on light). Contrast WARN on 3 slots is
# relieved by per-method markers, the legend, and results/summary.csv.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
LABELS = {"ce": "CE", "wce": "Class-weighted CE", "la": "Logit adjustment", "kd": "KD (Hinton)",
          "kd_wce": "KD + weighted CE", "rap_kd": "RAP-KD (ours)", "la_kd": "LA-KD (ours)"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e6e5e0", "#fcfcfb"
NAMES = {"ton_iot": "TON_IoT", "unsw_nb15": "UNSW-NB15", "nbaiot": "N-BaIoT",
         "ciciot2023": "CICIoT2023", "edge_iiot": "Edge-IIoTset"}


def main():
    df = collect()
    sweep = df[df["model"].isin(models.SWEEP)]
    teachers = df[df["model"].isin(models.TEACHERS)]
    datasets = [d for d in NAMES if d in set(sweep["dataset"])]
    if not datasets:
        sys.exit("no sweep results yet")

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
                         "figure.facecolor": "white"})
    fig, axes = plt.subplots(len(datasets), 2, figsize=(7.2, 2.3 * len(datasets)),
                             squeeze=False, sharex=True)
    for row, d in enumerate(datasets):
        g = sweep[sweep["dataset"] == d]
        # Teacher reference = the one distillation used (best validation macro-F1).
        t = teachers[teachers["dataset"] == d].sort_values("val_macro_f1").iloc[-1]
        for col, metric in enumerate(("macro_f1", "rare_recall")):
            ax = axes[row, col]
            for i, m in enumerate(METHOD_ORDER):
                s = g[g["loss"] == m].groupby("params")[metric].agg(["mean", "std"]).reset_index()
                if s.empty:
                    continue
                ax.fill_between(s["params"], s["mean"] - s["std"].fillna(0),
                                s["mean"] + s["std"].fillna(0), color=COLORS[i], alpha=0.12, lw=0)
                ax.plot(s["params"], s["mean"], color=COLORS[i], lw=2 if m in ("rap_kd", "la_kd") else 1.4,
                        marker=MARKERS[i], ms=4, label=LABELS[m])
            ax.axhline(t[metric], color=MUTED, ls="--", lw=1)
            if row == 0 or True:
                ax.text(0.99, t[metric], f"teacher ({t['params'] / 1e6:.1f}M)", color=MUTED,
                        fontsize=7, ha="right", va="bottom", transform=ax.get_yaxis_transform())
            ax.set_xscale("log")
            ax.grid(True, color=GRID, lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel(f"{NAMES[d]}\nmacro-F1")
            else:
                ax.set_ylabel("rare-class recall")
            if row == len(datasets) - 1:
                ax.set_xlabel("student parameters (log scale)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / len(datasets)))
    out = config.RESULTS_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"sweep.{ext}", dpi=200)
    print(f"wrote {out / 'sweep.png'} ({len(datasets)} datasets)")


if __name__ == "__main__":
    main()
