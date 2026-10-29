"""Paper figures sized for the IEEE two-column layout (all from result files).

  sweep.pdf        Fig. 2  macro-F1 / rare recall vs. student size (2 x 5 grid, full width)
  ablation.pdf     H1 / H1b / H1c per dataset: what rescues small students
  balanced.pdf     teacher's vs. student's rare-recall gain; per-dataset effect
  quant.pdf        macro-F1 change after full INT8 vs. INT8 with FP32 input layer
  feasibility.pdf  rare recall vs. SRAM footprint with microcontroller budgets

    python scripts/plot_evidence.py   -> paper/figures/*.pdf
"""
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts import distill  # noqa: E402
from scripts.summarize import METHOD_ORDER, collect  # noqa: E402
from src import models  # noqa: E402

OUT = config.ROOT / "paper" / "figures"
# Validated categorical palette (CVD and normal-vision separation pass);
# markers give a second encoding, so colour is never the only cue.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
MK = ["o", "s", "^", "D", "v", "P", "X"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e6e5e0"
NAMES = {"ton_iot": "TON_IoT", "unsw_nb15": "UNSW-NB15", "nbaiot": "N-BaIoT",
         "ciciot2023": "CICIoT2023", "edge_iiot": "Edge-IIoTset"}
LABELS = {"ce": "CE", "wce": "WCE", "la": "LA", "kd": "KD", "kd_wce": "KD+WCE",
          "rap_kd": "RAP-KD", "la_kd": "LA-KD"}
COL1, COL2 = 3.5, 7.16          # IEEE column and text width (inches)
SMALL = ("mlp_h8", "mlp_h16", "mlp_h32")

plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42})


def style(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def sweep(runs):
    s = runs[runs["model"].isin(models.SWEEP)]
    t = runs[runs["model"].isin(models.TEACHERS) & (runs["loss"] == "ce")]
    fig, axes = plt.subplots(2, 5, figsize=(COL2, 3.0), sharex=True)
    for j, d in enumerate(NAMES):
        g = s[s["dataset"] == d]
        teach = t[t["dataset"] == d].sort_values("val_macro_f1").iloc[-1]
        for i, metric in enumerate(("macro_f1", "rare_recall")):
            ax = axes[i, j]
            for k, m in enumerate(METHOD_ORDER[:7]):
                v = g[g["loss"] == m].groupby("params")[metric].agg(["mean", "std"]).reset_index()
                ax.fill_between(v["params"], v["mean"] - v["std"].fillna(0), v["mean"] + v["std"].fillna(0),
                                color=C[k], alpha=0.12, lw=0)
                ax.plot(v["params"], v["mean"], color=C[k], marker=MK[k], ms=2.5, lw=1.0, label=LABELS[m])
            ax.axhline(teach[metric], color=MUTED, ls="--", lw=0.8)
            ax.set_xscale("log")
            style(ax)
            if i == 0:
                ax.set_title(NAMES[d], fontsize=7)
            if j == 0:
                ax.set_ylabel("macro-F1" if i == 0 else "rare-class recall")
            if i == 1:
                ax.set_xlabel("parameters")
    h, l = axes[0, 0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=MUTED, ls="--", lw=0.8)); l.append("teacher")
    fig.legend(h, l, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.04), fontsize=6.5)
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    save(fig, "sweep")


def ablation(runs):
    s = runs[runs["model"].isin(SMALL)].groupby(["dataset", "model", "loss"])["macro_f1"].mean().unstack()
    comps = [("H1: RAP-KD − CE", s["rap_kd"] - s["ce"]),
             ("H1b: RAP-KD − best(WCE, LA)", s["rap_kd"] - s[["wce", "la"]].max(axis=1)),
             ("H1c: KD − CE", s["kd"] - s["ce"])]
    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    ds = list(NAMES)
    y = np.arange(len(ds))
    for k, (lab, diff) in enumerate(comps):
        per = diff.groupby("dataset").mean().reindex(ds) * 100
        ax.scatter(per.values, y + (k - 1) * 0.22, color=C[k], marker=MK[k], s=16, label=lab, zorder=3)
        for yy, x in zip(y + (k - 1) * 0.22, per.values):
            ax.plot([0, x], [yy, yy], color=C[k], lw=0.8, alpha=0.6)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(y, [NAMES[d] for d in ds])
    ax.invert_yaxis()
    ax.set_xlabel("macro-F1 difference, small students (pts)")
    style(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=6)
    save(fig, "ablation")


def balanced(runs):
    s = runs[runs["model"].isin(("mlp_h16", "mlp_h64", "mlp_h256"))]
    rr = s.groupby(["dataset", "loss"])["rare_recall"].mean().unstack()
    student_gain = (rr["kd_wce_bt"] - rr["kd_wce"]) * 100
    t = runs[runs["model"].isin(models.TEACHERS)]
    teacher_gain = {}
    for d in NAMES:
        ce = distill.pick_teacher(d, "ce")
        cer = t[(t.dataset == d) & (t.model == ce["model"]) & (t.loss == "ce")].rare_recall.iloc[0]
        lar = t[(t.dataset == d) & (t.loss == "la")].rare_recall.iloc[0]
        teacher_gain[d] = (lar - cer) * 100
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL1, 3.7), gridspec_kw={"height_ratios": [1.15, 1]})
    # (a) teacher gain vs student gain
    x = np.array([teacher_gain[d] for d in NAMES]); yv = student_gain.reindex(list(NAMES)).values
    lim = (min(x.min(), yv.min()) - 4, max(x.max(), yv.max()) + 4)
    a.plot(lim, lim, color=MUTED, ls=":", lw=0.8)
    # Label positions in data coordinates; thin leader lines tie each label to
    # its point, since TON_IoT, Edge-IIoTset and CICIoT2023 sit close together.
    pos = {"ton_iot": (-4, 17), "unsw_nb15": (20, 30), "nbaiot": (3, -4),
           "ciciot2023": (21, 3), "edge_iiot": (8, 22)}
    for k, d in enumerate(NAMES):
        a.scatter(x[k], yv[k], color=C[k], marker=MK[k], s=18, zorder=3)
        a.annotate(NAMES[d], (x[k], yv[k]), xytext=pos[d], textcoords="data", fontsize=5.5,
                   color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5,
                                              shrinkA=1, shrinkB=3))
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("teacher rare-recall gain (pts)")
    a.set_ylabel("student rare-recall gain (pts)")
    a.set_title("(a)", fontsize=7, loc="left")
    style(a)
    # (b) per-dataset student rare recall: CE teacher vs balanced teacher
    ds = list(NAMES); yy = np.arange(len(ds))
    lo, hi = rr["kd_wce"].reindex(ds).values, rr["kd_wce_bt"].reindex(ds).values
    for i in range(len(ds)):
        b.plot([lo[i], hi[i]], [yy[i], yy[i]], color=MUTED, lw=1.0)
    b.scatter(lo, yy, color=C[0], marker=MK[0], s=16, label="CE teacher", zorder=3)
    b.scatter(hi, yy, color=C[1], marker=MK[1], s=16, label="balanced teacher", zorder=3)
    b.set_yticks(yy, [NAMES[d] for d in ds], fontsize=6)
    b.invert_yaxis()
    b.set_xlabel("student rare-class recall (KD+WCE)")
    b.set_title("(b)", fontsize=7, loc="left")
    b.set_xlim(0.15, 1.0)
    b.legend(loc="lower center", bbox_to_anchor=(0.6, 1.0), ncol=2, fontsize=6)
    style(b)
    fig.tight_layout(h_pad=1.2)
    save(fig, "balanced")


def quant():
    f = pd.concat([pd.read_csv(x) for x in glob.glob(str(config.RESULTS_DIR / "quant" / "*_fpin.csv"))])
    f = f[f["split"] == "val"].copy()
    f["arch"] = f["tag"].str.extract(r"^(mlp_h\d+|cnn_bilstm_mha|ft_transformer)")[0].replace(
        {"cnn_bilstm_mha": "teacher", "ft_transformer": "teacher"})
    g = f.groupby(["dataset", "arch", "variant"])["macro_f1"].mean().unstack()
    g["int8"] = (g["int8_bal"] - g["fp32"]) * 100
    g["fpin"] = (g["int8_bal_fpin"] - g["fp32"]) * 100
    arch_lab = {"teacher": "teacher", "mlp_h256": "MLP-256", "mlp_h64": "MLP-64", "mlp_h16": "MLP-16"}
    rows = [(d, a) for d in NAMES for a in ("teacher", "mlp_h256", "mlp_h64", "mlp_h16")]
    fig, ax = plt.subplots(figsize=(COL1, 3.0))
    y = np.arange(len(rows))
    for i, (d, a) in enumerate(rows):
        r = g.loc[(d, a)]
        ax.plot([r["int8"], r["fpin"]], [i, i], color=MUTED, lw=1.0)
    ax.scatter([g.loc[r, "int8"] for r in rows], y, color=C[1], marker=MK[1], s=12, label="full INT8", zorder=3)
    ax.scatter([g.loc[r, "fpin"] for r in rows], y, color=C[0], marker=MK[0], s=12,
               label="INT8, FP32 input layer", zorder=3)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(y, [f"{NAMES[d]} · {arch_lab[a]}" for d, a in rows], fontsize=5.5)
    for k in range(4, len(rows), 4):
        ax.axhline(k - 0.5, color=GRID, lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("macro-F1 change vs. FP32, validation (pts)")
    style(ax)
    ax.legend(loc="lower left", fontsize=6)
    save(fig, "quant")


def feasibility(runs):
    cost = pd.read_csv(config.RESULTS_DIR / "cost_model.csv")
    mem = cost.groupby("arch")["ram_int8_kb"].mean()
    s = runs[runs["model"].isin(models.SWEEP)]
    fig, ax = plt.subplots(figsize=(COL1, 2.1))
    for k, m in ((0, "ce"), (1, "wce"), (5, "rap_kd")):
        v = s[s["loss"] == m].groupby("model")["rare_recall"].mean().reindex(models.SWEEP)
        ax.plot(mem.reindex(models.SWEEP).values, v.values, color=C[k], marker=MK[k], ms=3, lw=1.0,
                label=f"student, {LABELS[m]}")
    t = runs[runs["model"].isin(models.TEACHERS) & (runs["loss"] == "ce")]
    for k, a in ((6, "cnn_bilstm_mha"), (3, "ft_transformer")):
        ax.scatter(mem[a], t[t.model == a]["rare_recall"].mean(), color=C[k], marker=MK[k], s=24,
                   zorder=3, label={"cnn_bilstm_mha": "CNN-BiLSTM-MHA (CE)",
                                    "ft_transformer": "FT-Transformer (CE)"}[a])
    for kb, name in ((8, "Arduino Mega\n8 KB"), (400, "ESP32-C3\n400 KB")):
        ax.axvline(kb, color=MUTED, ls="--", lw=0.8)
        ax.text(kb * 1.12, 0.05, name, fontsize=5.5, color=MUTED, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("SRAM footprint, INT8 (KB, mean over datasets)")
    ax.set_ylabel("rare-class recall")
    ax.set_ylim(0, 1)
    style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=5.5, ncol=3)
    save(fig, "feasibility")


def main():
    runs = collect()
    sweep(runs)
    ablation(runs)
    balanced(runs)
    quant()
    feasibility(runs)


if __name__ == "__main__":
    main()
