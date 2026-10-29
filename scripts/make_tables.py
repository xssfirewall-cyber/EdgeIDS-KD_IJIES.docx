"""Generates every LaTeX table of the paper from the result files.

No number in the paper is typed by hand: rerun this after any experiment.
    python scripts/make_tables.py   -> paper/tables/*.tex
"""
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.summarize import collect, rare_classes  # noqa: E402
from src import models  # noqa: E402

OUT = config.ROOT / "paper" / "tables"
NAMES = {"ton_iot": "TON\\_IoT", "unsw_nb15": "UNSW-NB15", "nbaiot": "N-BaIoT",
         "ciciot2023": "CICIoT2023", "edge_iiot": "Edge-IIoTset"}
ORDER = list(NAMES)
METHODS = ["ce", "wce", "la", "kd", "kd_wce", "rap_kd", "la_kd"]
MNAME = {"ce": "CE", "wce": "WCE", "la": "LA", "kd": "KD", "kd_wce": "KD+WCE",
         "rap_kd": "RAP-KD", "la_kd": "LA-KD", "kd_wce_bt": "KD+WCE (BT)"}


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body, encoding="utf-8")
    print("wrote", name)


def f3(x):
    return f"{x:.3f}"


def datasets_table():
    rows = []
    for d in ORDER:
        r = json.loads((config.CACHE_DIR / f"{d}_report.json").read_text())
        rare = rare_classes(d)
        # N-BaIoT and CICIoT2023 loaders de-duplicate (float32) before capping;
        # those counts come from scripts/count_loader_dups.py.
        ld = config.CACHE_DIR / f"{d}_loader_dups.json"
        if ld.exists():
            l = json.loads(ld.read_text())
            n = l.get("within_file_float32_duplicates", l.get("within_sample_float32_duplicates"))
            base = l.get("raw_rows_all_files") if "within_file_float32_duplicates" in l else l["sampled_rows"]
            pre = f"{n:,} ({100 * n / base:.0f}\\%)$^{{a}}$" if "within_file" in str(l) else f"{n:,} ({100 * n / base:.0f}\\%)$^{{b}}$"
        else:
            pre = "--"
        rows.append(f"{NAMES[d]} & {pre} & {r['raw_rows']:,} & {r['exact_duplicates_removed']:,} "
                    f"({100 * r['exact_duplicates_removed'] / r['raw_rows']:.0f}\\%) & "
                    f"{r['conflicting_label_rows_removed']:,} & {r['rows_after']:,} & {r['n_features']} & "
                    f"{len(r['classes'])} & {len(rare)} \\\\")
    write("datasets", "\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
          "& Pre-cap dup. & Rows used & Duplicates & Conflicts & Kept & Feat. & Classes & Rare \\\\\n\\midrule\n"
          + "\n".join(rows).replace(",", "{,}") + "\n\\bottomrule\n\\end{tabular}\n")


def teachers_table(runs):
    t = runs[runs["model"].isin(models.TEACHERS)]
    rows = []
    for d in ORDER:
        g = t[t["dataset"] == d]
        for _, r in g.sort_values(["loss", "model"]).iterrows():
            name = {"cnn_bilstm_mha": "CNN-BiLSTM-MHA", "ft_transformer": "FT-Transformer"}[r["model"]]
            loss = {"ce": "CE", "la": "LA (balanced)"}[r["loss"]]
            rows.append(f"{NAMES[d]} & {name} & {loss} & {r['params'] / 1e6:.2f}M & "
                        f"{f3(r['macro_f1'])} & {f3(r['rare_recall'])} \\\\")
    write("teachers", "\\begin{tabular}{lllrrr}\n\\toprule\n"
          "Dataset & Teacher & Loss & Params & Macro-F1 & Rare recall \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def methods_table(runs, size, metric, name):
    s = runs[(runs["model"] == size) & runs["loss"].isin(METHODS)]
    g = s.groupby(["dataset", "loss"])[metric].agg(["mean", "std"])
    rows = []
    for d in ORDER:
        vals = {m: g.loc[(d, m)] for m in METHODS if (d, m) in g.index}
        best = max(vals, key=lambda m: vals[m]["mean"])
        cells = [(f"\\textbf{{{vals[m]['mean']:.3f}}}" if m == best else f"{vals[m]['mean']:.3f}")
                 + f"\\,{{\\scriptsize$\\pm${vals[m]['std']:.3f}}}" for m in METHODS]
        params = int(s[s["dataset"] == d]["params"].iloc[0])
        rows.append(f"{NAMES[d]} ({params:,}) & ".replace(",", "{,}") + " & ".join(cells) + " \\\\")
    write(name, "\\begin{tabular}{l" + "c" * len(METHODS) + "}\n\\toprule\n"
          "Dataset (params) & " + " & ".join(MNAME[m] for m in METHODS) + " \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def balanced_table(runs):
    s = runs[runs["model"] == "mlp_h64"]
    g = s.groupby(["dataset", "loss"])[["macro_f1", "rare_recall"]].mean()
    rows = []
    for d in ORDER:
        cells = []
        for m in ("wce", "kd_wce", "kd_wce_bt"):
            cells += [f3(g.loc[(d, m), "macro_f1"]), f3(g.loc[(d, m), "rare_recall"])]
        rows.append(f"{NAMES[d]} & " + " & ".join(cells) + " \\\\")
    write("balanced", "\\begin{tabular}{lcccccc}\n\\toprule\n"
          "& \\multicolumn{2}{c}{WCE} & \\multicolumn{2}{c}{KD+WCE, CE teacher} "
          "& \\multicolumn{2}{c}{KD+WCE, balanced teacher} \\\\\n"
          "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
          "Dataset & Macro-F1 & Rare rec. & Macro-F1 & Rare rec. & Macro-F1 & Rare rec. \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def _arch(tag):
    for a in ("mlp_h16", "mlp_h64", "mlp_h256"):
        if tag.startswith(a):
            return a
    return "teacher"


def quant_table():
    q = pd.concat([pd.read_csv(f) for f in glob.glob(str(config.RESULTS_DIR / "quant" / "*.csv"))
                   if not f.endswith("_fpin.csv")])
    f = pd.concat([pd.read_csv(f) for f in glob.glob(str(config.RESULTS_DIR / "quant" / "*_fpin.csv"))])
    f = f[(f["split"] == "test") & (f["variant"] == "int8_bal_fpin")]
    allv = pd.concat([q, f.drop(columns="split")])
    allv["arch"] = allv["tag"].map(_arch)
    g = allv.groupby(["arch", "variant"])[["macro_f1", "rare_recall", "onnx_kb"]].mean()
    label = {"teacher": "Teachers", "mlp_h256": "MLP-256", "mlp_h64": "MLP-64", "mlp_h16": "MLP-16"}
    rows = []
    for a in ("teacher", "mlp_h256", "mlp_h64", "mlp_h16"):
        cells = []
        for v in ("fp32", "int8_rand", "int8_bal", "int8_bal_fpin"):
            r = g.loc[(a, v)]
            kb = f"{r['onnx_kb'] / 1024:.1f}\\,MB" if r["onnx_kb"] > 1024 else f"{r['onnx_kb']:.0f}\\,KB"
            cells.append(f"{f3(r['macro_f1'])} / {f3(r['rare_recall'])} / {kb}")
        rows.append(f"{label[a]} & " + " & ".join(cells) + " \\\\")
    write("quant", "\\begin{tabular}{lcccc}\n\\toprule\n"
          "Model & FP32 & INT8, random calib. & INT8, balanced calib. & INT8, bal. + FP32 input \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def edge_table():
    b = pd.read_csv(config.RESULTS_DIR / "edge_bench.csv")
    b["arch"] = b["tag"].str.replace(r"_ce_s0$", "", regex=True)
    b["arch"] = b["arch"].replace({"cnn_bilstm_mha": "CNN-BiLSTM-MHA", "ft_transformer": "FT-Transformer",
                                   "mlp_h16": "MLP-16", "mlp_h64": "MLP-64", "mlp_h256": "MLP-256"})
    g = b.groupby(["arch", "variant"]).agg(
        p50_min=("latency_us_p50", "min"), p50_max=("latency_us_p50", "max"),
        rss=("session_rss_mb", "mean"), kb=("file_kb", "mean"), n=("dataset", "count"))
    def span(lo_v, hi_v):
        fmt = "{:.1f}" if hi_v < 100 else "{:.0f}"
        lo, hi = fmt.format(lo_v), fmt.format(hi_v)
        return lo if lo == hi else f"{lo}--{hi}"

    rows = []
    for a in ("CNN-BiLSTM-MHA", "FT-Transformer", "MLP-256", "MLP-64", "MLP-16"):
        cells = []
        for v in ("fp32", "int8_bal"):
            r = g.loc[(a, v)]
            cells += [span(r["p50_min"], r["p50_max"]), f"{r['rss']:.1f}"]
        rows.append(f"{a} & " + " & ".join(cells) + " \\\\")
    write("edge", "\\begin{tabular}{lcccc}\n\\toprule\n"
          "& \\multicolumn{2}{c}{FP32} & \\multicolumn{2}{c}{INT8} \\\\\n"
          "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
          "Model & p50 ($\\mu$s) & RSS (MB) & p50 ($\\mu$s) & RSS (MB) \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def cost_table():
    """Hardware-independent cost (scripts/cost_model.py) and SRAM feasibility."""
    c = pd.read_csv(config.RESULTS_DIR / "cost_model.csv")
    label = {"cnn_bilstm_mha": "CNN-BiLSTM-MHA", "ft_transformer": "FT-Transformer",
             **{f"mlp_h{h}": f"MLP-{h}" for h in models.SWEEP_WIDTHS}}

    def num(x):
        return f"{x / 1e6:.1f}M" if x >= 1e6 else (f"{x / 1e3:.1f}k" if x >= 1e3 else f"{x:.0f}")

    def kb(x):
        return f"{x / 1024:.1f}\\,MB" if x >= 1024 else f"{x:.1f}"

    def rng(s, fmt):
        lo, hi = fmt(s.min()), fmt(s.max())
        return lo if lo == hi else f"{lo}--{hi}"

    rows = []
    for a in list(models.SWEEP) + list(models.TEACHERS):
        g = c[c["arch"] == a]
        rows.append(f"{label[a]} & {rng(g['params'], num)} & {rng(g['macs'], num)} & "
                    f"{rng(g['weights_int8_kb'], kb)} & {rng(g['ram_int8_kb'], kb)} & "
                    f"{int(g['fits_Arduino Mega 2560'].sum())}/5 & {int(g['fits_ESP32-C3'].sum())}/5 \\\\")
        if a == models.SWEEP[-1]:
            rows.append("\\midrule")
    write("cost", "\\begin{tabular}{lcccccc}\n\\toprule\n"
          "& & & & & \\multicolumn{2}{c}{Fits on-chip SRAM (datasets)} \\\\\n\\cmidrule(lr){6-7}\n"
          "Model & Params & MACs/flow & INT8 weights (KB) & RAM, INT8 (KB) & Arduino Mega (8\\,KB) & ESP32-C3 (400\\,KB) \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def stats_table():
    s = pd.read_csv(config.RESULTS_DIR / "stats.csv")
    rows = []
    for _, r in s.iterrows():
        p = f"{r['p_holm']:.1e}" if r["p_holm"] < 1e-3 else f"{r['p_holm']:.3f}"
        rows.append(f"{r['id']} & {r['claim'].replace('_', ' ')} & "
                    f"{r['datasets_pos']}/{r['datasets']} & {r['p_sign_datasets']:.3f} & {r['n']} & "
                    f"${r['mean_diff'] * 100:+.1f}$ [${r['ci_low'] * 100:+.1f}$, ${r['ci_high'] * 100:+.1f}$] & "
                    f"{r['wins']}/{r['ties']}/{r['losses']} & {p} \\\\")
    write("stats", "\\begin{tabular}{llcccccc}\n\\toprule\n"
          "& & \\multicolumn{2}{c}{Dataset level (inferential)} & \\multicolumn{4}{c}{Setting level (descriptive)} \\\\\n"
          "\\cmidrule(lr){3-4}\\cmidrule(lr){5-8}\n"
          "& Comparison & Datasets $>0$ & $p_{\\mathrm{sign}}$ & $n$ & Mean diff. (pts) [95\\% CI] & W/T/L & $p_{\\mathrm{Holm}}$ \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def main():
    runs = collect()
    datasets_table()
    teachers_table(runs)
    methods_table(runs, "mlp_h16", "macro_f1", "methods_small")
    methods_table(runs, "mlp_h256", "macro_f1", "methods_large")
    balanced_table(runs)
    quant_table()
    edge_table()
    cost_table()
    stats_table()


if __name__ == "__main__":
    main()
