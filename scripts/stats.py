"""Statistical tests for the paper's claims (E9).

Settings (dataset x size [x method]) are averaged over seeds, so seeds are
never counted as independent. Settings within one dataset still share data,
split and teacher, so the INFERENTIAL unit is the dataset: each claim gets a
dataset-level exact sign test (n = 5, smallest attainable p = 0.0625). The
setting-level paired Wilcoxon (Holm-corrected), rank-biserial effect size
and bootstrap 95% CI are reported as descriptive statistics.

    python scripts/stats.py   -> results/stats.csv (+ markdown on stdout)
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.summarize import collect  # noqa: E402

SMALL = ("mlp_h8", "mlp_h16", "mlp_h32")         # 0.5k - 4.4k params
LARGE = ("mlp_h128", "mlp_h256", "mlp_h512")     # 16k - 194k params
BOOT, RNG = 10_000, np.random.default_rng(0)


def paired(df, unit, key, a, b, metric) -> pd.DataFrame:
    """Seed-averaged values of variant a and b, aligned on the unit columns
    (unit[0] must be "dataset"). b may be a tuple: the per-setting best of
    those variants (e.g. the strongest non-distillation baseline)."""
    m = df.groupby(unit + [key])[metric].mean().unstack(key)
    bs = list(b) if isinstance(b, tuple) else [b]
    out = pd.DataFrame({"a": m[a], "b": m[bs].max(axis=1)}).dropna()
    return out


def test(name, claim, pair, metric):
    """Setting-level paired Wilcoxon (descriptive: settings within a dataset
    share data, split and teacher) plus the dataset-level sign test, which is
    the inferential unit: one mean difference per dataset."""
    x, y = pair["a"].to_numpy(), pair["b"].to_numpy()
    diff = x - y
    nz = diff[diff != 0]
    stat = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    # rank-biserial correlation from the signed ranks of the non-zero diffs
    ranks = pd.Series(np.abs(nz)).rank().to_numpy()
    r = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum() if len(nz) else 0.0
    boot = RNG.choice(diff, (BOOT, len(diff))).mean(1)
    per_ds = (pair["a"] - pair["b"]).groupby(level="dataset").mean()
    n_pos, n_ds = int((per_ds > 0).sum()), len(per_ds)
    return {"id": name, "claim": claim, "metric": metric, "n": len(diff),
            "mean_diff": diff.mean(), "ci_low": np.percentile(boot, 2.5),
            "ci_high": np.percentile(boot, 97.5), "median_diff": np.median(diff),
            "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()),
            "losses": int((diff < 0).sum()), "rank_biserial": r, "p": stat.pvalue,
            "datasets_pos": n_pos, "datasets": n_ds,
            "p_sign_datasets": binomtest(n_pos, n_ds).pvalue,
            "per_dataset": "; ".join(f"{k}={v * 100:+.1f}" for k, v in per_ds.items())}


def holm(p):
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - k) * p[i]))
        adj[i] = running
    return adj


def main():
    runs = collect()
    q = pd.concat([pd.read_csv(f) for f in glob.glob(str(config.RESULTS_DIR / "quant" / "*.csv"))
                   if not f.endswith("_fpin.csv")])
    f = pd.concat([pd.read_csv(f) for f in glob.glob(str(config.RESULTS_DIR / "quant" / "*_fpin.csv"))])
    for d in (q, f):   # setting = dataset x architecture x training method, seed stripped
        d["setting"] = d["tag"].str.replace(r"_s\d+$", "", regex=True)

    out = []
    small = runs[runs["model"].isin(SMALL)]
    large = runs[runs["model"].isin(LARGE)]
    u = ["dataset", "model"]
    sizes = runs[runs["model"].isin(SMALL + ("mlp_h64",) + LARGE)]
    # H1 is the confounded comparison of the first draft; H1b/H1c separate the
    # imbalance term from the teacher.
    out.append(test("H1", "RAP-KD vs CE, small students", paired(small, u, "loss", "rap_kd", "ce", "macro_f1"), "macro_f1"))
    out.append(test("H1b", "RAP-KD vs best of WCE/LA (no teacher), small", paired(small, u, "loss", "rap_kd", ("wce", "la"), "macro_f1"), "macro_f1"))
    out.append(test("H1c", "plain KD vs CE, small students", paired(small, u, "loss", "kd", "ce", "macro_f1"), "macro_f1"))
    out.append(test("H2", "RAP-KD vs CE, large students", paired(large, u, "loss", "rap_kd", "ce", "macro_f1"), "macro_f1"))
    out.append(test("H2b", "RAP-KD vs KD+WCE, all sizes", paired(sizes, u, "loss", "rap_kd", "kd_wce", "macro_f1"), "macro_f1"))
    bt = runs[runs["model"].isin(("mlp_h16", "mlp_h64", "mlp_h256"))]
    out.append(test("H3", "balanced vs CE teacher (KD+WCE), rare recall", paired(bt, u, "loss", "kd_wce_bt", "kd_wce", "rare_recall"), "rare_recall"))
    out.append(test("H4", "balanced vs CE teacher (KD+WCE), macro-F1", paired(bt, u, "loss", "kd_wce_bt", "kd_wce", "macro_f1"), "macro_f1"))
    uq = ["dataset", "setting"]
    out.append(test("H5", "balanced vs random calibration, rare recall (test)", paired(q, uq, "variant", "int8_bal", "int8_rand", "rare_recall"), "rare_recall"))
    fv = f[f["split"] == "val"]
    out.append(test("H6", "FP32 input layer vs full INT8, macro-F1 (val)", paired(fv, uq, "variant", "int8_bal_fpin", "int8_bal", "macro_f1"), "macro_f1"))
    out.append(test("H7", "FP32 input layer vs full INT8, rare recall (val)", paired(fv, uq, "variant", "int8_bal_fpin", "int8_bal", "rare_recall"), "rare_recall"))

    res = pd.DataFrame(out)
    res["p_holm"] = holm(res["p"].to_numpy())
    res.to_csv(config.RESULTS_DIR / "stats.csv", index=False)
    print("| id | claim | n | mean diff [95% CI] | W/T/L | p (Holm) | datasets + | p sign |")
    print("|---|---|---|---|---|---|---|---|")
    for _, r in res.iterrows():
        print(f"| {r.id} | {r.claim} ({r.metric}) | {r.n} | {r.mean_diff:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}] "
              f"| {r.wins}/{r.ties}/{r.losses} | {r.p_holm:.2g} | {r.datasets_pos}/{r.datasets} | {r.p_sign_datasets:.3f} |")
        print(f"|   | per dataset: {r.per_dataset} |||||||")


if __name__ == "__main__":
    main()
