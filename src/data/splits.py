"""De-duplication, stratified splitting, and leakage checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def _row_hash(df: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(df, index=False)


def deduplicate(df: pd.DataFrame, feature_cols: list) -> tuple[pd.DataFrame, dict]:
    """Drops exact duplicates, then drops every row whose feature vector occurs
    with more than one label (these cannot be classified and would otherwise
    leak across splits with contradictory targets)."""
    n0 = len(df)
    df = df.drop_duplicates(subset=feature_cols + ["y"])
    n1 = len(df)
    h = _row_hash(df[feature_cols])
    n_labels = df.groupby(h.values)["y"].transform("nunique")
    df = df[n_labels.values == 1]
    stats = {"raw_rows": n0, "exact_duplicates_removed": n0 - n1,
             "conflicting_label_rows_removed": n1 - len(df), "rows_after": len(df)}
    return df.reset_index(drop=True), stats


def stratified_split(df: pd.DataFrame, seed: int = 0) -> dict[str, pd.DataFrame]:
    s = config.SPLIT
    train, rest = train_test_split(df, train_size=s["train"], stratify=df["y"], random_state=seed)
    val_frac = s["val"] / (s["val"] + s["test"])
    val, test = train_test_split(rest, train_size=val_frac, stratify=rest["y"], random_state=seed)
    return {"train": train, "val": val, "test": test}


def leakage_checks(splits: dict[str, pd.DataFrame], feature_cols: list, n_classes: int) -> dict:
    """Asserts no feature row is shared between splits and every class is in
    train. Raises instead of warning, so a leaky split can never be cached."""
    hashes = {k: set(_row_hash(v[feature_cols])) for k, v in splits.items()}
    overlap = {f"{a}&{b}": len(hashes[a] & hashes[b])
               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(overlap.values()):
        raise AssertionError(f"feature rows shared across splits: {overlap}")
    missing = set(range(n_classes)) - set(splits["train"]["y"].unique())
    if missing:
        raise AssertionError(f"classes absent from train: {sorted(missing)}")
    counts = {k: np.bincount(v["y"], minlength=n_classes).tolist() for k, v in splits.items()}
    return {"overlap": overlap, "class_counts": counts}
