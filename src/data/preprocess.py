"""Feature transform. Fitted on the training split only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def _signed_log1p(x):
    # Byte and packet counts span ~9 orders of magnitude.
    return np.sign(x) * np.log1p(np.abs(x))


def build(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(_signed_log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    parts = [("num", numeric, numeric_cols)]
    if categorical_cols:
        parts.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                           min_frequency=10, sparse_output=False),
                      categorical_cols))
    return ColumnTransformer(parts)


def fit_transform(splits: dict[str, pd.DataFrame], numeric_cols, categorical_cols):
    ct = build(numeric_cols, categorical_cols)
    ct.fit(splits["train"])
    out = {k: ct.transform(v).astype(np.float32) for k, v in splits.items()}
    return out, list(ct.get_feature_names_out())
