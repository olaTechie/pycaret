"""Preprocessing pipeline builder for classification tasks."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Ensure _shared is importable regardless of CWD
_HERE = Path(__file__).resolve()
_PLUGIN_ROOT = _HERE.parents[3]
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

from references._shared import deps  # noqa: E402


CARDINALITY_THRESHOLD = 10


def _split_columns(df: pd.DataFrame):
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    low_card = [c for c in categorical if df[c].nunique(dropna=True) <= CARDINALITY_THRESHOLD]
    high_card = [c for c in categorical if df[c].nunique(dropna=True) > CARDINALITY_THRESHOLD]
    return numeric, low_card, high_card


def _make_imputers(kind: str):
    if kind == "simple" or kind == "drop":
        return SimpleImputer(strategy="median"), SimpleImputer(strategy="most_frequent")
    if kind == "iterative":
        return IterativeImputer(random_state=42), SimpleImputer(strategy="most_frequent")
    if kind == "knn":
        return KNNImputer(), SimpleImputer(strategy="most_frequent")
    raise ValueError(f"unknown imputation kind: {kind}")


def _safe_encoders_mod():
    try:
        from _shared.encoders_safe import is_sensitive_column, safe_high_cardinality_encoder
    except ImportError:
        from references._shared.encoders_safe import is_sensitive_column, safe_high_cardinality_encoder
    return is_sensitive_column, safe_high_cardinality_encoder


def build_preprocessor(
    df: pd.DataFrame,
    *,
    imputation: str = "simple",
    sensitive: Sequence[str] = (),
    allow_te_on_sensitive: bool = False,
) -> ColumnTransformer:
    """Build a ColumnTransformer tuned for the given DataFrame's schema.

    `sensitive` columns are never target-encoded unless
    `allow_te_on_sensitive=True`. If the high-cardinality group includes any
    sensitive column and the override is not set, build_preprocessor raises.
    """
    numeric, low_card, high_card = _split_columns(df)
    imp_num, imp_cat = _make_imputers(imputation)

    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([("imp", imp_num), ("scl", StandardScaler())]),
            numeric,
        ))
    if low_card:
        transformers.append((
            "cat_low",
            Pipeline([("imp", imp_cat),
                      ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
            low_card,
        ))
    if high_card:
        is_sensitive, _safe = _safe_encoders_mod()
        for col in high_card:
            if is_sensitive(col, sensitive) and not allow_te_on_sensitive:
                # Raises ValueError — callers pass --allow-target-encode-on-sensitive to override.
                _safe(col, sensitive, allow_target_encode_on_sensitive=False)
        if deps.has_category_encoders():
            import category_encoders as ce
            enc = ce.TargetEncoder()
        else:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        transformers.append((
            "cat_high",
            Pipeline([("imp", imp_cat), ("enc", enc)]),
            high_card,
        ))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
