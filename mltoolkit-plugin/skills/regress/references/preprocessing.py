"""Preprocessing pipeline for regression tasks (same feature prep as classify)."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from references._shared import deps  # noqa: E402


CARDINALITY_THRESHOLD = 10


def _split_columns(df: pd.DataFrame):
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    low_card = [c for c in categorical if df[c].nunique(dropna=True) <= CARDINALITY_THRESHOLD]
    high_card = [c for c in categorical if df[c].nunique(dropna=True) > CARDINALITY_THRESHOLD]
    return numeric, low_card, high_card


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric, low_card, high_card = _split_columns(df)
    transformers = []
    if numeric:
        transformers.append(("num",
            Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler())]),
            numeric))
    if low_card:
        transformers.append(("cat_low",
            Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
            low_card))
    if high_card:
        if deps.has_category_encoders():
            import category_encoders as ce
            enc = ce.TargetEncoder()
        else:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        transformers.append(("cat_high",
            Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", enc)]),
            high_card))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
