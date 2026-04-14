"""Cross-validation splitter routing.

    make_splitter(y, n_splits=5, groups=None, time_order=None) -> sklearn splitter

- time_order given  → TimeSeriesSplit (caller must pre-sort X/y by time).
- groups given      → StratifiedGroupKFold if y is categorical (nunique <= 20),
                      else GroupKFold.
- neither given     → StratifiedKFold if y is categorical, else KFold.
Raises ValueError if both groups and time_order are passed.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import (
    GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold, TimeSeriesSplit,
)

_CATEGORICAL_CARDINALITY = 20


def _is_categorical(y) -> bool:
    y = pd.Series(y)
    return y.dtype == "O" or y.nunique(dropna=True) <= _CATEGORICAL_CARDINALITY


def make_splitter(y, *, n_splits: int = 5, groups=None, time_order=None,
                  random_state: int = 42):
    if groups is not None and time_order is not None:
        raise ValueError("`groups` and `time_order` are mutually exclusive.")
    if time_order is not None:
        return TimeSeriesSplit(n_splits=n_splits)
    if groups is not None:
        if _is_categorical(y):
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=random_state)
        return GroupKFold(n_splits=n_splits)
    if _is_categorical(y):
        return StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
