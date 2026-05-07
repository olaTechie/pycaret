"""Reference dataset loaders for the parity harness.

Each loader returns (X: DataFrame, y: Series, task: str) deterministically
so parity comparisons against frozen 3.4.0 baselines are reproducible.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes, load_iris

RANDOM_STATE = 42


def _load_iris() -> Tuple[pd.DataFrame, pd.Series, str]:
    data = load_iris(as_frame=True)
    return data.data, data.target, "classification"


def _load_diabetes() -> Tuple[pd.DataFrame, pd.Series, str]:
    data = load_diabetes(as_frame=True)
    return data.data, data.target, "regression"


def _load_california_housing() -> Tuple[pd.DataFrame, pd.Series, str]:
    data = fetch_california_housing(as_frame=True)
    return data.data, data.target, "regression"


def _load_credit() -> Tuple[pd.DataFrame, pd.Series, str]:
    from pycaret.datasets import get_data

    df = get_data("credit", verbose=False)
    y = df["default"].astype(int)
    X = df.drop(columns=["default"])
    return X, y, "classification"


def _load_airline_passengers() -> Tuple[pd.DataFrame, pd.Series, str]:
    from pycaret.datasets import get_data

    df = get_data("airline", verbose=False)
    if isinstance(df, pd.Series):
        y = df.astype(float)
    else:
        y = df.iloc[:, 0].astype(float)
    X = pd.DataFrame(index=y.index)
    return X, y, "time_series"


REFERENCE_DATASETS: Dict[str, Callable[[], Tuple[pd.DataFrame, pd.Series, str]]] = {
    "iris": _load_iris,
    "diabetes": _load_diabetes,
    "california_housing": _load_california_housing,
    "credit": _load_credit,
    "airline_passengers": _load_airline_passengers,
}


def load_reference(name: str) -> Tuple[pd.DataFrame, pd.Series, str]:
    if name not in REFERENCE_DATASETS:
        raise KeyError(
            f"Unknown reference dataset {name!r}. "
            f"Available: {sorted(REFERENCE_DATASETS)}"
        )
    return REFERENCE_DATASETS[name]()
