"""splits.py — routing to Stratified / Group / TimeSeries splitters."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import (
    GroupKFold, StratifiedKFold, StratifiedGroupKFold, TimeSeriesSplit,
)

from references._shared.splits import make_splitter


def test_default_returns_stratifiedkfold():
    y = pd.Series([0, 1] * 100)
    s = make_splitter(y, n_splits=5)
    assert isinstance(s, StratifiedKFold)
    assert s.n_splits == 5


def test_group_col_returns_stratifiedgroupkfold_when_y_is_categorical():
    y = pd.Series([0, 1] * 50)
    groups = pd.Series(np.repeat(np.arange(20), 5))
    s = make_splitter(y, n_splits=5, groups=groups)
    assert isinstance(s, StratifiedGroupKFold)


def test_group_col_returns_groupkfold_when_y_is_continuous():
    y = pd.Series(np.random.default_rng(0).normal(size=100))
    groups = pd.Series(np.repeat(np.arange(20), 5))
    s = make_splitter(y, n_splits=5, groups=groups)
    assert isinstance(s, GroupKFold)


def test_time_col_returns_timeseries_split():
    y = pd.Series(np.random.default_rng(0).normal(size=100))
    times = pd.Series(pd.date_range("2024-01-01", periods=100))
    s = make_splitter(y, n_splits=5, time_order=times)
    assert isinstance(s, TimeSeriesSplit)
    assert s.n_splits == 5


def test_time_and_group_raises():
    y = pd.Series([0, 1] * 50)
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_splitter(
            y, groups=pd.Series([1] * 100),
            time_order=pd.Series(pd.date_range("2024", periods=100)),
        )
