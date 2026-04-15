import numpy as np
import pandas as pd
import pytest

from tests.parity.datasets import REFERENCE_DATASETS, load_reference


@pytest.mark.parametrize("name", list(REFERENCE_DATASETS.keys()))
def test_load_reference_returns_expected_shape(name):
    X, y, task = load_reference(name)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert len(X) > 0
    assert task in {"classification", "regression", "time_series"}


def test_reference_datasets_includes_all_five():
    assert set(REFERENCE_DATASETS.keys()) == {
        "iris",
        "diabetes",
        "california_housing",
        "credit",
        "airline_passengers",
    }
