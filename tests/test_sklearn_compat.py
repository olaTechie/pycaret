"""Unit tests for pycaret.utils._sklearn_compat."""

from __future__ import annotations

import pytest


def test_get_base_scorer_class_returns_a_class():
    from pycaret.utils._sklearn_compat import get_base_scorer_class

    cls = get_base_scorer_class()
    assert isinstance(cls, type), f"Expected a class, got {type(cls)}"


def test_get_base_scorer_class_is_sklearn_base_scorer():
    """Round-trip: a sklearn make_scorer() result must be an instance of the returned class."""
    from sklearn.metrics import make_scorer
    from pycaret.utils._sklearn_compat import get_base_scorer_class

    scorer = make_scorer(lambda y_true, y_pred: 0.0)
    cls = get_base_scorer_class()
    assert isinstance(scorer, cls), (
        f"make_scorer() returned {type(scorer).__mro__}; " f"shim returned {cls}"
    )


def test_get_base_scorer_class_is_cached():
    """Shim should not re-import on every call."""
    from pycaret.utils._sklearn_compat import get_base_scorer_class

    a = get_base_scorer_class()
    b = get_base_scorer_class()
    assert a is b


def test_get_check_reg_targets_returns_a_callable():
    from pycaret.utils._sklearn_compat import get_check_reg_targets

    fn = get_check_reg_targets()
    assert callable(fn)


def test_get_check_reg_targets_round_trip():
    """Calling the function on simple regression vectors should not raise."""
    import numpy as np
    from pycaret.utils._sklearn_compat import get_check_reg_targets

    fn = get_check_reg_targets()
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    # sklearn >=1.8 made sample_weight positionally required; pre-1.8 accepted it as kw-only.
    # Passing it explicitly keeps the call valid across the supported range (1.5-1.x).
    result = fn(y_true, y_pred, sample_weight=None, multioutput="uniform_average")
    # Signature varies subtly across sklearn versions but always returns >= 3 items.
    assert len(result) >= 3
