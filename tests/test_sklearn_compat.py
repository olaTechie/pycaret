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
        f"make_scorer() returned {type(scorer).__mro__}; "
        f"shim returned {cls}"
    )


def test_get_base_scorer_class_is_cached():
    """Shim should not re-import on every call."""
    from pycaret.utils._sklearn_compat import get_base_scorer_class

    a = get_base_scorer_class()
    b = get_base_scorer_class()
    assert a is b
