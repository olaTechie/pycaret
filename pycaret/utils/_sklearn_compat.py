"""scikit-learn version-compat shim for pycaret-ng.

Centralises every private/relocated sklearn symbol pycaret depends on.
When sklearn moves a symbol, fix one site here instead of dozens of call
sites. Inspired by sktime PR #8546's dual-API tag inspection helper.

Targets sklearn>=1.5,<2. Intentionally tolerant of 1.6/1.7 internal moves.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Type


@lru_cache(maxsize=1)
def get_base_scorer_class() -> Type:
    """Return sklearn's `_BaseScorer` class regardless of internal location.

    sklearn 1.5/1.6/1.7 keep it at sklearn.metrics._scorer._BaseScorer.
    If a future sklearn relocates it, extend the fallback chain here.
    """
    try:
        from sklearn.metrics._scorer import _BaseScorer
        return _BaseScorer
    except ImportError as e:
        raise ImportError(
            "pycaret-ng could not locate sklearn's _BaseScorer. "
            "Tried sklearn.metrics._scorer._BaseScorer. "
            "Add a fallback path in pycaret/utils/_sklearn_compat.py."
        ) from e


@lru_cache(maxsize=1)
def get_check_reg_targets():
    """Return sklearn's `_check_reg_targets` function regardless of location.

    sklearn 1.5/1.6/1.7 keep it at sklearn.metrics._regression._check_reg_targets.
    Add fallback paths here if a future sklearn relocates it.
    """
    try:
        from sklearn.metrics._regression import _check_reg_targets
        return _check_reg_targets
    except ImportError as e:
        raise ImportError(
            "pycaret-ng could not locate sklearn's _check_reg_targets. "
            "Tried sklearn.metrics._regression._check_reg_targets. "
            "Add a fallback path in pycaret/utils/_sklearn_compat.py."
        ) from e
