"""Optional-dependency detection helpers.

Each ``has_*`` function returns a bool without raising. Reference scripts
use these to gracefully skip models/features when their packages are missing.
"""
import importlib


def _check(pkg: str) -> bool:
    try:
        importlib.import_module(pkg)
        return True
    except ImportError:
        return False


def has_sklearn() -> bool:
    return _check("sklearn")


def has_xgboost() -> bool:
    return _check("xgboost")


def has_lightgbm() -> bool:
    return _check("lightgbm")


def has_catboost() -> bool:
    return _check("catboost")


def has_imblearn() -> bool:
    return _check("imblearn")


def has_category_encoders() -> bool:
    return _check("category_encoders")


def has_optuna() -> bool:
    return _check("optuna")


def has_plotly() -> bool:
    return _check("plotly")
