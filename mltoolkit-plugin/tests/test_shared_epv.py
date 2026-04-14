"""epv.py — minority-class prevalence and events-per-variable checks."""
import numpy as np
import pandas as pd

from references._shared.epv import audit_epv


def test_balanced_binary_passes_checks():
    y = pd.Series([0, 1] * 500)
    X = pd.DataFrame(np.zeros((1000, 10)))
    a = audit_epv(X, y)
    assert a["minority_prevalence"] == 0.5
    assert a["n_features"] == 10
    assert a["epv"] == 500 / 10
    assert not a["low_epv_warning"]
    assert not a["rare_outcome_warning"]


def test_rare_outcome_triggers_warning():
    y = pd.Series([0] * 990 + [1] * 10)
    X = pd.DataFrame(np.zeros((1000, 5)))
    a = audit_epv(X, y)
    assert a["minority_prevalence"] == 0.01
    assert a["rare_outcome_warning"] is True


def test_low_epv_triggers_warning():
    y = pd.Series([0] * 85 + [1] * 15)
    X = pd.DataFrame(np.zeros((100, 20)))
    a = audit_epv(X, y)
    assert a["epv"] < 10
    assert a["low_epv_warning"] is True
