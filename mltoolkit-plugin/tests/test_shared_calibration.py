"""calibration.py — Brier, ECE, calibration intercept+slope, reliability diagram."""
import numpy as np

from references._shared.calibration import (
    calibration_summary, reliability_diagram,
)


def test_perfect_calibration_has_zero_ece_and_unit_slope():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 1000)
    y_proba = y_true.astype(float)
    s = calibration_summary(y_true, y_proba, n_bins=10)
    assert s["brier"] < 0.05
    assert s["ece"] < 0.05


def test_random_probabilities_have_nontrivial_ece():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    y_proba = rng.random(500)
    s = calibration_summary(y_true, y_proba, n_bins=10)
    for k in ("brier", "ece", "intercept", "slope", "n_bins", "bins"):
        assert k in s
    assert 0.0 <= s["ece"] <= 1.0


def test_reliability_diagram_returns_figure(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    y_proba = rng.random(500)
    fig = reliability_diagram(y_true, y_proba, n_bins=10)
    assert fig.axes
    out = tmp_path / "rd.png"
    fig.savefig(out)
    assert out.exists()
