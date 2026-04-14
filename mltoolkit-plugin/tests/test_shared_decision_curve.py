"""decision_curve.py — net-benefit vs threshold with treat-all/treat-none baselines."""
import numpy as np

from references._shared.decision_curve import (
    net_benefit_curve, decision_curve_figure,
)


def test_net_benefit_curve_includes_model_and_baselines():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    y_proba = rng.random(500)
    df = net_benefit_curve(y_true, y_proba, thresholds=np.linspace(0.05, 0.5, 10))
    for col in ("threshold", "model", "treat_all", "treat_none"):
        assert col in df.columns
    assert (df["treat_none"] == 0).all()


def test_decision_curve_figure_returns_axes(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    y_proba = rng.random(500)
    fig = decision_curve_figure(y_true, y_proba)
    assert fig.axes
    fig.savefig(tmp_path / "dc.png")
