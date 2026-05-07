"""Phase 3 plotting smoke harness.

Minimal local-only coverage: one estimator per task type, parametrized
over each `experiment._available_plots.keys()`. Pass condition is "no
exception raised" — no image diff, no SHAP, no interpret_model.

Per-plot timeout 30 s, aggregate target <90 s on a developer laptop.

This file is NOT discovered by `pytest tests/` (it lives under
`tests/smoke/` and the root `tests/conftest.py` is bypassed via
`--confcutdir=tests/smoke`). Run explicitly:

    .venv-phase3/bin/python -m pytest --confcutdir=tests/smoke \\
        tests/smoke/test_plotting.py -v
"""

from __future__ import annotations

import pytest

from pycaret.classification import ClassificationExperiment
from pycaret.clustering import ClusteringExperiment
from pycaret.datasets import get_data
from pycaret.regression import RegressionExperiment

# --- Classification ----------------------------------------------------


@pytest.fixture(scope="module")
def clf_setup():
    # Binary classification — needed because several plot_model entries
    # (gain, ks, lift, threshold, manifold, rfe) are blocked by pycaret's
    # multiclass_not_available guard or by scikitplot's binary-only check.
    # juice is what tests/test_classification_plots.py already uses.
    data = get_data("juice", verbose=False)
    exp = ClassificationExperiment()
    exp.setup(
        data=data,
        target="Purchase",
        session_id=42,
        fold=2,
        n_jobs=1,
        html=False,
        verbose=False,
    )
    model = exp.create_model("rf", n_estimators=5, max_depth=2, verbose=False)
    return exp, model


# Skip-list mirrors DEGRADED.md; entries added as Phase 3 fixes uncover
# unfixable visualizers under fallback policy (b).
CLF_DEGRADED: set[str] = {
    # DEGRADED row: yellowbrick 1.5's ClassPredictionError unpacks a
    # 3-tuple from a sklearn helper whose shape changed under sklearn 1.6.
    "error",
}

CLF_PLOTS = sorted(
    [
        "pipeline",
        "parameter",
        "auc",
        "confusion_matrix",
        "threshold",
        "pr",
        "error",
        "class_report",
        "rfe",
        "learning",
        "manifold",
        "calibration",
        "vc",
        "dimension",
        "feature",
        "feature_all",
        "boundary",
        "lift",
        "gain",
        "tree",
        "ks",
    ]
)


@pytest.mark.timeout(30)
@pytest.mark.parametrize("plot", CLF_PLOTS)
def test_classification_plot(clf_setup, plot, tmp_path):
    if plot in CLF_DEGRADED:
        pytest.skip(f"plot='{plot}' is degraded — see DEGRADED.md")
    exp, model = clf_setup
    exp.plot_model(model, plot=plot, save=str(tmp_path), verbose=False)


# --- Regression --------------------------------------------------------


@pytest.fixture(scope="module")
def reg_setup():
    data = get_data("diabetes", verbose=False)
    target = "Class variable" if "Class variable" in data.columns else data.columns[-1]
    exp = RegressionExperiment()
    exp.setup(
        data=data,
        target=target,
        session_id=42,
        fold=2,
        n_jobs=1,
        html=False,
        verbose=False,
    )
    model = exp.create_model("rf", n_estimators=5, max_depth=2, verbose=False)
    return exp, model


REG_DEGRADED: set[str] = {
    # Row 23 (closed): requires `anywidget` (transitive plotly
    # interactive-widget dep, not in pycaret-ng base deps). Lazy-import
    # guard at tabular_experiment.py:residuals_interactive raises
    # NotImplementedError with a clear `pip install anywidget` hint.
    # Smoke skip stays — the harness intentionally doesn't install
    # anywidget to verify the degrade path keeps user errors loud.
    "residuals_interactive",
}

REG_PLOTS = sorted(
    [
        "pipeline",
        "parameter",
        "residuals",
        "error",
        "cooks",
        "rfe",
        "learning",
        "manifold",
        "vc",
        "feature",
        "feature_all",
        "tree",
        "residuals_interactive",
    ]
)


@pytest.mark.timeout(30)
@pytest.mark.parametrize("plot", REG_PLOTS)
def test_regression_plot(reg_setup, plot, tmp_path):
    if plot in REG_DEGRADED:
        pytest.skip(f"plot='{plot}' is degraded — see DEGRADED.md")
    exp, model = reg_setup
    exp.plot_model(model, plot=plot, save=str(tmp_path), verbose=False)


# --- Clustering --------------------------------------------------------


@pytest.fixture(scope="module")
def clu_setup():
    data = get_data("iris", verbose=False).drop(columns=["species"])
    exp = ClusteringExperiment()
    exp.setup(
        data=data,
        session_id=42,
        n_jobs=1,
        html=False,
        verbose=False,
    )
    model = exp.create_model("kmeans", num_clusters=3, verbose=False)
    return exp, model


CLU_DEGRADED: set[str] = {
    # DEGRADED row: yellowbrick's InterclusterDistance still passes the
    # removed `interpolation=` kwarg to np.percentile() (numpy 2 drop).
    "distance",
}

CLU_PLOTS = sorted(
    [
        "pipeline",
        "cluster",
        "tsne",
        "elbow",
        "silhouette",
        "distance",
        "distribution",
    ]
)


@pytest.mark.timeout(30)
@pytest.mark.parametrize("plot", CLU_PLOTS)
def test_clustering_plot(clu_setup, plot, tmp_path):
    if plot in CLU_DEGRADED:
        pytest.skip(f"plot='{plot}' is degraded — see DEGRADED.md")
    exp, model = clu_setup
    exp.plot_model(model, plot=plot, save=str(tmp_path), verbose=False)
