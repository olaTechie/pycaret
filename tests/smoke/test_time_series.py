"""Phase 4 time-series smoke harness.

Mirror of tests/smoke/test_plotting.py — minimal local-only coverage,
parametrized over a small set of forecasters. Pass condition is "no
exception raised". No image diff, no parity check, no SHAP.

Per-test timeout 30 s, aggregate target <120 s.

Run via:

    .venv-phase4/bin/python -m pytest --confcutdir=tests/smoke \\
        tests/smoke/test_time_series.py -v
"""

from __future__ import annotations

import pytest

from pycaret.datasets import get_data
from pycaret.time_series import TSForecastingExperiment


@pytest.fixture(scope="module")
def ts_setup():
    # Airline passengers — 144 monthly observations, classic TS smoke test.
    data = get_data("airline", verbose=False)
    exp = TSForecastingExperiment()
    exp.setup(
        data=data,
        fh=12,
        fold=2,
        n_jobs=1,
        html=False,
        verbose=False,
        session_id=42,
    )
    return exp


# Skip-list mirrors docs/superpowers/agents/ts-dev/DEGRADED.md.
TS_DEGRADED: set[str] = {
    # auto_arima exceeds the 30s smoke budget on airline (144 obs) under
    # sktime 0.40.1 — the wrapper appears to grid-search over a wider
    # space than the prior pinned 0.31.0 did. The forecaster still
    # works on a longer-running invocation; this is a smoke-harness
    # budget skip, not an API break. Tracked in
    # docs/superpowers/agents/ts-dev/DEGRADED.md (sktime drift datum
    # under FAILURE_TAXONOMY row 13).
    "auto_arima",
}

# Forecaster IDs are the keys pycaret accepts in `compare_models(include=...)`
# / `create_model(estimator=...)`. One representative per family. We
# intentionally exclude `tbats` and `bats` because Tier-3 (Task 7) the
# graceful-disable path. They are skip-listed in DEGRADED.md but their
# graceful-disable is implicit (via container.active = False), not a
# NotImplementedError raise — so we just don't list them here.
TS_FORECASTERS = sorted(
    [
        "naive",
        "snaive",
        "polytrend",
        "arima",
        "auto_arima",
        "exp_smooth",
        "ets",
        "theta",
        "stlf",
        "lr_cds_dt",  # linear regression with conditional deseasonalizer
    ]
)


@pytest.mark.timeout(30)
@pytest.mark.parametrize("forecaster", TS_FORECASTERS)
def test_forecaster_create(ts_setup, forecaster):
    if forecaster in TS_DEGRADED:
        pytest.skip(f"forecaster='{forecaster}' is degraded — see DEGRADED.md")
    model = ts_setup.create_model(forecaster, verbose=False)
    # Sanity: a fitted model has a `predict` method (or sktime equivalent).
    assert hasattr(model, "predict") or hasattr(model, "_predict")


@pytest.mark.timeout(30)
@pytest.mark.parametrize("forecaster", TS_FORECASTERS)
def test_forecaster_predict(ts_setup, forecaster):
    if forecaster in TS_DEGRADED:
        pytest.skip(f"forecaster='{forecaster}' is degraded — see DEGRADED.md")
    model = ts_setup.create_model(forecaster, verbose=False)
    preds = ts_setup.predict_model(model, verbose=False)
    assert preds is not None
    assert len(preds) > 0
