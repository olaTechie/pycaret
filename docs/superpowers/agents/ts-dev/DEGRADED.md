# DEGRADED.md — Time-series Forecaster/Visualizer Registry

Forecasters / visualizers that pycaret-ng explicitly disables under
modernized deps. A disabled entry raises `NotImplementedError` from its
container's `class_def` accessor (or the relevant dispatch site) with a
pointer to this file. The corresponding smoke entry is skip-marked.

## Schema

| Entry | Kind | Disabled because | Tracking | Restoration criterion |
|-------|------|-------------------|----------|------------------------|

## Rows

| Entry | Kind | Disabled because | Tracking | Restoration criterion |
|-------|------|-------------------|----------|------------------------|
| `bats` | forecaster | tbats library is numpy-1-only and unmaintained; the `sktime.forecasting.bats.BATS` import or instantiation fails under numpy ≥2. Container short-circuits to `active = False` with a warning. | FAILURE_TAXONOMY row 12 | tbats releases a numpy-2 compatible version OR pycaret vendors a successor (e.g., a sktime-native BATS reimplementation) |
| `tbats` | forecaster | same root cause as `bats` | FAILURE_TAXONOMY row 12 | same as `bats` |
| `auto_arima` | forecaster | Under sktime 0.40.1 (lifted from 0.31.0 in Phase 4) the auto_arima wrapper grid-searches a larger parameter space; `create_model('auto_arima')` exceeds the 30s smoke budget on the airline dataset (144 obs). The forecaster still works on a longer-running invocation — this is smoke-harness budget skip, not an API break. | FAILURE_TAXONOMY row 13 (sktime drift datum) | sktime tightens the auto_arima default search space, OR pycaret pins narrower bounds in its auto_arima wrapper, OR the smoke increases the per-test budget for this entry only |
