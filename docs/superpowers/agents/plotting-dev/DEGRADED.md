# DEGRADED.md — Plotting Visualizer Registry

Visualizers that pycaret-ng explicitly disables under modernized deps. A
disabled visualizer raises `NotImplementedError` from its dispatch site
in `tabular_experiment.py` or `clustering/oop.py`. The corresponding
smoke entry in `tests/smoke/test_plotting.py` is skip-marked.

## Schema

| Plot key | Task | Disabled because | Tracking | Restoration criterion |
|----------|------|-------------------|----------|------------------------|

## Rows

| Plot key | Task | Disabled because | Tracking | Restoration criterion |
|----------|------|-------------------|----------|------------------------|
| `error` | classification | `yellowbrick.classifier.ClassPredictionError.__init__` (yellowbrick 1.5) unpacks a 3-tuple from a sklearn helper that returns a different shape under sklearn ≥1.6. The error originates inside yellowbrick, not pycaret. | FAILURE_TAXONOMY row 24; upstream yellowbrick issue | yellowbrick releases a sklearn-1.6-aware version OR pycaret monkey-patches the unpack site |
| `distance` | clustering | `yellowbrick.cluster.InterclusterDistance` still passes the removed `interpolation=` kwarg to `np.percentile()`, which numpy ≥2 dropped (replaced with `method=`). | FAILURE_TAXONOMY row 25; upstream yellowbrick issue | yellowbrick releases a numpy-2-aware version OR pycaret monkey-patches `np.percentile` for the call duration |
| `residuals_interactive` | regression | Requires the optional `anywidget` package (transitive plotly interactive-widget dep under modern Jupyter). Pycaret-ng does not include it in base deps. | FAILURE_TAXONOMY row 23 | User installs `anywidget` (`pip install anywidget`) — runtime guard at `tabular_experiment.py:residuals_interactive` lifts automatically. Alternative: use `plot='residuals'` (static). |
