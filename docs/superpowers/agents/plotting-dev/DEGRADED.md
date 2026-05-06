# DEGRADED.md — Plotting Visualizer Registry

Visualizers that pycaret-ng explicitly disables under modernized deps. A
disabled visualizer raises `NotImplementedError` from its dispatch site
in `tabular_experiment.py` or `clustering/oop.py`. The corresponding
smoke entry in `tests/smoke/test_plotting.py` is skip-marked.

## Schema

| Plot key | Task | Disabled because | Tracking | Restoration criterion |
|----------|------|-------------------|----------|------------------------|

## Rows

(none — populate as Phase 3 fixes uncover unfixable visualizers)
