# Migrating to pycaret-ng 1.0.0

`pycaret-ng` is a soft-fork of [PyCaret 3.4.0](https://github.com/pycaret/pycaret) that resumes modernization on Python 3.10–3.13, modern scikit-learn / pandas / numpy, and the current sktime / pmdarima / matplotlib / yellowbrick. **The internal import path stays `pycaret`** so existing user code continues to run unchanged.

## TL;DR

```bash
pip uninstall pycaret
pip install pycaret-ng
```

```python
import pycaret  # unchanged
```

Same API. Modernized dependency floors. A short list of known-degraded plot/forecaster entries (see § 5 below).

## Why pycaret-ng

Upstream `pycaret==3.4.0` capped scikit-learn at `<1.5`, pandas at `<2.2`, and numpy at `<1.27`. Modernization across those caps stalled in upstream during 2024–2025. pycaret-ng resumes that work in five phases — sklearn (Phase 1), pandas+numpy (Phase 2), plotting (Phase 3), time-series (Phase 4), release (Phase 5) — with cherry-pickable commits where possible (Gate D), so upstream can adopt phases as their own roadmap permits.

## Install

| What you want | Command |
|---------------|---------|
| Base | `pip install pycaret-ng` |
| All extras | `pip install pycaret-ng[full]` |
| Time-series only | `pip install pycaret-ng[time_series]` |
| MLOps extras | `pip install pycaret-ng[mlops]` |
| Test extras | `pip install pycaret-ng[test]` |

The optional extras list mirrors upstream pycaret 3.4.0 — same names, same contents minus the modernization-blocked deps. See `pyproject.toml` for the authoritative extras table.

## Modernized dependency floors

| Package | Before (pycaret 3.4.0) | After (pycaret-ng 1.0.0) | Phase that lifted it |
|---------|------------------------|--------------------------|----------------------|
| Python | `>=3.9,<3.13` | `>=3.10` (3.10–3.13 in CI) | Phase 5 — 3.9 dropped in round 6 release-hygiene because imbalanced-learn ≥0.14 (required by sklearn 1.6) needs Python ≥3.10. |
| scikit-learn | `<1.5` | `>=1.6,<2` | Phase 1 |
| imbalanced-learn | `>=0.12,<0.14` | `>=0.14,<0.15` | Phase 1 |
| category-encoders | `>=2.4` | `>=2.7,<3` | Phase 1 |
| pandas | `<2.2` | `>=2.2,<3` | Phase 2 |
| numpy | `>=1.21,<1.27` | `>=1.26,<3` | Phase 2 |
| matplotlib | `<3.8` | `>=3.8` | Phase 3 |
| schemdraw | `==0.15` | `>=0.16` | Phase 3 |
| sktime | `>=0.31.0,<0.31.1` | `>=0.31` | Phase 4 |
| statsmodels | `>=0.12.1` | `>=0.14,<1` | Phase 4 |
| pmdarima | `>=2.0.4` | `>=2.0.4` (unchanged; latent shim deferred) | Phase 4 |
| tbats | `>=1.1.3` | `>=1.1.3` (graceful-disable on numpy ≥2) | Phase 4 |
| yellowbrick | `>=1.4` | `>=1.4` (with pycaret-side patches) | Phase 3 |
| joblib | `>=1.4.2,<1.5` | `>=1.4.2,<1.5` (held; row 14 deferred) | — |

Tested combinations in CI: Python `{3.10, 3.11, 3.12, 3.13}` × OS `{linux, macos, windows}` against the floors above.

**Note on Python 3.9:** dropped in v1.0.0. Phase 1's sklearn ≥1.6 floor requires `imbalanced-learn>=0.14`, which itself requires Python ≥3.10. Older `imbalanced-learn` versions cap sklearn at `<1.5`, so Python 3.9 cannot satisfy both constraints simultaneously. Users on Python 3.9 should stay on upstream `pycaret==3.4.0` until they can upgrade their interpreter.

## Known limitations under modern deps

These are documented in the per-phase DEGRADED registries:

- `docs/superpowers/agents/plotting-dev/DEGRADED.md` (Phase 3)
- `docs/superpowers/agents/ts-dev/DEGRADED.md` (Phase 4)

User-facing summary:

### plot_model

| Plot | Task | Reason | Workaround |
|------|------|--------|------------|
| `error` | classification | yellowbrick's `ClassPredictionError` unpacks a 3-tuple from a sklearn helper whose return shape changed under sklearn ≥1.6. Internal to yellowbrick. | Use `plot="confusion_matrix"` or `plot="class_report"` instead. |
| `distance` | clustering | yellowbrick's `InterclusterDistance` calls `np.percentile(..., interpolation=...)` which numpy ≥2 removed. | Use `plot="silhouette"` or `plot="elbow"` instead. |
| `residuals_interactive` | regression | requires the optional `anywidget` package (transitive plotly interactive-widget dep). pycaret-ng raises `NotImplementedError` with a `pip install anywidget` hint when the dep is missing — fail-loud, not silent. | Either `pip install anywidget` (the runtime guard lifts automatically) or use `plot="residuals"` (static). |

The disabled visualizers raise `NotImplementedError` with a pointer to `docs/superpowers/agents/plotting-dev/DEGRADED.md` — they fail loudly, not silently.

### Time-series forecasters

| Forecaster | Reason | Workaround |
|------------|--------|------------|
| `bats`, `tbats` | `tbats` is unmaintained and numpy-1-only. Container deactivates with a logger warning when the import fails under numpy ≥2. | Use `auto_arima`, `exp_smooth`, or `theta` instead. |
| `auto_arima` (smoke skip only) | Default search space wider under sktime 0.40.1; smoke harness skip-listed it because `create_model('auto_arima')` exceeds 30 s on the airline dataset. **Real workloads are unaffected.** | None needed for production. The forecaster runs fine; the smoke skip is a budget thing. |

### Latent (no-op today)

- pmdarima still calls sklearn's `force_all_finite=` kwarg internally. sklearn 1.6+ deprecated it; sklearn 1.8 removes it entirely. sktime 0.40.1 caps sklearn `<1.8`, so the breakage is currently latent. A defensive shim is sketched in the Phase 4 plan and will activate when sklearn 1.8 becomes installable.
- joblib's `Memory.bytes_limit` API change at joblib 1.5 — pinned `<1.5` for now; revives when downstream caps lift.

## Cherry-pick provenance

For users / forks tracking modernization commits independently:

| Phase | Commit prefix | Cherry-pick clean? |
|-------|---------------|---------------------|
| Phase 1 (sklearn) | `fix(sklearn)`, `feat(sklearn)` | yes |
| Phase 2 (pandas + numpy) | `fix(pandas)`, `feat(pandas)`, `fix(numpy)`, `feat(numpy)` | yes |
| Phase 3 (plotting) | `fix(plot)`, `feat(plot)` | yes (one mixed-scope commit noted in PR #3) |
| Phase 4 (time-series) | `fix(ts)`, `feat(ts)`, `fix(sklearn)` (force_all_finite rename) | yes |
| Phase 5 (release) | `fix(release)` (Python guard, soft-dep gating) | partial |

Phase 5's `feat(release)` commits (the rename + version bump + publish workflow) are pycaret-ng-specific and intentionally not upstream-cherry-pickable — they only make sense in this fork's context. The `fix(release)` commits (Python guard widening, soft-dep test gating) DO apply to upstream.

The full per-phase commit list lives in each phase's spec at `docs/superpowers/specs/`. Each `fix(*)` / `feat(*)` commit on Phases 1–4 was certified to apply onto upstream `pycaret/pycaret:master` (Gate D). Upstream PRs against `pycaret/pycaret` are not part of pycaret-ng's release process; opening them is a separate user-driven activity if desired.

## Roadmap

pycaret-ng v1.0.0 is the modernization-complete release. The master spec at `docs/superpowers/specs/2026-04-15-pycaret-ng-modernization-design.md` describes v1.1.0+ feature work (LLM phases 6.0–6.4): conversational SDK, EDA advisor, auto reports, LLM zoo estimators, MCP server. None of those land in v1.0.x.

Punch list for v1.0.x patch releases (not blocking 1.0.0):
- `anywidget` for `residuals_interactive` (row 23).
- joblib `Memory.bytes_limit` migration when joblib 1.5+ becomes installable (row 14).
- pmdarima `force_all_finite` shim activation when sklearn 1.8 reachable (Phase 4 latent).

## Compatibility commitment

- Existing user code that does `from pycaret.classification import setup, compare_models, ...` continues to work unchanged.
- Public API surface is unchanged from upstream 3.4.0 except for the documented degraded entries above (each disabled entry raises `NotImplementedError` with a pointer to the registry — fail-loud, not fail-silent).
- Semver: pycaret-ng v1.0.x for patch + small additive features; v1.1.0 introduces the LLM optional extras.
- pycaret-ng does *not* promise semver alignment with upstream pycaret. Once upstream resumes releases, the version trees may diverge.
