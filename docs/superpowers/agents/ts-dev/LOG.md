# Time-series Migration Dev — Log

Append-only progress log.

## 2026-05-06 — Phase 4 kickoff
- Spec committed: `docs/superpowers/specs/2026-05-06-phase-4-timeseries-design.md` (`e212bdfd`).
- Plan committed: `docs/superpowers/plans/2026-05-06-phase-4-timeseries.md` (`ff0cb610`).
- Branch: `phase-4-timeseries` off `phase-3-plotting` HEAD `0a23a615`.
- `.venv-phase4` created (Python 3.12.13, uv-managed).

## 2026-05-06 — Phase 4 close
- Install probe **forced** the sktime unpin (W4): the existing `>=0.31.0,<0.31.1` pin caps sklearn at `<1.6.0`, conflicting with Phase 1's sklearn>=1.6 floor. uv refused to resolve. Lifted to `sktime>=0.31` (`3a1523eb`); resolver picked sktime 0.40.1 / sklearn 1.7.2 / pmdarima 2.0.4 / tbats 1.1.3 / statsmodels 0.14.6 / numpy 1.26.4 cleanly.
- pycaret's own `force_all_finite=` rename in `iterative_imputer.py` (`fbfbc50c`) closes the pycaret-side of row 16. pmdarima-side is currently latent (sktime 0.40.1 caps sklearn <1.8; the deprecated kwarg still works). The Plan's Task 5 shim is deferred until sklearn 1.8 becomes reachable.
- BATS/TBATS graceful-disable (`e3b0066b`) closes row 12. Path is dormant in this venv (numpy 1.26 satisfies tbats); activates when a future install pins numpy ≥2.
- statsmodels floor lifted from `>=0.12.1` to `>=0.14,<1` (`dc1fa7f4`).
- TS smoke harness (`b735748e`): 18 passed, 2 skipped (`auto_arima` create + predict — pathologically slow under sktime 0.40.1's wider search space; documented in DEGRADED.md as a sktime-drift smoke skip), 0 failed in 10.43s. Aggregate well under the 120s budget.
- Phase 3 plotting smoke regression check: 38 passed, 3 skipped (unchanged).
- Conftest amended to set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` — the .venv-phase4 install pulled in protobuf >= 4 which clashes with mlflow's older _pb2.py modules.
- Phase 4 ready for push + PR.
