"""Parity harness fixtures and config.

Tolerances are the gate-B thresholds from the pycaret-ng spec:
  - Metric absolute delta < 1e-4
  - Prediction rank-correlation > 0.999

Per-estimator widening is allowed via PARITY_TOLERANCE_OVERRIDES and must
be justified in docs/superpowers/agents/qa/phase-N-parity.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pytest

BASELINE_VERSION = "3.4.0"
BASELINE_ROOT = Path(__file__).parent / "baselines" / BASELINE_VERSION

METRIC_ABS_TOLERANCE = 1e-4
PREDICTION_RANK_CORR_MIN = 0.999

# Per-estimator overrides. Populated by QA as needed; empty in Phase 0.
PARITY_TOLERANCE_OVERRIDES: Dict[str, Dict[str, float]] = {}


@dataclass
class ParityConfig:
    baseline_root: Path
    metric_abs_tol: float
    rank_corr_min: float
    overrides: Dict[str, Dict[str, float]]


@pytest.fixture(scope="session")
def parity_config() -> ParityConfig:
    return ParityConfig(
        baseline_root=BASELINE_ROOT,
        metric_abs_tol=METRIC_ABS_TOLERANCE,
        rank_corr_min=PREDICTION_RANK_CORR_MIN,
        overrides=PARITY_TOLERANCE_OVERRIDES,
    )


def pytest_collection_modifyitems(config, items):
    """Skip parity tests when baseline artifacts are missing.

    This lets CI run on a fresh clone before Task 12 produces baselines.
    """
    if not BASELINE_ROOT.exists():
        skip = pytest.mark.skip(
            reason=f"Baseline not built yet at {BASELINE_ROOT}. "
            f"Run scripts/build_parity_baseline.py."
        )
        for item in items:
            if "parity" in str(item.fspath) and item.name.startswith(
                ("test_compare_models_parity", "test_predict_parity")
            ):
                item.add_marker(skip)
