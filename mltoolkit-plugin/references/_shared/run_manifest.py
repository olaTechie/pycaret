"""Reproducibility manifest — versions, seed, split, CV, hyperparams."""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Optional

_TRACKED_PACKAGES = (
    "scikit-learn", "pandas", "numpy", "scipy", "matplotlib",
    "seaborn", "joblib", "xgboost", "lightgbm", "catboost",
    "imbalanced-learn", "shap", "optuna", "mlflow", "category-encoders",
)


def _pkg(name: str) -> Optional[str]:
    try:
        return _pkg_version(name)
    except PackageNotFoundError:
        return None


def build_manifest(*, stage: str, args_dict: dict,
                   extra: Optional[dict] = None) -> dict:
    pkgs = {p: _pkg(p) for p in _TRACKED_PACKAGES}
    pkgs = {k: v for k, v in pkgs.items() if v is not None}
    serial_args = {}
    for k, v in args_dict.items():
        serial_args[k] = v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
    return {
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": pkgs,
        "args": serial_args,
        "extra": extra or {},
    }


def write_manifest(out: Path, manifest: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / "run_manifest.json"
    existing: list = []
    if p.exists():
        try:
            existing = json.loads(p.read_text())
            if not isinstance(existing, list):
                existing = [existing]
        except Exception:
            existing = []
    existing.append(manifest)
    p.write_text(json.dumps(existing, indent=2))
    return p
