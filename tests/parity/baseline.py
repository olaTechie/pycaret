"""Schema & I/O for parity baseline artifacts.

Artifacts live under tests/parity/baselines/<version>/<dataset>/:
  - leaderboard.json — sorted compare_models() output.
  - predictions.npz  — per-model predictions + probabilities on the holdout.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class LeaderboardBaseline:
    dataset: str
    version: str
    task: str  # "classification" | "regression" | "time_series"
    rows: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "version": self.version,
            "task": self.task,
            "rows": self.rows,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LeaderboardBaseline":
        return cls(
            dataset=d["dataset"],
            version=d["version"],
            task=d["task"],
            rows=d["rows"],
        )


@dataclass
class PredictionBaseline:
    dataset: str
    version: str
    model: str
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None  # None for regression/TS


def save_leaderboard(b: LeaderboardBaseline, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(b.to_dict(), indent=2, sort_keys=True))


def load_leaderboard(path: Path) -> LeaderboardBaseline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return LeaderboardBaseline.from_dict(json.loads(path.read_text()))


def save_predictions(b: PredictionBaseline, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"predictions": b.predictions}
    if b.probabilities is not None:
        arrays["probabilities"] = b.probabilities
    np.savez(
        path,
        dataset=np.array(b.dataset),
        version=np.array(b.version),
        model=np.array(b.model),
        **arrays,
    )


def load_predictions(path: Path) -> PredictionBaseline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    npz = np.load(path, allow_pickle=False)
    probs = npz["probabilities"] if "probabilities" in npz.files else None
    return PredictionBaseline(
        dataset=str(npz["dataset"]),
        version=str(npz["version"]),
        model=str(npz["model"]),
        predictions=npz["predictions"],
        probabilities=probs,
    )
