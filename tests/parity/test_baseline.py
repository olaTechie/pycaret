import numpy as np
import pytest

from tests.parity.baseline import (
    LeaderboardBaseline,
    PredictionBaseline,
    load_leaderboard,
    load_predictions,
    save_leaderboard,
    save_predictions,
)


def test_leaderboard_roundtrip(tmp_path):
    b = LeaderboardBaseline(
        dataset="iris",
        version="3.4.0",
        task="classification",
        rows=[
            {"Model": "lr", "Accuracy": 0.96, "AUC": 0.99},
            {"Model": "rf", "Accuracy": 0.95, "AUC": 0.98},
        ],
    )
    path = tmp_path / "leaderboard.json"
    save_leaderboard(b, path)
    loaded = load_leaderboard(path)
    assert loaded == b


def test_predictions_roundtrip(tmp_path):
    b = PredictionBaseline(
        dataset="iris",
        version="3.4.0",
        model="lr",
        predictions=np.array([0, 1, 2, 1]),
        probabilities=np.array(
            [
                [0.9, 0.05, 0.05],
                [0.1, 0.8, 0.1],
                [0.05, 0.05, 0.9],
                [0.1, 0.8, 0.1],
            ]
        ),
    )
    path = tmp_path / "predictions.npz"
    save_predictions(b, path)
    loaded = load_predictions(path)
    assert loaded.dataset == b.dataset
    assert loaded.model == b.model
    np.testing.assert_array_equal(loaded.predictions, b.predictions)
    np.testing.assert_array_almost_equal(loaded.probabilities, b.probabilities)


def test_load_leaderboard_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_leaderboard(tmp_path / "nonexistent.json")
