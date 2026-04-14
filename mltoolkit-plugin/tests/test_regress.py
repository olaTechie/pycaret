"""Smoke test for regress reference."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "skills/regress/references"))

import model_zoo as regress_zoo  # noqa: E402


def test_regress_model_zoo_has_expanded_entries():
    zoo = regress_zoo.get_zoo()
    for mid in ["lr", "ridge", "lasso", "en", "huber", "ransac", "theilsen",
                "br", "ard", "omp", "lassolars", "llars_ic", "dummy"]:
        assert mid in zoo, f"missing regress model: {mid}"
        assert "estimator" in zoo[mid]
        assert "param_grid" in zoo[mid]


def test_regress_reference_runs_end_to_end(regression_data, tmp_path):
    script = REPO_ROOT / "skills/regress/references/regress_reference.py"
    out = tmp_path / "out"
    result = subprocess.run(
        ["python", str(script),
         "--data", regression_data["path"],
         "--target", regression_data["target"],
         "--output-dir", str(out),
         "--stage", "all"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (out / "results/leaderboard.csv").exists()
    assert (out / "model.joblib").exists()
    figures = list((out / "artifacts").glob("*.png"))
    assert len(figures) >= 1
