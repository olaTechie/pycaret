"""Smoke test for regress reference."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


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
