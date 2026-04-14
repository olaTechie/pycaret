"""Smoke test for cluster reference."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def test_cluster_reference_runs_end_to_end(cluster_data, tmp_path):
    script = REPO_ROOT / "skills/cluster/references/cluster_reference.py"
    out = tmp_path / "out"
    result = subprocess.run(
        ["python", str(script),
         "--data", cluster_data["path"],
         "--output-dir", str(out),
         "--stage", "all",
         "--n-clusters", "4"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (out / "results/leaderboard.csv").exists()
    assert (out / "artifacts/pca_scatter.png").exists()
    assert (out / "results/assigned.csv").exists()
