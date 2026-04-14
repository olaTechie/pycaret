"""Smoke test for anomaly reference."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def test_anomaly_reference_runs_end_to_end(anomaly_data, tmp_path):
    script = REPO_ROOT / "skills/anomaly/references/anomaly_reference.py"
    out = tmp_path / "out"
    result = subprocess.run(
        ["python", str(script),
         "--data", anomaly_data["path"],
         "--output-dir", str(out),
         "--stage", "all",
         "--contamination", "0.05"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (out / "results/scores.csv").exists()
    figures = list((out / "artifacts").glob("*.png"))
    assert len(figures) >= 1
