"""Stager + end-to-end copy-and-run tests for every reference script."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGER = REPO_ROOT / "scripts/stage_session.py"


def _run_stager(task: str, dest: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(STAGER), "--task", task, "--dest", str(dest)],
        capture_output=True, text=True,
    )


def test_stager_exists():
    assert STAGER.exists(), f"Stager missing at {STAGER}"


def test_stager_classify_creates_expected_layout(tmp_path):
    dest = tmp_path / "mlt"
    r = _run_stager("classify", dest)
    assert r.returncode == 0, f"stager failed: {r.stderr}"
    assert (dest / "session.py").is_file()
    assert (dest / "preprocessing.py").is_file()
    assert (dest / "model_zoo.py").is_file()
    assert (dest / "_shared/__init__.py").is_file()
    assert (dest / "_shared/deps.py").is_file()
    assert (dest / "_shared/plotting.py").is_file()
    assert (dest / "_shared/reporting.py").is_file()


def test_stager_rewrites_header_in_session_py(tmp_path):
    dest = tmp_path / "mlt"
    _run_stager("classify", dest).check_returncode()
    text = (dest / "session.py").read_text()
    assert "references._shared" not in text
    assert "_PLUGIN_ROOT" not in text
    assert "from _shared import" in text


def test_stager_refuses_unknown_task(tmp_path):
    r = _run_stager("not_a_task", tmp_path / "mlt")
    assert r.returncode != 0
    msg = (r.stderr + r.stdout).lower()
    assert "unknown task" in msg or "invalid choice" in msg
