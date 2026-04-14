"""Stage a reference script + its dependencies into a self-contained bundle.

After staging, `<dest>/session.py` imports work without plugin-root path magic
because its siblings (`preprocessing.py`, `model_zoo.py`) and the `_shared/`
package are co-located.

Usage:
    python scripts/stage_session.py --task classify --dest .mltoolkit

Known tasks: classify, regress, cluster, anomaly, setup.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]

TASK_SIBLINGS = {
    "classify": ["classify_reference.py", "preprocessing.py", "model_zoo.py"],
    "regress":  ["regress_reference.py",  "preprocessing.py", "model_zoo.py"],
    "cluster":  ["cluster_reference.py",  "model_zoo.py"],
    "anomaly":  ["anomaly_reference.py",  "model_zoo.py"],
    "setup":    ["setup_reference.py"],
}


def _sources_for(task: str) -> list[Path]:
    try:
        names = TASK_SIBLINGS[task]
    except KeyError as exc:
        raise SystemExit(f"stage_session: unknown task '{task}'. "
                         f"Known: {sorted(TASK_SIBLINGS)}") from exc
    skill_refs = PLUGIN_ROOT / "skills" / task / "references"
    return [skill_refs / n for n in names]


def _rewrite_session_header(dest_file: Path) -> None:
    """Rewrite in-place import header → staging-compatible header."""
    text = dest_file.read_text()
    # Drop the plugin-root sys.path insertion (bare variant).
    text = text.replace(
        "_PLUGIN_ROOT = _HERE.parents[3]\n"
        "sys.path.insert(0, str(_PLUGIN_ROOT))\n",
        "",
    )
    # Drop the guarded variant (used inside preprocessing.py).
    text = text.replace(
        "_PLUGIN_ROOT = _HERE.parents[3]\n"
        "if str(_PLUGIN_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(_PLUGIN_ROOT))\n",
        "",
    )
    # Remap outer package → co-located _shared package.
    text = text.replace("from references._shared", "from _shared")
    dest_file.write_text(text)


def stage(task: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)

    sources = _sources_for(task)
    primary = sources[0]
    dest_session = dest / "session.py"
    shutil.copy2(primary, dest_session)

    for src in sources[1:]:
        shutil.copy2(src, dest / src.name)

    shared_src = PLUGIN_ROOT / "references/_shared"
    shared_dst = dest / "_shared"
    if shared_dst.exists():
        shutil.rmtree(shared_dst)
    shutil.copytree(
        shared_src, shared_dst,
        ignore=shutil.ignore_patterns("__pycache__"),
    )

    _rewrite_session_header(dest_session)
    for src in sources[1:]:
        _rewrite_session_header(dest / src.name)

    return dest_session


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=sorted(TASK_SIBLINGS),
                    help="Which task's reference to stage.")
    ap.add_argument("--dest", default=".mltoolkit",
                    help="Destination directory (default: .mltoolkit).")
    args = ap.parse_args()
    session = stage(args.task, Path(args.dest))
    print(f"Staged {args.task} → {session}")


if __name__ == "__main__":
    main()
