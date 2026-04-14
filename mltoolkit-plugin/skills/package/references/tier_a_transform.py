"""Tier A packaging: copy session.py into a clean named deliverable.

The session script is already end-to-end; Tier A strips its `.mltoolkit` output
default to a user-supplied directory and writes it to the chosen filename.
"""
import argparse
from pathlib import Path


def transform(session_path: Path, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    content = session_path.read_text()
    if "pycaret" in content.lower():
        raise SystemExit("session.py contains pycaret references — refusing to package.")
    content = content.replace('default=".mltoolkit"', 'default="output"')
    deliverable = output_dir / f"{name}.py"
    deliverable.write_text(content)
    return deliverable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    p = transform(Path(args.session), Path(args.output_dir), args.name)
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
