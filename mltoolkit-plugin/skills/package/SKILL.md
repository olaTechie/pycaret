---
name: package
description: Package the in-session ML work into a deliverable — Tier A (single file), Tier B (mini project), or Tier C (full scaffold).
triggers:
  - package
  - package this
  - create script
  - save project
  - finalize
allowed-tools:
  - Bash(python*)
  - Read
  - Write
  - Edit
---

# Package Playbook

## Prerequisite

An existing session at `.mltoolkit/session.py` (produced by `classify`/`regress`/`cluster`/`anomaly` skills).

## Workflow

1. **Ask user which tier**:
   - **A** — single self-contained `.py` script
   - **B** — script + `requirements.txt` + `README.md` (default, recommended)
   - **C** — full scaffold: `src/` (preprocess/train/predict), `tests/`, optional `api.py` + `Dockerfile`
2. **Ask output directory** (default: `./<task>_pipeline/`).
3. **Ask the name** of the pipeline (default: `<task>_pipeline`).
4. **Execute the tier transformation** (see below).
5. **Verify output** by running the emitted script against a small sample and confirming no pycaret imports leaked in.
6. **Show the user** the directory tree of what was produced.

## Tier A

Run: `python {SKILL_DIR}/references/tier_a_transform.py --session .mltoolkit/session.py --output-dir <OUT> --name <NAME>`

## Tier B (default)

Run: `python {SKILL_DIR}/references/tier_b_transform.py --session .mltoolkit/session.py --output-dir <OUT> --name <NAME> --task <TASK>`

Where `<TASK>` is one of `classification`, `regression`, `clustering`, `anomaly`.

Produces:
- `<OUT>/<NAME>.py`
- `<OUT>/requirements.txt` (derived from imports actually used)
- `<OUT>/README.md`

## Tier C (full scaffold)

Ask the user whether to include:
- `api.py` + FastAPI (y/n)
- `Dockerfile` (y/n)

Then run:
`python {SKILL_DIR}/references/tier_c_transform.py --session .mltoolkit/session.py --output-dir <OUT> --name <NAME> --task <TASK> [--with-api] [--with-docker]`

Produces:
- `<OUT>/src/{preprocess,train,predict}.py`
- `<OUT>/tests/test_pipeline.py`
- `<OUT>/requirements.txt` (combined template + detected)
- `<OUT>/README.md`
- `<OUT>/api.py` (if `--with-api`)
- `<OUT>/Dockerfile` (if `--with-docker`)

The `src/train.py` is a skeleton — the user will need to swap in their chosen model from the `session.py`. Advise them of this.

## Verification

After any tier:
1. Run `python <OUT>/<NAME>.py --data <USER_DATA> --target <TARGET> --output-dir <TEMP>` (Tiers A/B) or `python <OUT>/src/train.py --data <USER_DATA> --target <TARGET>` (Tier C)
2. Confirm it completes and produces the expected artifacts.
3. Grep output for `pycaret` — must be empty.
