---
name: compare
description: Run cross-validated model comparison for the current task. Produces a leaderboard CSV + markdown.
triggers:
  - compare models
  - leaderboard
  - which model is best
  - model comparison
allowed-tools:
  - Bash(python*)
  - Read
  - Write
  - Edit
---

# Model Comparison Playbook

## Prerequisite

An active session scratchpad at `.mltoolkit/session.py`. If missing, invoke `mltoolkit:setup` first, then the task-specific skill (`classify`/`regress`/`cluster`/`anomaly`).

## Workflow

1. **Detect task type** by reading `.mltoolkit/session.py` (look for imports of classify/regress/cluster/anomaly modules).
2. **Run compare stage**: `python .mltoolkit/session.py --data <DATA> --target <TARGET> --output-dir .mltoolkit --stage compare`
   - Pass `--target` only if task is supervised.
3. **Read `.mltoolkit/results/leaderboard.csv`** and present as markdown.
4. **Identify top model(s)** for next steps.
