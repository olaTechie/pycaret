---
name: eda
description: Generate exploratory figures for the current dataset (correlation heatmap, target distribution, feature distributions).
triggers:
  - eda
  - explore data
  - exploratory analysis
  - visualize data
allowed-tools:
  - Bash(python*)
  - Read
  - Write
  - Edit
---

# EDA Playbook

## Workflow

1. **If no session exists**: run the generic EDA via `python {PLUGIN_ROOT}/skills/setup/references/setup_reference.py --data <DATA> --target <TARGET> --output-dir .mltoolkit`.
2. **If session exists**: run task-specific EDA — `python .mltoolkit/session.py --data <DATA> --target <TARGET> --output-dir .mltoolkit --stage eda`.
3. **Read all figures** under `.mltoolkit/artifacts/` and present them to the user.
4. **Flag issues**: high missing rate, severe class imbalance (>4:1), highly skewed numeric targets, highly correlated features (>0.95).
