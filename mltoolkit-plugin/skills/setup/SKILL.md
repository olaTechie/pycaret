---
name: setup
description: Prepare the mltoolkit session — load data, run EDA, auto-detect task type, create .mltoolkit scratchpad.
triggers:
  - load data
  - start ml
  - setup experiment
  - initialize
  - prepare data
allowed-tools:
  - Bash(python*)
  - Read
  - Write
  - Edit
---

# mltoolkit Setup Playbook

## Workflow

1. **Check environment**: run `bash {PLUGIN_ROOT}/scripts/check-env.sh`. Report any missing required packages.
2. **Locate data**: confirm the data path with the user.
3. **Create `.mltoolkit/`** in user's CWD if missing. Add `.mltoolkit/` to `.gitignore` (create `.gitignore` if absent).
4. **Run setup_reference**:
   `python {SKILL_DIR}/references/setup_reference.py --data <DATA> [--target <TARGET>] --output-dir .mltoolkit`
5. **Read `schema.csv`** and the generated figures. Present to user.
6. **Fill out `.mltoolkit/datasheet.md`** with the user. Ask them in order:
   a. **Data provenance** (source, collection dates, sampling)
   b. **Consent & ethics** (IRB status, consent basis, PHI/PII presence)
   c. **Protected attributes** (race, ethnicity, sex, gender, age, zip, religion, disability, national origin, pregnancy, sexual orientation). Record each column name the user identifies as sensitive — these are later passed to classify/regress as `--sensitive-features`.
   d. **Known limitations** (bias, coverage gaps, measurement issues)
7. **Infer task type**:
   - User specified target + target has ≤20 unique values or dtype object → classification
   - User specified target + target continuous → regression
   - No target → clustering or anomaly detection (ask user)
8. **Suggest next skill**: `mltoolkit:classify`, `mltoolkit:regress`, `mltoolkit:cluster`, or `mltoolkit:anomaly`.

## Adaptation rules

- Always create `.mltoolkit/` before any downstream skill runs.
- Never modify the user's data file.
- If rows > 1M, warn user and suggest subsampling.
