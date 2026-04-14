---
name: anomaly
description: Build an anomaly detection pipeline — IsolationForest, LOF, EllipticEnvelope, OneClassSVM. Ranks anomalies and visualizes on PCA.
triggers:
  - anomaly
  - outlier detection
  - find outliers
  - anomaly detection
allowed-tools:
  - Bash(python*)
  - Read
  - Write
  - Edit
---

# Anomaly Detection Playbook

## Prerequisites
- Session scratchpad: `.mltoolkit/session.py`
- Reference: `{SKILL_DIR}/references/anomaly_reference.py`

## Workflow

1. Read reference.
2. Copy to `.mltoolkit/session.py`.
3. Ask user for expected contamination rate (default 5%).
4. Run compare with contamination:
   `python .mltoolkit/session.py --data <DATA> --output-dir .mltoolkit --stage compare --contamination 0.05`
5. Present leaderboard (anomaly counts per model).
6. Run assign with chosen model (default iforest):
   `python .mltoolkit/session.py --data <DATA> --output-dir .mltoolkit --stage all --model iforest --contamination 0.05`
7. Present score histogram, PCA scatter, top-20 anomaly table.
8. Offer `mltoolkit:package`.

## Adaptation rules

- **No pycaret.**
- Contamination rate is critical — ask the user if unclear.
- Numeric features only after prep; prompt before dropping categoricals.
- LOF doesn't serialize (no `.joblib` saved) — note this to the user.
