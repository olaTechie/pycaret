---
name: cluster
description: Build a clustering pipeline — KMeans, DBSCAN, Agglomerative, GMM. Runs inline with elbow + silhouette + PCA visualization.
triggers:
  - cluster
  - clustering
  - segment
  - group similar
allowed-tools:
  - Bash(python*)
  - Read
  - Write
  - Edit
---

# Clustering Playbook

## Prerequisites
- Session scratchpad: `.mltoolkit/session.py`
- Reference: `{SKILL_DIR}/references/cluster_reference.py`

## Workflow

1. Read reference script.
2. Copy to `.mltoolkit/session.py`.
3. Run EDA — elbow plot for choosing k:
   `python .mltoolkit/session.py --data <DATA> --output-dir .mltoolkit --stage eda`
4. Read elbow plot from `.mltoolkit/artifacts/elbow.png` — suggest a k to the user.
5. Run compare with chosen k:
   `python .mltoolkit/session.py --data <DATA> --output-dir .mltoolkit --stage compare --n-clusters <K>`
6. Present leaderboard (silhouette, noise points).
7. Show PCA scatter colored by cluster labels.
8. Offer `mltoolkit:package`.

## Adaptation rules

- **No pycaret.**
- All features must be numeric after prep — if categorical columns exist, prompt user whether to one-hot them or drop.
- For DBSCAN, `n_clusters` is ignored (density-based).
- Suggest scaling is always on (already baked into the pipeline).
