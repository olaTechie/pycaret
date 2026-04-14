---
name: regress
description: Build a regression pipeline with native sklearn code. Runs inline, shows leaderboard + residual/Q-Q/prediction plots, packages on demand.
triggers:
  - regress
  - regression
  - predict number
  - continuous prediction
allowed-tools:
  - Bash(python*)
  - Read
  - Write
  - Edit
---

# Regression Playbook

## Prerequisites
- Session scratchpad: `.mltoolkit/session.py` (user's CWD)
- Reference: `{SKILL_DIR}/references/regress_reference.py`
- Shared helpers: `{PLUGIN_ROOT}/references/_shared/`

## Workflow

1. Read `{SKILL_DIR}/references/regress_reference.py`.
2. Ensure `.mltoolkit/` exists in user's CWD (gitignored).
3. Copy the reference to `.mltoolkit/session.py`.
4. Run EDA:
   `python .mltoolkit/session.py --data <DATA> --target <TARGET> --output-dir .mltoolkit --stage eda`
5. Show target distribution + correlation heatmap from `.mltoolkit/artifacts/`.
6. Run compare:
   `python .mltoolkit/session.py --data <DATA> --target <TARGET> --output-dir .mltoolkit --stage compare`
7. Present `leaderboard.csv` — R², RMSE, MAE across models.
8. Ask which model to tune; default to top of R².
9. Run tune + evaluate stages.
10. Present residuals, Q-Q plot, prediction-vs-actual, permutation importance.
11. Offer to invoke `mltoolkit:package`.

## Adaptation rules

- **No pycaret imports, ever.**
- **Target always parameterized.**
- If rows > 100k, use `--cv 3`.
- If the target distribution is highly skewed (suggest via EDA), propose log-transform: edit `session.py` to wrap `y` with `np.log1p` / `np.expm1` via `TransformedTargetRegressor`.
- If residuals show strong heteroskedasticity, suggest quantile regression or a different model family.

## Iteration prompts

- **"transform the target"** → wrap estimator in `sklearn.compose.TransformedTargetRegressor`.
- **"remove outliers"** → add `IsolationForest` filter before splitting.
- **"try robust regression"** → add `HuberRegressor` or `RANSACRegressor` to the zoo import in `session.py`.

## Hand-off

Same as classify — invoke `mltoolkit:package` with task type `regression`.
