---
name: tune
description: Hyperparameter tuning for the current model — uses RandomizedSearchCV (or optuna if installed).
triggers:
  - tune
  - hyperparameter tuning
  - optimize hyperparameters
allowed-tools:
  - Bash(python*)
  - Read
  - Write
  - Edit
---

# Tuning Playbook

## Workflow

1. **Confirm model id** with user (from leaderboard).
2. **Run tune stage**: `python .mltoolkit/session.py --data <DATA> --target <TARGET> --output-dir .mltoolkit --stage tune --model <ID> [--search-library optuna] [--n-iter 50]`
3. **Present `best_params.json`** and the new CV score.
4. If the tuned score is not better than the untuned CV score, mention that and offer to try another model or expand the grid (edit `session.py` accordingly).

## Search backends

The tune stage accepts `--search-library {sklearn,optuna}`.

- `sklearn` (default) uses `RandomizedSearchCV` with `--n-iter` trials.
- `optuna` uses a TPE sampler (requires the `optuna` package). If requested but optuna is not installed, the script prints a warning and transparently falls back to `sklearn`.

Both backends write `results/best_params.json` with identical schema.
