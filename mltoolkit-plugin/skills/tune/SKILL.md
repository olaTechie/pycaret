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
2. **Run tune stage**: `python .mltoolkit/session.py --data <DATA> --target <TARGET> --output-dir .mltoolkit --stage tune --model <ID>`
3. **Present `best_params.json`** and the new CV score.
4. If the tuned score is not better than the untuned CV score, mention that and offer to try another model or expand the grid (edit `session.py` accordingly).

## Optuna support

If `optuna` is installed, the generated session can be edited to use `optuna` instead of `RandomizedSearchCV`. Propose this when user asks for more aggressive tuning.
