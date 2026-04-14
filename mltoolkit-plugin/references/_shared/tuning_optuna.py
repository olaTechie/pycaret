"""Optuna-backed hyperparameter search, API-compatible with tune_model callers.

Exposed callable:
    optuna_search(pipe, grid, X, y, scoring, cv, n_iter, random_state)
      -> (best_estimator, best_score, best_params)

`grid` uses the same shape as RandomizedSearchCV param_distributions
(a dict of `step__param` → list of candidate values). This keeps both
backends drop-in interchangeable at the call site.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score


def optuna_search(
    pipe,
    grid: dict,
    X,
    y,
    *,
    scoring: str = "accuracy",
    cv: int = 5,
    n_iter: int = 50,
    random_state: int = 42,
):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {k: trial.suggest_categorical(k, v) for k, v in grid.items()}
        est = clone(pipe).set_params(**params)
        scores = cross_val_score(est, X, y, cv=cv, scoring=scoring, n_jobs=1)
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_iter, show_progress_bar=False)

    best_params = study.best_params
    best = clone(pipe).set_params(**best_params)
    best.fit(X, y)
    return best, float(study.best_value), best_params
