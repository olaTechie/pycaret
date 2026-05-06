"""Pycaret-side shims for yellowbrick API drift.

Yellowbrick probes a model's role (classifier / regressor / clusterer)
via ``is_classifier`` / ``is_regressor`` / ``is_clusterer`` (and their
underscore-less aliases ``isclassifier`` / ``isregressor`` /
``isclusterer``) on the *Pipeline* pycaret hands it. Modern sklearn
(>=1.6) reads ``__sklearn_tags__`` / ``_estimator_type`` directly off
the wrapped object, so yellowbrick's checks fail on a Pipeline that
doesn't expose those attributes from its outermost step.

These shims unwrap pycaret's meta-estimator/pipeline first, then
delegate to yellowbrick's original detector. They are installed via
``mock.patch`` in
``pycaret/internal/pycaret_experiment/tabular_experiment.py``.
"""
from __future__ import annotations

from yellowbrick.utils.helpers import get_model_name as _get_model_name_original
from yellowbrick.utils.types import (
    is_classifier as _is_classifier_original,
    is_clusterer as _is_clusterer_original,
    is_regressor as _is_regressor_original,
)

from pycaret.internal.meta_estimators import get_estimator_from_meta_estimator


def _unwrap(model):
    """Return the innermost estimator pycaret would dispatch on."""
    return get_estimator_from_meta_estimator(model)


def is_estimator(model):
    try:
        return callable(getattr(model, "fit"))
    except Exception:
        return False


def is_classifier(model):
    return _is_classifier_original(_unwrap(model))


def is_regressor(model):
    return _is_regressor_original(_unwrap(model))


def is_clusterer(model):
    return _is_clusterer_original(_unwrap(model))


def get_model_name(model):
    return _get_model_name_original(_unwrap(model))
