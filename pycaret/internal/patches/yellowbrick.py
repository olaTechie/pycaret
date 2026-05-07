"""Pycaret-side shims for yellowbrick API drift.

Two interlocking issues drive the need for these shims:

1. Yellowbrick's ``is_classifier`` / ``is_regressor`` / ``is_clusterer``
   check ``estimator._estimator_type``. Scikit-learn >= 1.6 dropped that
   attribute in favour of ``__sklearn_tags__``, so yellowbrick's
   detectors return ``False`` for every modern sklearn estimator —
   pipeline-wrapped or not. This causes
   ``YellowbrickTypeError: This estimator is not a classifier`` (and
   the regressor / clusterer siblings).

2. Pycaret occasionally hands yellowbrick a meta-estimator wrapping the
   real model (e.g., ``TransformedTargetClassifier``); we want the
   wrapped estimator to be probed, not the wrapper.

These shims resolve both: they unwrap any meta-estimator pycaret added,
then delegate to sklearn's own role checkers (which are aware of
``__sklearn_tags__``). For ``is_clusterer`` (which sklearn does not
expose publicly), we read ``__sklearn_tags__().estimator_type`` and
fall back to ``_estimator_type`` for older sklearn.

These shims are installed via ``mock.patch`` against each yellowbrick
*consumer* module's local binding (``yellowbrick.classifier.base``,
``yellowbrick.regressor.base``, ``yellowbrick.cluster.base``); patching
the source ``yellowbrick.utils.types`` alone does not reach those
consumers because they import the symbols at module load time.

Installation site: ``pycaret/internal/pycaret_experiment/tabular_experiment.py``.
"""

from __future__ import annotations

from sklearn.base import (
    is_classifier as _sk_is_classifier,
    is_regressor as _sk_is_regressor,
)
from yellowbrick.utils.helpers import get_model_name as _get_model_name_original

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
    return _sk_is_classifier(_unwrap(model))


def is_regressor(model):
    return _sk_is_regressor(_unwrap(model))


def is_clusterer(model):
    """sklearn does not expose ``is_clusterer`` publicly; read tags."""
    m = _unwrap(model)
    tags = getattr(m, "__sklearn_tags__", None)
    if callable(tags):
        try:
            return tags().estimator_type == "clusterer"
        except Exception:
            pass
    return getattr(m, "_estimator_type", None) == "clusterer"


def get_model_name(model):
    return _get_model_name_original(_unwrap(model))
