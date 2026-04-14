"""Refuse target-encoding on protected-attribute columns by default.

    is_sensitive_column(name, sensitive) -> bool
        True if `name` is in the explicit `sensitive` list OR matches a
        built-in SENSITIVE_ATTRIBUTE_PATTERNS regex.

    safe_high_cardinality_encoder(name, sensitive, *,
                                  allow_target_encode_on_sensitive=False)
        Returns a fitted-ready encoder instance. Raises ValueError if the
        column is sensitive and the caller has not explicitly opted in.

Reason: target-encoding a protected attribute leaks outcome rate into a
proxy-discrimination vector. This module makes that a loud failure by
default.
"""
from __future__ import annotations

import re
from typing import Sequence

from sklearn.preprocessing import OrdinalEncoder


SENSITIVE_ATTRIBUTE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in (
        r"^race(_|$)", r"^ethnicity", r"^sex(_|$)", r"^gender",
        r"^zip(_?code)?$", r"^postcode", r"^religion",
        r"^age_bucket", r"disability(_status)?$", r"^sexual_orientation",
        r"^national_origin", r"^pregnancy",
    )
]


def is_sensitive_column(name: str, sensitive: Sequence[str]) -> bool:
    if name in set(sensitive):
        return True
    return any(p.search(name) for p in SENSITIVE_ATTRIBUTE_PATTERNS)


def safe_high_cardinality_encoder(
    name: str,
    sensitive: Sequence[str],
    *,
    allow_target_encode_on_sensitive: bool = False,
):
    if is_sensitive_column(name, sensitive) and not allow_target_encode_on_sensitive:
        raise ValueError(
            f"Refusing to target-encode sensitive column '{name}'. "
            "Pass --allow-target-encode-on-sensitive to override, "
            "or drop/one-hot/ordinal-encode instead."
        )
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
