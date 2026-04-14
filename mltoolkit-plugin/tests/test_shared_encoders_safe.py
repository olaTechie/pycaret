"""encoders_safe.py — refuse TargetEncoder on sensitive columns."""
import pytest

from references._shared.encoders_safe import (
    is_sensitive_column, safe_high_cardinality_encoder,
)


def test_known_sensitive_names_are_flagged():
    for col in ["race", "ethnicity", "sex", "gender", "zip_code", "zipcode",
                "postcode", "religion", "age_bucket", "disability_status"]:
        assert is_sensitive_column(col, sensitive=[])


def test_explicit_sensitive_overrides_patterns():
    assert is_sensitive_column("patient_id", sensitive=["patient_id"])


def test_nonsensitive_names_not_flagged():
    for col in ["age", "bmi", "income", "region"]:
        assert not is_sensitive_column(col, sensitive=[])


def test_safe_encoder_refuses_target_encode_on_sensitive_without_override():
    with pytest.raises(ValueError, match="sensitive"):
        safe_high_cardinality_encoder(
            "race", sensitive=[], allow_target_encode_on_sensitive=False,
        )


def test_safe_encoder_returns_encoder_when_allowed():
    enc = safe_high_cardinality_encoder(
        "race", sensitive=["race"], allow_target_encode_on_sensitive=True,
    )
    assert enc is not None
