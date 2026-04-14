"""table1.py — cohort characteristics, overall and stratified by a group."""
import numpy as np
import pandas as pd

from references._shared.table1 import table1


def test_table1_numeric_row_has_mean_sd():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.normal(60, 10, 200),
        "sex": rng.choice(["M", "F"], size=200),
    })
    t = table1(df)
    age_row = t[t["variable"] == "age"].iloc[0]
    assert "±" in str(age_row["overall"])


def test_table1_categorical_row_has_n_and_pct():
    df = pd.DataFrame({"sex": ["M"] * 120 + ["F"] * 80})
    t = table1(df)
    assert any("120" in str(v) for v in t["overall"])


def test_table1_stratified_by_group_produces_group_columns_and_pvalue():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.normal(60, 10, 300),
        "sex": rng.choice(["M", "F"], size=300),
    })
    group = pd.Series(rng.choice(["case", "ctrl"], size=300))
    t = table1(df, group=group)
    assert "case" in t.columns and "ctrl" in t.columns
    assert "p_value" in t.columns
    assert t["p_value"].notna().any()
