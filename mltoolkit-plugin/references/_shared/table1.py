"""Table 1 — cohort descriptor with optional stratification and p-values.

    table1(df, *, group=None) -> DataFrame
        Columns: variable, overall, (one column per group value if given),
                 p_value (if group given).

Numeric variables → 'mean ± SD'. Categorical → 'n (pct%)' per level.
Tests: numeric uses two-sample t-test; categorical uses chi-square.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from scipy import stats


def _fmt_num(s: pd.Series) -> str:
    return f"{s.mean():.2f} ± {s.std(ddof=1):.2f}"


def _fmt_cat(s: pd.Series, level) -> str:
    n = int((s == level).sum())
    pct = 100.0 * n / max(len(s), 1)
    return f"{n} ({pct:.1f}%)"


def _numeric_pvalue(s: pd.Series, group: pd.Series) -> float:
    groups = [s[group == g].dropna() for g in group.unique()]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return float("nan")
    if len(groups) == 2:
        return float(stats.ttest_ind(groups[0], groups[1], equal_var=False).pvalue)
    return float(stats.f_oneway(*groups).pvalue)


def _categorical_pvalue(s: pd.Series, group: pd.Series) -> float:
    try:
        ct = pd.crosstab(s, group)
        if ct.size == 0:
            return float("nan")
        return float(stats.chi2_contingency(ct).pvalue)
    except Exception:
        return float("nan")


def table1(df: pd.DataFrame, *, group: Optional[pd.Series] = None) -> pd.DataFrame:
    rows: list[dict] = []
    group_labels = list(group.unique()) if group is not None else []

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            row = {"variable": col, "overall": _fmt_num(s)}
            for g in group_labels:
                row[g] = _fmt_num(s[group == g])
            if group is not None:
                row["p_value"] = _numeric_pvalue(s, group)
            rows.append(row)
        else:
            for level in sorted(s.dropna().unique()):
                row = {"variable": f"{col}={level}",
                       "overall": _fmt_cat(s, level)}
                for g in group_labels:
                    row[g] = _fmt_cat(s[group == g], level)
                if group is not None:
                    row["p_value"] = _categorical_pvalue(s, group)
                rows.append(row)
    return pd.DataFrame(rows)
