"""Markdown / LaTeX / HTML report generation for ML results."""
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render a DataFrame as a GitHub-flavored markdown table."""
    return df.to_markdown(index=False, floatfmt=floatfmt)


def df_to_latex(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render a DataFrame as a LaTeX table (booktabs style)."""
    return df.to_latex(index=False, float_format=lambda x: f"%{floatfmt}" % x)


def summary_report(
    title: str,
    tables: Dict[str, pd.DataFrame],
    figures: List[Union[str, Path]],
    output: Union[str, Path],
) -> Path:
    """Emit a one-page HTML summary with tables + embedded figures.

    Parameters
    ----------
    title : str
    tables : dict of {section_name: DataFrame}
    figures : list of image paths (PNG)
    output : destination HTML path
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        f"<title>{title}</title>",
        "<style>",
        "body { font-family: -apple-system, sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; }",
        "h1 { border-bottom: 2px solid #333; padding-bottom: 0.3em; }",
        "h2 { color: #333; margin-top: 1.5em; }",
        "table { border-collapse: collapse; width: 100%; margin: 1em 0; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background: #f5f5f5; }",
        "img { max-width: 100%; margin: 1em 0; }",
        "</style></head><body>",
        f"<h1>{title}</h1>",
    ]
    for name, df in tables.items():
        parts.append(f"<h2>{name}</h2>")
        parts.append(df.to_html(index=False, float_format=lambda x: f"{x:.4f}"))
    if figures:
        parts.append("<h2>Figures</h2>")
        for fig_path in figures:
            parts.append(f'<img src="{Path(fig_path).as_posix()}" alt="figure">')
    parts.append("</body></html>")
    output.write_text("\n".join(parts))
    return output
