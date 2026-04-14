"""Consistent figure styling + save helpers for publication-quality output."""
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import seaborn as sns


def set_style() -> None:
    """Apply the mltoolkit default publication style."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.autolayout": True,
    })


def save_fig(fig, path: Union[str, Path]) -> dict:
    """Save a figure as both PNG and PDF at 300 DPI.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str or Path
        Base path without extension. Both .png and .pdf are written.

    Returns
    -------
    dict with keys 'png' and 'pdf' mapping to the actual Path objects.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    png_path = path.with_suffix(".png")
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return {"png": png_path, "pdf": pdf_path}
