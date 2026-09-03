from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl

from easy_glm.core.plots import plot_all_ratetables


def test_numeric_tables_plot_as_steps_and_text_tables_as_bars(monkeypatch):
    """The visual form distinguishes a numeric shape from text levels."""
    monkeypatch.setattr(plt, "show", lambda: None)
    tables = {
        "BonusMalus": pl.DataFrame(
            {
                "from": [None, 50.0],
                "to": [50.0, 70.0],
                "label": ["< 50", "[50, 70)"],
                "relativity": [1.0, 1.4],
            }
        ),
        "Region": pl.DataFrame(
            {
                "from": ["North", "South"],
                "to": ["North", "South"],
                "label": ["North", "South"],
                "relativity": [1.0, 1.2],
            }
        ),
    }

    plot_all_ratetables(tables)
    figures = {
        figure.axes[0].get_title(): figure
        for figure in map(plt.figure, plt.get_fignums())
    }

    numeric_axis = figures["Relativity for BonusMalus"].axes[0]
    assert numeric_axis.lines[0].get_drawstyle() == "steps-post"

    text_axis = figures["Relativity for Region"].axes[0]
    assert text_axis.patches
    plt.close("all")
