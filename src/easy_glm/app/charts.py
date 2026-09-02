"""Plotly charts for the workbench (no Streamlit imports)."""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

BLUE = "#1f5f99"
ORANGE = "#e07b39"
GREY = "#c9cfd6"
GREEN = "#2e8b57"
RED = "#c0392b"

_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=40, r=40, t=50, b=80),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified",
)


def _fig(height: int) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(height=height, **_LAYOUT)
    return fig


def exposure_rate_chart(
    table: pl.DataFrame,
    *,
    title: str = "",
    rate_name: str = "observed rate",
    height: int = 380,
) -> go.Figure:
    """Exposure bars + observed rate line by band/level (``univariate`` table)."""
    fig = _fig(height)
    labels = table["label"].to_list()
    fig.add_bar(
        x=labels,
        y=table["exposure"],
        name="exposure",
        marker_color=GREY,
        opacity=0.8,
        secondary_y=False,
    )
    if "rate" in table.columns and table["rate"].null_count() < table.height:
        fig.add_scatter(
            x=labels,
            y=table["rate"],
            name=rate_name,
            mode="lines+markers",
            line=dict(color=BLUE, width=2),
            secondary_y=True,
        )
    fig.update_yaxes(title_text="exposure", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text=rate_name, secondary_y=True, rangemode="tozero")
    fig.update_layout(title=title, xaxis=dict(type="category", tickangle=-45))
    return fig


def ae_chart(
    table: pl.DataFrame,
    *,
    title: str = "",
    height: int = 380,
    compare: pl.DataFrame | None = None,
    compare_name: str = "challenger",
) -> go.Figure:
    """Actual vs expected rates with exposure bars (``ae_by_variable`` table)."""
    fig = _fig(height)
    labels = table["label"].to_list()
    fig.add_bar(
        x=labels,
        y=table["exposure"],
        name="exposure",
        marker_color=GREY,
        opacity=0.7,
        secondary_y=False,
    )
    fig.add_scatter(
        x=labels,
        y=table["actual_rate"],
        name="actual",
        mode="lines+markers",
        line=dict(color=BLUE, width=2.5),
        secondary_y=True,
    )
    fig.add_scatter(
        x=labels,
        y=table["expected_rate"],
        name="expected",
        mode="lines+markers",
        line=dict(color=ORANGE, width=2.5),
        secondary_y=True,
    )
    if compare is not None:
        fig.add_scatter(
            x=compare["label"].to_list(),
            y=compare["expected_rate"],
            name=f"expected ({compare_name})",
            mode="lines+markers",
            line=dict(color=GREEN, width=2, dash="dash"),
            secondary_y=True,
        )
    fig.update_yaxes(title_text="exposure", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="rate", secondary_y=True, rangemode="tozero")
    fig.update_layout(title=title, xaxis=dict(type="category", tickangle=-45))
    return fig


def relativity_chart(
    table: pl.DataFrame,
    *,
    title: str = "",
    height: int = 340,
    working: pl.DataFrame | None = None,
) -> go.Figure:
    """Fitted relativities (and an optional working copy) by band."""
    fig = go.Figure()
    labels = table["label"].to_list()
    fig.add_scatter(
        x=labels,
        y=table["relativity"],
        name="fitted",
        mode="lines+markers",
        line=dict(color=BLUE, width=2.5),
    )
    if working is not None:
        fig.add_scatter(
            x=working["label"].to_list(),
            y=working["relativity"],
            name="working",
            mode="lines+markers",
            line=dict(color=ORANGE, width=2.5),
        )
    fig.add_hline(y=1.0, line=dict(color=GREY, dash="dot"))
    fig.update_layout(
        height=height,
        title=title,
        xaxis=dict(type="category", tickangle=-45),
        yaxis=dict(title="relativity", rangemode="tozero"),
        **_LAYOUT,
    )
    return fig


def lift_chart(
    table: pl.DataFrame, *, title: str = "Lift", height: int = 360
) -> go.Figure:
    fig = _fig(height)
    x = table["bin"].to_list()
    fig.add_bar(
        x=x,
        y=table["exposure"],
        name="exposure",
        marker_color=GREY,
        opacity=0.6,
        secondary_y=False,
    )
    fig.add_scatter(
        x=x,
        y=table["actual_rate"],
        name="actual",
        mode="lines+markers",
        line=dict(color=BLUE, width=2.5),
        secondary_y=True,
    )
    fig.add_scatter(
        x=x,
        y=table["expected_rate"],
        name="expected",
        mode="lines+markers",
        line=dict(color=ORANGE, width=2.5),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="exposure", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="rate", secondary_y=True, rangemode="tozero")
    fig.update_layout(
        title=title, xaxis=dict(title="predicted-rate bin (low → high)", dtick=1)
    )
    return fig


def double_lift_chart(
    table: pl.DataFrame, *, name_a: str, name_b: str, height: int = 360
) -> go.Figure:
    fig = _fig(height)
    x = table["bin"].to_list()
    fig.add_bar(
        x=x,
        y=table["exposure"],
        name="exposure",
        marker_color=GREY,
        opacity=0.6,
        secondary_y=False,
    )
    fig.add_scatter(
        x=x,
        y=table["ae_a"],
        name=f"A/E {name_a}",
        mode="lines+markers",
        line=dict(color=BLUE, width=2.5),
        secondary_y=True,
    )
    fig.add_scatter(
        x=x,
        y=table["ae_b"],
        name=f"A/E {name_b}",
        mode="lines+markers",
        line=dict(color=ORANGE, width=2.5),
        secondary_y=True,
    )
    fig.add_hline(y=1.0, line=dict(color=GREY, dash="dot"), secondary_y=True)
    fig.update_yaxes(title_text="exposure", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="A/E", secondary_y=True)
    fig.update_layout(
        title=f"Double lift — sorted by {name_a} / {name_b}", xaxis=dict(dtick=1)
    )
    return fig


def alpha_path_chart(path: pl.DataFrame, *, height: int = 360) -> go.Figure:
    fig = _fig(height)
    for l1 in path["l1_ratio"].unique().sort().to_list():
        sub = path.filter(pl.col("l1_ratio") == l1).sort("alpha")
        suffix = f" (l1={l1:g})" if path["l1_ratio"].n_unique() > 1 else ""
        if sub["cv_deviance"].null_count() < sub.height:
            fig.add_scatter(
                x=sub["alpha"],
                y=sub["cv_deviance"],
                name="CV deviance" + suffix,
                mode="lines+markers",
                line=dict(color=BLUE),
                error_y=dict(
                    type="data",
                    array=sub["cv_deviance_std"].fill_null(0).to_list(),
                    visible=True,
                    thickness=1,
                ),
                secondary_y=False,
            )
        fig.add_scatter(
            x=sub["alpha"],
            y=sub["n_nonzero"],
            name="non-zero coefficients" + suffix,
            mode="lines+markers",
            line=dict(color=ORANGE, dash="dot"),
            secondary_y=True,
        )
    sel = path.filter(pl.col("selected"))
    if sel.height:
        fig.add_vline(
            x=sel["alpha"][0],
            line=dict(color=GREEN, dash="dash"),
            annotation_text="selected",
        )
    fig.update_xaxes(type="log", title="alpha (log scale)")
    fig.update_yaxes(title_text="CV deviance", secondary_y=False)
    fig.update_yaxes(title_text="# non-zero", secondary_y=True, showgrid=False)
    fig.update_layout(title="Regularisation path", hovermode="closest")
    return fig


def split_balance_chart(table: pl.DataFrame, *, height: int = 300) -> go.Figure:
    fig = _fig(height)
    fig.add_bar(
        x=table["subset"],
        y=table["exposure"],
        name="exposure",
        marker_color=GREY,
        secondary_y=False,
    )
    fig.add_scatter(
        x=table["subset"],
        y=table["rate"],
        name="target rate",
        mode="markers",
        marker=dict(color=BLUE, size=14),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="exposure", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="rate", secondary_y=True, rangemode="tozero")
    fig.update_layout(title="Train / holdout balance", hovermode="closest")
    return fig
