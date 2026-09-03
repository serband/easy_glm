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
    marks: dict[str, str] | None = None,
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
    for label, text in (marks or {}).items():
        # marks: {category label: annotation text}, e.g. the clamp bands
        if label in labels:
            fig.add_vline(
                x=labels.index(label),
                line=dict(color=GREEN, dash="dash"),
                annotation_text=text,
                annotation_position="top",
            )
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
    """The regularisation path of **one** stage; filter on ``stage`` first for a
    two-stage interaction fit, whose two paths are over different columns."""
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


# --------------------------------------------------------------------------
# heatmaps (interactions, A/E by pair) and continuous curves (linear terms)
# --------------------------------------------------------------------------
def matrix_heatmap(
    row_labels: list[str],
    col_labels: list[str],
    values: list[list[float]],
    *,
    title: str = "",
    row_name: str = "",
    col_name: str = "",
    hover: dict[str, list[list[float]]] | None = None,
    log_colour: bool = True,
    centred: bool = True,
    height: int = 460,
) -> go.Figure:
    """Heatmap of a matrix indexed by the parents' rate-table rows.

    ``values`` are multiplicative (relativities or A/E), so the colour is
    centred on 1.0 (log scale when ``log_colour``); cells with a ``None``
    value are blank. ``hover`` adds named matrices to the hover text (e.g.
    exposure, actual, expected)."""
    import math

    if not centred:  # plain magnitudes (e.g. exposure): no log, no centre
        log_colour = False
    z = [
        [
            (
                (math.log(v) if log_colour else v)
                if v is not None and (v > 0 or not log_colour)
                else None
            )
            for v in row
        ]
        for row in values
    ]
    hover = hover or {}
    names = list(hover)
    ticks = [0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0]
    colorbar = (
        dict(
            title="ratio",
            thickness=12,
            tickvals=[math.log(t) for t in ticks],
            ticktext=[f"{t:.2f}" for t in ticks],
        )
        if (centred and log_colour)
        else dict(title=("ratio" if centred else ""), thickness=12)
    )
    custom = [
        [
            [values[i][j]] + [hover[n][i][j] for n in names]
            for j in range(len(col_labels))
        ]
        for i in range(len(row_labels))
    ]
    parts = [f"{row_name or 'row'}: %{{y}}", f"{col_name or 'column'}: %{{x}}"]
    parts.append("value: %{customdata[0]:.4f}")
    for k, n in enumerate(names, start=1):
        parts.append(f"{n}: %{{customdata[{k}]:,.1f}}")
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=col_labels,
            y=row_labels,
            customdata=custom,
            hovertemplate="<br>".join(parts) + "<extra></extra>",
            colorscale="RdBu_r" if centred else "Blues",
            zmid=(0.0 if log_colour else 1.0) if centred else None,
            colorbar=colorbar,
            xgap=1,
            ygap=1,
        )
    )
    fig.update_layout(
        height=height,
        title=title,
        xaxis=dict(type="category", title=col_name, tickangle=-45),
        yaxis=dict(type="category", title=row_name, autorange="reversed"),
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=100),
        hovermode="closest",
    )
    return fig


def linear_curve_chart(
    table: pl.DataFrame,
    *,
    title: str = "",
    working: pl.DataFrame | None = None,
    clamp: tuple[float, float] | None = None,
    x_base: float | None = None,
    height: int = 360,
    log_x: bool = False,
) -> go.Figure:
    """Continuous relativity curve of a piecewise-linear table
    (``from``, ``to``, ``relativity`` at the band start, ``relativity_to`` at
    the band end); flat end rows drawn as short horizontal segments, the null
    row omitted. ``working`` overlays a second table (after edits)."""
    fig = go.Figure()

    def _polyline(tbl: pl.DataFrame) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        bands = tbl.filter(pl.col("from").is_not_null() & pl.col("to").is_not_null())
        for row in bands.sort("from").iter_rows(named=True):
            x0, x1 = float(row["from"]), float(row["to"])
            y0 = float(row["relativity"])
            y1 = float(row.get("relativity_to", y0))
            n = 12
            for k in range(n + 1):
                t = k / n
                xs.append(x0 + t * (x1 - x0))
                # log-linear inside the band
                ys.append(y0 * (y1 / y0) ** t if y0 > 0 and y1 > 0 else y0)
        return xs, ys

    xs, ys = _polyline(table)
    fig.add_scatter(
        x=xs, y=ys, name="fitted", mode="lines", line=dict(color=BLUE, width=2.5)
    )
    if working is not None:
        wx, wy = _polyline(working)
        fig.add_scatter(
            x=wx, y=wy, name="working", mode="lines", line=dict(color=ORANGE, width=2.5)
        )
    fig.add_hline(y=1.0, line=dict(color=GREY, dash="dot"))
    if clamp is not None:
        for x, label in zip(clamp, ("clamp lo", "clamp hi"), strict=True):
            fig.add_vline(
                x=x,
                line=dict(color=GREEN, dash="dash"),
                annotation_text=label,
                annotation_position="top",
            )
    if x_base is not None:
        fig.add_vline(
            x=x_base,
            line=dict(color=GREY, dash="dot"),
            annotation_text="base (1.00)",
            annotation_position="bottom",
        )
    fig.update_layout(
        height=height,
        title=title,
        xaxis=dict(
            title="value" + (" (log scale)" if log_x else ""),
            type="log" if log_x else "linear",
        ),
        yaxis=dict(title="relativity", rangemode="tozero"),
        **_LAYOUT,
    )
    return fig
