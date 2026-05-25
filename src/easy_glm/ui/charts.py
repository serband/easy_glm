from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from easy_glm.engine.models import FromToRow, VariableConfig

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"


def _level_labels(rows: list[FromToRow]) -> list[str]:
    labels: list[str] = []
    for row in rows:
        if row.from_ is None and row.to_ is None:
            labels.append("Other / Unknown")
        elif row.from_ is None:
            labels.append(f"< {row.to_}")
        elif row.to_ is None:
            labels.append(f"≥ {row.from_}")
        elif row.from_ == row.to_:
            labels.append(str(row.from_))
        else:
            labels.append(f"[{row.from_}, {row.to_})")
    return labels


def build_histogram(
    data: pl.DataFrame,
    variable: str,
    weight_col: str | None = None,
    height: int = 300,
) -> go.Figure:
    col = data[variable]
    if col.dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int8, pl.Int16):
        values = col.drop_nulls().to_numpy()
        title = f"Histogram of {variable}"
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(x=values, name="Count", marker_color=BLUE, nbinsx=30)
        )
    else:
        cats = col.drop_nulls().value_counts().sort("count", descending=True)
        values = cats[variable].to_list()
        counts = cats["count"].to_list()
        title = f"Distribution of {variable}"
        fig = go.Figure()
        fig.add_trace(go.Bar(x=values, y=counts, name="Count", marker_color=BLUE))

    fig.update_layout(
        title=title,
        height=height,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        bargap=0.05,
    )
    return fig


def build_relativity_chart(
    config: VariableConfig,
    variable: str,
    height: int = 400,
) -> go.Figure:
    labels = _level_labels(config.table)
    rels = [row.relativity for row in config.table]
    fig = go.Figure()

    if config.type == "numeric":
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=rels,
                mode="lines+markers",
                name="Relativity",
                line={"color": BLUE, "width": 2},
                marker={"size": 8, "color": BLUE},
                hovertemplate="%{x}<br>Relativity: %{y:.4f}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[labels[0], labels[-1]],
                y=[1.0, 1.0],
                mode="lines",
                name="Base (1.0)",
                line={"color": "gray", "dash": "dash", "width": 1},
                showlegend=True,
            )
        )
    else:
        colors = [ORANGE if r > 1 else GREEN if r < 1 else "gray" for r in rels]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=rels,
                name="Relativity",
                marker_color=colors,
                hovertemplate="%{x}<br>Relativity: %{y:.4f}",
            )
        )
        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="gray",
            annotation_text="Base (1.0)",
        )

    fig.update_layout(
        title=f"Relativities — {variable}",
        height=height,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        xaxis_title=variable,
        yaxis_title="Relativity",
    )
    return fig


def build_relativity_comparison(
    baseline: VariableConfig,
    working: VariableConfig,
    variable: str,
    height: int = 400,
) -> go.Figure:
    """Overlay baseline (dashed) and working copy (solid) relativities."""
    labels = _level_labels(baseline.table)
    rels_base = [r.relativity for r in baseline.table]
    rels_work = [r.relativity for r in working.table]
    fig = go.Figure()

    if baseline.type == "numeric":
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=rels_base,
                mode="lines+markers",
                name="Original",
                line={"color": "gray", "width": 2, "dash": "dash"},
                marker={"size": 6, "color": "gray"},
                hovertemplate="%{x}<br>Original: %{y:.4f}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=rels_work,
                mode="lines+markers",
                name="Revised",
                line={"color": BLUE, "width": 2.5},
                marker={"size": 8, "color": BLUE},
                hovertemplate="%{x}<br>Revised: %{y:.4f}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[labels[0], labels[-1]],
                y=[1.0, 1.0],
                mode="lines",
                name="Base (1.0)",
                line={"color": "lightgray", "dash": "dot", "width": 1},
                showlegend=True,
            )
        )
    else:
        fig.add_trace(
            go.Bar(
                x=labels,
                y=rels_base,
                name="Original",
                marker_color="lightgray",
                hovertemplate="%{x}<br>Original: %{y:.4f}",
            )
        )
        colors = [ORANGE if r > 1 else GREEN if r < 1 else "gray" for r in rels_work]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=rels_work,
                name="Revised",
                marker_color=colors,
                hovertemplate="%{x}<br>Revised: %{y:.4f}",
            )
        )
        fig.add_hline(y=1.0, line_dash="dot", line_color="gray")

    fig.update_layout(
        title=f"Relativities — {variable}",
        height=height,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        xaxis_title=variable,
        yaxis_title="Relativity",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def build_ae_comparison(
    metrics_base: dict,
    metrics_work: dict,
    variable: str,
    height: int = 400,
) -> go.Figure:
    """Overlay baseline and working copy Actual vs Expected."""
    subsets_b = metrics_base["subsets"]
    subsets_w = metrics_work.get("subsets", subsets_b)
    has_split = "train" in subsets_b and "test" in subsets_b

    train_rows = subsets_b.get("train", subsets_b.get("all", []))
    labels = [r["level"] for r in train_rows] if train_rows else []

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    actual_b = [r["actual"] for r in train_rows]
    expected_b = [r["expected"] for r in train_rows]
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=actual_b,
            mode="lines+markers",
            name="Actual (original)",
            line={"color": BLUE, "width": 1.5, "dash": "dot"},
            marker={"size": 5, "color": BLUE},
            opacity=0.5,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=expected_b,
            mode="lines+markers",
            name="Expected (original)",
            line={"color": ORANGE, "width": 1.5, "dash": "dot"},
            marker={"size": 5, "color": ORANGE},
            opacity=0.5,
        ),
        secondary_y=False,
    )

    train_rows_w = subsets_w.get("train", subsets_w.get("all", []))
    actual_w = [r["actual"] for r in train_rows_w]
    expected_w = [r["expected"] for r in train_rows_w]
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=actual_w,
            mode="lines+markers",
            name="Actual (revised)",
            line={"color": BLUE, "width": 2.5},
            marker={"size": 7, "color": BLUE},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=expected_w,
            mode="lines+markers",
            name="Expected (revised)",
            line={"color": ORANGE, "width": 2.5},
            marker={"size": 7, "color": ORANGE},
        ),
        secondary_y=False,
    )

    exposures = [r.get("exposure", 0) for r in train_rows]
    fig.add_trace(
        go.Bar(
            x=labels,
            y=exposures,
            name="Exposure",
            marker_color="rgba(180,180,180,0.25)",
            marker_line_width=0,
            showlegend=False,
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=f"Actual vs Expected — {variable}",
        height=height,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    fig.update_xaxes(title_text=variable)
    fig.update_yaxes(title_text="Rate", secondary_y=False)
    fig.update_yaxes(title_text="Exposure", secondary_y=True)

    return fig


def build_actual_vs_expected(
    metrics: dict,
    variable: str,
    height: int = 400,
) -> go.Figure:
    subsets = metrics["subsets"]
    has_split = "train" in subsets and "test" in subsets

    train_rows = subsets.get("train", subsets.get("all", []))
    test_rows = subsets.get("test", [])
    if not has_split and not train_rows:
        train_rows = subsets.get("all", [])
    labels = [r["level"] for r in train_rows] if train_rows else []

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    bar_color_train = "rgba(180,180,180,0.3)"
    bar_color_test = "rgba(180,180,180,0.15)"

    exposures_train = [r.get("exposure", 0) for r in train_rows]
    fig.add_trace(
        go.Bar(
            x=labels,
            y=exposures_train,
            name="Exposure (train)",
            marker_color=bar_color_train,
            marker_line_width=0,
            showlegend=False,
        ),
        secondary_y=True,
    )

    if has_split and test_rows:
        exposures_test = [r.get("exposure", 0) for r in test_rows]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=exposures_test,
                name="Exposure (test)",
                marker_color=bar_color_test,
                marker_line_width=0,
                showlegend=False,
            ),
            secondary_y=True,
        )

    actual_train = [r["actual"] for r in train_rows]
    expected_train = [r["expected"] for r in train_rows]
    suffix = " (train)" if has_split else ""

    fig.add_trace(
        go.Scatter(
            x=labels,
            y=actual_train,
            mode="lines+markers",
            name=f"Actual{suffix}",
            line={"color": BLUE, "width": 2},
            marker={"size": 6, "color": BLUE},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=expected_train,
            mode="lines+markers",
            name=f"Expected{suffix}",
            line={"color": ORANGE, "width": 2},
            marker={"size": 6, "color": ORANGE},
        ),
        secondary_y=False,
    )

    if has_split and test_rows:
        actual_test = [r["actual"] for r in test_rows]
        expected_test = [r["expected"] for r in test_rows]

        fig.add_trace(
            go.Scatter(
                x=labels,
                y=actual_test,
                mode="lines+markers",
                name="Actual (test)",
                line={"color": BLUE, "width": 2, "dash": "dash"},
                marker={"size": 6, "color": BLUE},
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=expected_test,
                mode="lines+markers",
                name="Expected (test)",
                line={"color": ORANGE, "width": 2, "dash": "dash"},
                marker={"size": 6, "color": ORANGE},
            ),
            secondary_y=False,
        )

    fig.update_layout(
        title=f"Actual vs Expected — {variable}",
        height=height,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.2},
    )
    fig.update_xaxes(title_text=variable)
    fig.update_yaxes(title_text="Rate", secondary_y=False)
    fig.update_yaxes(title_text="Exposure", secondary_y=True)

    return fig
