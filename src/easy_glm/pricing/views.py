"""Notebook-friendly tables and charts for the pricing workflow facade."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from textwrap import shorten
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from easy_glm.core.design import NUMERIC_DTYPES, frequent_levels, quantile_knots
from easy_glm.core.excel import pair_table_frames, rate_model_tables
from easy_glm.engine._scoring import row_index
from easy_glm.engine.models import FromToRow, VariableConfig, level_label
from easy_glm.engine.rate_model import RateModel
from easy_glm.workflow.diagnostics import totals
from easy_glm.workflow.run import integer_knots, other_label_for


@dataclass
class DisplayResult:
    """A diagnostic table with an optional Plotly chart.

    Leaving a result as the last expression in a notebook cell displays the
    chart and a compact copy of the table.  ``show()`` provides the same result
    in scripts and IDE notebook windows.
    """

    table: pl.DataFrame
    figure: go.Figure | None = None
    title: str = ""
    note: str = ""

    def _repr_html_(self) -> str:
        chart = (
            self.figure.to_html(full_html=False, include_plotlyjs="cdn")
            if self.figure is not None
            else ""
        )
        heading = f"<h3>{escape(self.title)}</h3>" if self.title else ""
        note = f"<p>{escape(self.note)}</p>" if self.note else ""
        return (
            heading
            + chart
            + note
            + self.table.head(200).to_pandas().to_html(index=False)
        )

    def show(self) -> DisplayResult:
        """Display in IPython when available and return this result."""
        try:
            from IPython.display import display

            display(self)
        except ImportError:
            if self.figure is not None:
                self.figure.show()
            print(self.table)
        return self


def _frame(model: Any, subset: str) -> pl.DataFrame:
    try:
        frame = model._frame(subset)
    except (AttributeError, TypeError) as exc:
        raise ValueError(
            "Diagnostics need policy data. Load the saved model with data=... first."
        ) from exc
    if frame is None or frame.is_empty():
        raise ValueError(f"The {subset!r} subset has no rows")
    return frame


def _actual_expected(
    model: Any, frame: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = model._run.rate_model.predict(frame, exposure_col=None)
    return totals(frame, model._run.config, prediction)


def _axis_table(
    frame: pl.DataFrame,
    variable: str,
    axis: VariableConfig,
    actual: np.ndarray,
    expected: np.ndarray,
    weight: np.ndarray,
) -> pl.DataFrame:
    indices = row_index(frame[variable], axis)
    rows = []
    for index, band in enumerate(axis.table):
        selected = indices == index
        exposure = float(weight[selected].sum())
        observed = float(actual[selected].sum())
        fitted = float(expected[selected].sum())
        rows.append(
            {
                "label": level_label(band, axis.other_label),
                "exposure": exposure,
                "actual": observed,
                "expected": fitted,
                "ae": observed / fitted if fitted > 0 else None,
                "actual_rate": observed / exposure if exposure > 0 else None,
                "expected_rate": fitted / exposure if exposure > 0 else None,
                "order": index,
            }
        )
    return pl.DataFrame(rows)


def _pair_table(
    frame: pl.DataFrame,
    names: tuple[str, str],
    axes: tuple[VariableConfig, VariableConfig],
    actual: np.ndarray,
    expected: np.ndarray,
    weight: np.ndarray,
) -> pl.DataFrame:
    ia = row_index(frame[names[0]], axes[0])
    ib = row_index(frame[names[1]], axes[1])
    rows = []
    for a_index, a_row in enumerate(axes[0].table):
        for b_index, b_row in enumerate(axes[1].table):
            selected = (ia == a_index) & (ib == b_index)
            exposure = float(weight[selected].sum())
            observed = float(actual[selected].sum())
            fitted = float(expected[selected].sum())
            rows.append(
                {
                    "label_a": level_label(a_row, axes[0].other_label),
                    "label_b": level_label(b_row, axes[1].other_label),
                    "exposure": exposure,
                    "actual": observed,
                    "expected": fitted,
                    "ae": observed / fitted if fitted > 0 else None,
                    "actual_rate": observed / exposure if exposure > 0 else None,
                    "expected_rate": fitted / exposure if exposure > 0 else None,
                    "order_a": a_index,
                    "order_b": b_index,
                }
            )
    return pl.DataFrame(rows)


def _candidate_axis(model: Any, train: pl.DataFrame, variable: str) -> VariableConfig:
    """Build an omitted factor's saved design from training rows only."""
    series = train[variable]
    design = model._project.design.variables.get(variable)
    defaults = model._project.design.defaults
    numeric = series.dtype in NUMERIC_DTYPES
    kind = design.kind if design is not None else None
    if kind == "categorical" or not numeric:
        if design is not None and design.levels:
            levels = [str(value) for value in design.levels]
        else:
            minimum = (
                design.min_level_share
                if design is not None and design.min_level_share is not None
                else defaults.min_level_share
            )
            weight_col = model._run.config.weight
            levels = frequent_levels(
                series,
                min_share=minimum,
                max_levels=design.max_levels if design is not None else None,
                weights=train[weight_col] if weight_col else None,
            )
        axis = VariableConfig(
            "categorical",
            [FromToRow(level, level, 1.0) for level in levels]
            + [FromToRow(None, None, 1.0)],
            other_label=other_label_for(levels),
        )
    else:
        if design is not None and isinstance(design.knots, (list, tuple)):
            cuts = [float(value) for value in design.knots]
        elif design is not None and design.knots == "integer":
            cuts = integer_knots(series, defaults.max_integer_knots) or quantile_knots(
                series, design.n_bins or defaults.n_bins
            )
        else:
            cuts = quantile_knots(
                series, design.n_bins if design and design.n_bins else defaults.n_bins
            )
        cuts = sorted(set(cuts or []))
        if not cuts:
            median = series.cast(pl.Float64).drop_nulls().median()
            cuts = [float(cast(Any, median)) if median is not None else 0.0]
        rows = [FromToRow(None, cuts[0], 1.0)]
        rows.extend(
            FromToRow(left, right, 1.0)
            for left, right in zip(cuts[:-1], cuts[1:], strict=True)
        )
        rows.extend([FromToRow(cuts[-1], None, 1.0), FromToRow(None, None, 1.0)])
        axis = VariableConfig("numeric", rows)
    RateModel._precompute_variables({variable: axis})
    return axis


def _plot_title(model_name: str, detail: str) -> dict[str, Any]:
    """Bounded two-line title that stays inside narrow static exports."""
    model_line = shorten(model_name, width=64, placeholder="…")
    detail_line = shorten(detail, width=72, placeholder="…")
    return {
        "text": f"<b>{escape(model_line)}</b><br><span>{escape(detail_line)}</span>",
        "x": 0.02,
        "xanchor": "left",
        "y": 0.94,
        "yanchor": "top",
    }


def _ae_chart(table: pl.DataFrame, model_name: str, detail: str) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    labels = table["label"].to_list()
    figure.add_bar(
        x=labels,
        y=table["exposure"],
        name="exposure",
        marker_color="#c9cfd6",
        opacity=0.7,
        secondary_y=False,
    )
    figure.add_scatter(
        x=labels,
        y=table["actual_rate"],
        name="actual rate",
        mode="lines+markers",
        line={"color": "#1f5f99", "width": 2.5},
        secondary_y=True,
    )
    figure.add_scatter(
        x=labels,
        y=table["expected_rate"],
        name="expected rate",
        mode="lines+markers",
        line={"color": "#e07b39", "width": 2.5},
        secondary_y=True,
    )
    figure.update_yaxes(title_text="exposure", secondary_y=False, showgrid=False)
    figure.update_yaxes(title_text="rate", secondary_y=True, rangemode="tozero")
    figure.update_layout(
        title=_plot_title(model_name, detail),
        template="plotly_white",
        height=400,
        xaxis={"type": "category", "tickangle": -45},
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": 1.03,
            "yanchor": "bottom",
        },
        margin={"l": 70, "r": 70, "t": 125, "b": 115},
    )
    return figure


def _heatmap(
    table: pl.DataFrame,
    value: str,
    a: str,
    b: str,
    model_name: str,
    detail: str,
) -> go.Figure:
    order_a = "order_a" if "order_a" in table.columns else "axis_a_row"
    order_b = "order_b" if "order_b" in table.columns else "axis_b_row"
    rows = table.select(order_a, "label_a").unique().sort(order_a)["label_a"].to_list()
    columns = (
        table.select(order_b, "label_b").unique().sort(order_b)["label_b"].to_list()
    )
    values: list[list[float | None]] = []
    exposure: list[list[float]] = []
    statuses: list[list[str]] = []
    labels: list[list[str]] = []
    for row in rows:
        values.append([])
        exposure.append([])
        statuses.append([])
        labels.append([])
        for column in columns:
            cell = table.filter(
                (pl.col("label_a") == row) & (pl.col("label_b") == column)
            )
            raw = cell[value][0] if cell.height else None
            rounded = None if raw is None else round(float(raw), 3)
            status = (
                str(cell["support_status"][0])
                if cell.height and "support_status" in cell.columns
                else "observed policies"
            )
            values[-1].append(rounded)
            exposure[-1].append(float(cell["exposure"][0]) if cell.height else 0.0)
            statuses[-1].append(status)
            marker = "*" if status not in {"observed cell", "observed policies"} else ""
            labels[-1].append("—" if rounded is None else f"{rounded:.3f}{marker}")
    custom = [
        [[values[i][j], exposure[i][j], statuses[i][j]] for j in range(len(columns))]
        for i in range(len(rows))
    ]
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=columns,
            y=rows,
            customdata=custom,
            text=labels,
            texttemplate="%{text}",
            colorscale="RdBu_r",
            zmid=1.0,
            xgap=1,
            ygap=1,
            hovertemplate=(
                f"{a}: %{{y}}<br>{b}: %{{x}}<br>{value}: %{{customdata[0]:.3f}}"
                "<br>support: %{customdata[1]:,.1f}"
                "<br>status: %{customdata[2]}<extra></extra>"
            ),
            colorbar={"title": value},
        )
    )
    figure.update_layout(
        title=_plot_title(model_name, detail),
        template="plotly_white",
        height=480,
        xaxis={"title": b, "tickangle": -45},
        yaxis={"title": a, "autorange": "reversed"},
        margin={"l": 90, "r": 90, "t": 105, "b": 125},
    )
    return figure


def ae(
    model: Any, a: str, b: str | None = None, *, subset: str = "train"
) -> DisplayResult:
    """Actual-versus-expected by one factor or a two-factor grid."""
    frame = _frame(model, subset)
    actual, expected, weight = _actual_expected(model, frame)
    scorer = model._run.rate_model
    if b is None:
        if a in scorer.variables and scorer.variables[a].type != "interaction":
            table = _axis_table(frame, a, scorer.variables[a], actual, expected, weight)
        else:
            train = _frame(model, "train")
            table = _axis_table(
                frame, a, _candidate_axis(model, train, a), actual, expected, weight
            )
        title = f"{model.name}: {subset} actual versus expected by {a}"
        detail = f"{subset} actual versus expected by {a}"
        return DisplayResult(table, _ae_chart(table, model.name, detail), title)

    deployed = next(
        (table for table in scorer.pair_tables if set(table.parents) == {a, b}), None
    )
    if deployed is not None:
        if deployed.parents == (a, b):
            names, axes = deployed.parents, deployed.axes
        else:
            names, axes = (a, b), (deployed.axes[1], deployed.axes[0])
        table = _pair_table(frame, names, axes, actual, expected, weight)
    else:
        train = _frame(model, "train")
        table = _pair_table(
            frame,
            (a, b),
            (_candidate_axis(model, train, a), _candidate_axis(model, train, b)),
            actual,
            expected,
            weight,
        )
    title = f"{model.name}: {subset} A/E by {a} × {b}"
    detail = f"{subset} A/E by {a} × {b}"
    return DisplayResult(table, _heatmap(table, "ae", a, b, model.name, detail), title)


def relativities(
    model: Any, a: str | None = None, b: str | None = None
) -> DisplayResult:
    """Show the exact current scoring table for a main factor or pair stage."""
    scorer = model._run.rate_model
    if a is None:
        frames = []
        for variable, table in rate_model_tables(scorer).items():
            frames.append(table.with_columns(pl.lit(variable).alias("variable")))
        combined = (
            pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
        )
        return DisplayResult(combined, title=f"{model.name}: current main tables")
    if b is None:
        try:
            table = rate_model_tables(scorer)[a]
        except KeyError as exc:
            raise ValueError(f"{a!r} is not a fitted main factor") from exc
        figure = go.Figure()
        figure.add_scatter(
            x=table["label"],
            y=table["relativity"],
            mode="lines+markers",
            name="current",
        )
        figure.add_hline(y=1.0, line_dash="dot", line_color="#999999")
        figure.update_layout(
            title=_plot_title(model.name, f"{a} relativity"),
            template="plotly_white",
            height=380,
            xaxis={"type": "category", "tickangle": -45},
            yaxis={"title": "relativity"},
            margin={"l": 70, "r": 60, "t": 100, "b": 110},
        )
        return DisplayResult(table, figure, f"{model.name}: {a} relativity")

    deployed = next(
        (table for table in scorer.pair_tables if set(table.parents) == {a, b}), None
    )
    if deployed is None:
        raise ValueError(f"No deployed pair table for {a!r} × {b!r}")
    table = pair_table_frames(scorer)[deployed.stage_id]
    if deployed.parents != (a, b):
        swapped = {
            "parent_a": "parent_b",
            "parent_b": "parent_a",
            "axis_a_row": "axis_b_row",
            "axis_b_row": "axis_a_row",
            "from_a": "from_b",
            "from_b": "from_a",
            "to_a": "to_b",
            "to_b": "to_a",
            "label_a": "label_b",
            "label_b": "label_a",
        }
        temporary = {name: f"__swap_{name}" for name in swapped}
        table = table.rename(temporary).rename(
            {temporary[name]: destination for name, destination in swapped.items()}
        )
    support_name = (
        "exposure_total"
        if "exposure_total" in table.columns
        and table["exposure_total"].null_count() < table.height
        else "fitting_weight"
    )
    table = table.with_columns(
        pl.col(support_name).alias("exposure"),
        pl.lit(support_name).alias("support_source"),
        pl.col("fallback_reason").fill_null("observed cell").alias("support_status"),
    )
    title = f"{model.name}: {a} × {b} deployed relativity"
    return DisplayResult(
        table,
        _heatmap(
            table,
            "relativity",
            a,
            b,
            model.name,
            f"{a} × {b} deployed relativity",
        ),
        title,
        "Support is shown from the deployed table; fallback and unrepresented cells are explicit.",
    )
