"""Standalone SVGs for permutation importance and stored coefficient paths."""

from __future__ import annotations

import math
from html import escape
from numbers import Integral
from typing import Any

from ._svg import AXIS, BLUE, GRID, ORANGE, nice_ticks

TEXT = "#1c2530"
MUTED = "#b9c3cd"
PATH_COLOURS = (BLUE, ORANGE, "#496f8b", "#806692", "#267e79")


def _frame(width: int, height: int, body: str, title: str, kind: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="chart {kind}" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'role="img" style="max-width:100%;height:auto;background:#fff;'
        'font-family:Helvetica,Arial,sans-serif">'
        f"<title>{escape(title)}</title>{body}</svg>"
    )


def _text(
    x: float,
    y: float,
    value: str,
    *,
    anchor: str = "start",
    colour: str = AXIS,
    size: int = 12,
    kind: str = "diagnostic-label",
    tooltip: str | None = None,
) -> str:
    hint = f"<title>{escape(tooltip)}</title>" if tooltip is not None else ""
    return (
        f'<text class="{kind}" x="{x:.3f}" y="{y:.3f}" '
        f'text-anchor="{anchor}" fill="{colour}" font-size="{size}">'
        f"{hint}{escape(value)}</text>"
    )


def _short(value: str, length: int) -> str:
    return value if len(value) <= length else value[: length - 1] + "…"


def _number(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 0.01 or abs(value) >= 1_000_000:
        mantissa, exponent = format(value, ".3e").split("e")
        return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent):+d}"
    return format(value, ".3f").rstrip("0").rstrip(".")


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("Chart values must be numeric or missing.") from exc
    return converted if math.isfinite(converted) else None


def _domain(values: list[float]) -> tuple[float, float]:
    """A linear domain containing zero, with no arbitrary minimum data scale."""
    low, high = min([0.0, *values]), max([0.0, *values])
    if low == high:
        return -1.0, 1.0
    span = high - low
    if not math.isfinite(span):
        raise ValueError("Chart values span more than the supported numeric range.")
    padding = span * 0.06
    padded_low = low - (padding if low < 0 else 0)
    padded_high = high + (padding if high > 0 else 0)
    if not math.isfinite(padded_high - padded_low):
        return low, high
    return padded_low, padded_high


def _ticks(low: float, high: float, values: list[float]) -> list[float]:
    if not values or all(value == 0 for value in values):
        return [0.0]
    ticks = (
        [value for value in nice_ticks(low, high, 4) if low <= value <= high]
        if (high - low) / 4 >= 1e-300
        else []
    )
    # nice_ticks rounds to 12 decimal places; tiny diagnostic changes need their
    # own scale rather than several rounded copies of zero.
    if len(set(ticks)) < 2:
        ticks = [low, (low + high) / 2, high]
    return sorted({*ticks, 0.0})


def permutation_importance_chart(
    rows: list[dict[str, Any]],
    *,
    title: str,
    x_label: str = "Increase in mean deviance",
) -> str:
    """Draw already ranked ``variable/importance/std`` rows without reordering.

    Bars start at zero. Blue means positive and orange means negative; whiskers
    show exactly mean ± one supplied standard deviation. Missing estimates stay
    visible as undefined rows, and missing SD never becomes a zero-length SD.
    The caller may select a subset of rows for a compact report.
    """
    width = 900
    if not rows:
        return _frame(
            width,
            110,
            _text(24, 54, "No permutation importance available"),
            title,
            "permutation-importance-chart",
        )
    checked = []
    values = []
    for row in rows:
        label = str(row["variable"])
        importance, std = _finite(row.get("importance")), _finite(row.get("std"))
        if std is not None and std < 0:
            raise ValueError("Importance standard deviations cannot be negative.")
        if importance is not None:
            values.append(importance)
            if std is not None:
                lower, upper = importance - std, importance + std
                if not math.isfinite(lower) or not math.isfinite(upper):
                    raise ValueError("Importance ± standard deviation must be finite.")
                values.extend((lower, upper))
        checked.append((label, importance, std))
    low, high = _domain(values)
    left, right, top, row_height = 230, 772, 34, 28
    bottom = top + row_height * len(rows)
    height = bottom + 65

    def x(value: float) -> float:
        return left + (value - low) / (high - low) * (right - left)

    out = [
        _text(20, 18, "Predictor", colour=TEXT),
        _text(884, 18, "Mean", anchor="end", colour=TEXT),
    ]
    if any(std is not None for _, importance, std in checked if importance is not None):
        out.append(_text(left, 18, "Whiskers: ±1 SD", size=11))
    if values:
        for tick in _ticks(low, high, values):
            out += [
                f'<line class="importance-grid" x1="{x(tick):.3f}" x2="{x(tick):.3f}" '
                f'y1="{top}" y2="{bottom}" stroke="{GRID}"/>',
                _text(x(tick), bottom + 21, _number(tick), anchor="middle", size=11),
            ]
        out.append(
            f'<line class="importance-zero" x1="{x(0):.3f}" x2="{x(0):.3f}" '
            f'y1="{top}" y2="{bottom}" stroke="{AXIS}" stroke-width="1.3"/>'
        )
    for index, (label, importance, std) in enumerate(checked):
        y = top + (index + 0.5) * row_height
        description = (
            "Mean: undefined"
            if importance is None
            else f"Mean: {_number(importance)}\n"
            + (
                "SD: unavailable"
                if std is None
                else f"SD: {_number(std)} (±1 SD whisker)"
            )
        )
        out.append(
            f'<g class="importance-row" data-variable="{escape(label, quote=True)}">'
            f"<title>{escape(label)}\n{escape(description)}</title>"
        )
        out.append(
            _text(
                left - 14,
                y + 4,
                _short(label, 29),
                anchor="end",
                colour=TEXT,
                tooltip=label,
                kind="importance-variable",
            )
        )
        if importance is None:
            out.append(_text((left + right) / 2, y + 4, "Undefined", anchor="middle"))
        else:
            colour = BLUE if importance >= 0 else ORANGE
            start, end = x(0), x(importance)
            out.append(
                f'<rect class="importance-bar" x="{min(start, end):.3f}" y="{y - 7:.3f}" '
                f'width="{abs(end - start):.3f}" height="14" fill="{colour}" fill-opacity="0.8"/>'
            )
            if std is not None:
                lower, upper = x(importance - std), x(importance + std)
                out.append(
                    f'<line class="importance-whisker" x1="{lower:.3f}" x2="{upper:.3f}" '
                    f'y1="{y:.3f}" y2="{y:.3f}" stroke="{TEXT}" stroke-width="1.3"/>'
                )
                for point in (lower, upper):
                    out.append(
                        f'<line x1="{point:.3f}" x2="{point:.3f}" y1="{y - 4:.3f}" '
                        f'y2="{y + 4:.3f}" stroke="{TEXT}" stroke-width="1.3"/>'
                    )
            out.append(
                f'<circle class="importance-mean" cx="{end:.3f}" cy="{y:.3f}" '
                f'r="2.6" fill="{TEXT}"/>'
            )
        out.append(
            _text(
                884,
                y + 4,
                "—" if importance is None else _number(importance),
                anchor="end",
                colour=TEXT,
                kind="importance-value",
            )
            + "</g>"
        )
    out.append(_text((left + right) / 2, bottom + 47, x_label, anchor="middle"))
    return _frame(width, height, "".join(out), title, "permutation-importance-chart")


def _positive_alpha(value: Any) -> float:
    alpha = _finite(value)
    if alpha is None or alpha <= 0:
        raise ValueError(
            "A logarithmic coefficient chart requires positive finite alpha values."
        )
    return alpha


def _path_groups(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    groups: dict[int, dict[str, Any]] = {}
    contexts = {(row.get("stage"), row.get("l1_ratio")) for row in rows}
    if len(contexts) > 1:
        raise ValueError("Supply only one stage and l1_ratio per coefficient chart.")
    for row in rows:
        index = row["feature_index"]
        if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
            raise ValueError("Each coefficient requires a nonnegative feature_index.")
        index = int(index)
        alpha = _positive_alpha(row["alpha"])
        label = str(row["feature"])
        group = groups.setdefault(
            index,
            {"label": label, "variable": str(row.get("variable", "")), "points": {}},
        )
        if group["label"] != label or alpha in group["points"]:
            raise ValueError(
                "Each feature_index must have one label and one value per alpha."
            )
        group["points"][alpha] = _finite(row.get("coefficient"))
    return groups


def _log_ticks(alphas: list[float], low: float, high: float) -> list[float]:
    if len(alphas) == 1:
        return alphas
    start, stop = math.ceil(low), math.floor(high)
    step = max(1, math.ceil((stop - start) / 5))
    ticks = [
        10.0**power for power in range(start, stop + 1, step) if -323 <= power <= 308
    ]
    if len(ticks) < 2:
        ticks = [min(alphas), max(alphas)]
    return ticks


def coefficient_path_chart(
    rows: list[dict[str, Any]],
    *,
    title: str,
    selected_alpha: float | None = None,
) -> str:
    """Plot stored coefficient rows for one stage/l1 ratio on a log-alpha axis.

    Rows contain ``feature_index``, ``feature``, ``variable``, ``alpha`` and
    ``coefficient``. Identity comes from feature_index, never a parsed label.
    Supplied values are joined by straight segments in log-alpha coordinates;
    missing/non-finite coefficients or absent knots leave gaps. Isolated values
    are points, not invented paths. No model fitting or smoothing occurs.

    All trajectories remain visible. Up to five paths with the largest peak
    absolute coefficient are coloured and keyed; others are muted with full
    native tooltips. A caller exporting alpha=0 must describe/exclude those
    points before using this logarithmic chart.
    """
    width = 900
    selected = _positive_alpha(selected_alpha) if selected_alpha is not None else None
    groups = _path_groups(rows)
    finite_values = [
        value
        for group in groups.values()
        for value in group["points"].values()
        if value is not None
    ]
    if not finite_values:
        return _frame(
            width,
            110,
            _text(24, 54, "No finite coefficient values available"),
            title,
            "coefficient-path-chart",
        )
    alphas = sorted({alpha for group in groups.values() for alpha in group["points"]})
    axis_alphas = sorted(set(alphas + ([selected] if selected is not None else [])))
    log_low, log_high = math.log10(axis_alphas[0]), math.log10(axis_alphas[-1])
    if log_low == log_high:
        log_low, log_high = log_low - 0.5, log_high + 0.5
    else:
        padding = (log_high - log_low) * 0.025
        log_low, log_high = log_low - padding, log_high + padding
    low, high = _domain(finite_values)
    left, right, top, bottom = 78, 878, 42, 282

    def x(alpha: float) -> float:
        return left + (math.log10(alpha) - log_low) / (log_high - log_low) * (
            right - left
        )

    def y(coefficient: float) -> float:
        return bottom - (coefficient - low) / (high - low) * (bottom - top)

    ranked = sorted(
        (
            index
            for index, group in groups.items()
            if any(value is not None for value in group["points"].values())
        ),
        key=lambda index: (
            -max(
                abs(value)
                for value in groups[index]["points"].values()
                if value is not None
            ),
            index,
        ),
    )
    highlighted = {
        index: PATH_COLOURS[position] for position, index in enumerate(ranked[:5])
    }
    out = [_text(left, 17, "Coefficient", colour=TEXT)]
    for tick in _ticks(low, high, finite_values):
        out += [
            f'<line class="coefficient-grid" x1="{left}" x2="{right}" '
            f'y1="{y(tick):.3f}" y2="{y(tick):.3f}" stroke="{GRID}"/>',
            _text(left - 10, y(tick) + 4, _number(tick), anchor="end", size=11),
        ]
    out.append(
        f'<line class="coefficient-zero" x1="{left}" x2="{right}" '
        f'y1="{y(0):.3f}" y2="{y(0):.3f}" stroke="{AXIS}" stroke-dasharray="3 4"/>'
    )
    for tick in _log_ticks(axis_alphas, log_low, log_high):
        out += [
            f'<line x1="{x(tick):.3f}" x2="{x(tick):.3f}" y1="{bottom}" '
            f'y2="{bottom + 5}" stroke="{AXIS}"/>',
            _text(x(tick), bottom + 22, _number(tick), anchor="middle", size=11),
        ]
    out.append(
        _text((left + right) / 2, bottom + 46, "Penalty λ (log scale)", anchor="middle")
    )
    draw_order = [index for index in groups if index not in highlighted] + ranked[:5]
    for index in draw_order:
        group = groups[index]
        colour = highlighted.get(index, MUTED)
        tooltip = group["label"]
        if group["variable"]:
            tooltip += "\nVariable: " + group["variable"]
        out.append(
            f'<g class="coefficient-trajectory" data-feature-index="{index}" '
            f'data-highlighted="{str(index in highlighted).lower()}">'
            f"<title>{escape(tooltip)}</title>"
        )
        segments: list[list[tuple[float, float]]] = []
        segment: list[tuple[float, float]] = []
        for alpha in alphas:
            coefficient = group["points"].get(alpha)
            if coefficient is None:
                if segment:
                    segments.append(segment)
                    segment = []
            else:
                segment.append((alpha, coefficient))
        if segment:
            segments.append(segment)
        for segment in segments:
            if len(segment) > 1:
                path = " ".join(
                    f'{"M" if point == 0 else "L"}{x(alpha):.3f},{y(coefficient):.3f}'
                    for point, (alpha, coefficient) in enumerate(segment)
                )
                out.append(
                    f'<path class="coefficient-curve" d="{path}" fill="none" '
                    f'stroke="{colour}" stroke-width="{2 if index in highlighted else 1}" '
                    f'stroke-opacity="{1 if index in highlighted else 0.7}"/>'
                )
            if index in highlighted or len(segment) == 1:
                for alpha, coefficient in segment:
                    point_title = f'{group["label"]}\nλ = {_number(alpha)}\nCoefficient = {_number(coefficient)}'
                    out.append(
                        f'<circle class="coefficient-point" cx="{x(alpha):.3f}" '
                        f'cy="{y(coefficient):.3f}" r="2.4" fill="{colour}" '
                        f'data-alpha="{alpha}" data-coefficient="{coefficient}">'
                        f"<title>{escape(point_title)}</title></circle>"
                    )
        out.append("</g>")
    if selected is not None:
        position = x(selected)
        anchor = (
            "start"
            if position < left + 140
            else "end" if position > right - 140 else "middle"
        )
        out += [
            f'<line class="selected-alpha" x1="{position:.3f}" x2="{position:.3f}" '
            f'y1="{top}" y2="{bottom}" stroke="{TEXT}" stroke-width="1.5" stroke-dasharray="5 4">'
            f"<title>Selected λ = {escape(_number(selected))}</title></line>",
            _text(
                position,
                top - 10,
                "Selected λ = " + _number(selected),
                anchor=anchor,
                colour=TEXT,
                size=11,
            ),
        ]
    legend_top = bottom + 73
    coloured_count, muted_count = len(highlighted), len(ranked) - len(highlighted)
    description = f"{len(ranked)} coefficient {'path' if len(ranked) == 1 else 'paths'}"
    if muted_count:
        description += f" · {coloured_count} largest peak magnitudes coloured · {muted_count} in grey"
    out.append(_text(24, legend_top, description, size=11))
    for position, (index, colour) in enumerate(highlighted.items()):
        legend_x, legend_y = (
            24 + (position % 2) * 430,
            legend_top + 25 + (position // 2) * 22,
        )
        out += [
            f'<line x1="{legend_x}" x2="{legend_x + 18}" y1="{legend_y - 4}" '
            f'y2="{legend_y - 4}" stroke="{colour}" stroke-width="2"/>',
            _text(
                legend_x + 25,
                legend_y,
                _short(groups[index]["label"], 44),
                colour=TEXT,
                tooltip=groups[index]["label"],
                kind="coefficient-legend-label",
            ),
        ]
    height = legend_top + 25 + max(0, (coloured_count - 1) // 2) * 22 + 18
    return _frame(width, height, "".join(out), title, "coefficient-path-chart")
