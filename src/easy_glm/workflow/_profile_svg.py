"""Compact, self-contained SVG charts for the report's data profile."""

from __future__ import annotations

import math
from html import escape
from numbers import Integral

from ._svg import AXIS, BLUE, GREY, GRID, ORANGE

TEXT = "#1c2530"
WHITE = "#ffffff"


def _frame(width: int, height: int, body: str, title: str, kind: str) -> str:
    return (
        f'<svg class="chart {kind}" xmlns="http://www.w3.org/2000/svg" '
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
    size: int = 10,
    colour: str = AXIS,
    tooltip: str | None = None,
    kind: str = "profile-label",
) -> str:
    hint = f"<title>{escape(tooltip)}</title>" if tooltip is not None else ""
    return (
        f'<text class="{kind}" x="{x:.1f}" y="{y:.1f}" '
        f'text-anchor="{anchor}" font-size="{size}" fill="{colour}">'
        f"{hint}{escape(value)}</text>"
    )


def _short(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("Row counts must be nonnegative integers.")
    return int(value)


def _count_tick(value: int) -> str:
    for scale, suffix in ((1_000_000_000, "b"), (1_000_000, "m"), (1_000, "k")):
        if value >= scale:
            return f"{value / scale:.1f}".rstrip("0").rstrip(".") + suffix
    return str(value)


def _histogram_label_width(value: str) -> float:
    """Conservative Helvetica widths at 13px, including unusually wide labels."""
    return sum(
        (
            13.0
            if letter in "MWmw@%…" or ord(letter) >= 0x2E80
            else 10.5 if letter.isupper() else 8.0
        )
        for letter in value
    )


def _histogram_label(value: str, width: float = 112) -> str:
    if _histogram_label_width(value) <= width:
        return value
    shortened = value
    while shortened and _histogram_label_width(shortened + "…") > width:
        shortened = shortened[:-1]
    return shortened + "…"


def mini_histogram(labels: list[str], counts: list[int], *, title: str) -> str:
    """Draw at most 16 ordered bins/levels in a 320 × 110 chart.

    The caller supplies display labels and counts in their intended order.
    Sparse axis labels are shortened; every bar retains its full label/count.
    Empty input is an explicit empty chart, not a fabricated zero-valued bin.
    """
    if len(labels) != len(counts) or len(labels) > 16:
        raise ValueError("Supply matching labels/counts for at most 16 bars.")
    labels = [str(label) for label in labels]
    counts = [_count(value) for value in counts]
    width, height = 320, 110
    if not labels:
        return _frame(
            width,
            height,
            _text(width / 2, 57, "No observed values", anchor="middle"),
            title,
            "profile-histogram",
        )
    left, right, top, baseline = 48, 312, 18, 82
    maximum = max(counts)
    scale = maximum or 1
    out = [
        _text(left - 5, baseline + 4, "0", anchor="end", size=13),
        _text(left, 12, "Rows", size=13),
    ]
    if maximum:
        out += [
            f'<line x1="{left}" y1="{top}" x2="{right}" y2="{top}" stroke="{GRID}"/>',
            _text(left - 5, top + 4, _count_tick(maximum), anchor="end", size=13),
        ]
    span = (right - left) / len(labels)
    gap = min(3.0, span * 0.2)
    for index, (label, count) in enumerate(zip(labels, counts, strict=True)):
        x = left + index * span + gap / 2
        bar_height = (baseline - top) * count / scale
        out.append(
            '<g class="profile-bin">'
            f"<title>{escape(label)}: {count:,} {'row' if count == 1 else 'rows'}</title>"
            f'<rect x="{x:.2f}" y="{top}" width="{span - gap:.2f}" '
            f'height="{baseline - top}" fill="transparent"/>'
            f'<rect class="profile-bar" x="{x:.2f}" y="{baseline - bar_height:.2f}" '
            f'width="{span - gap:.2f}" height="{bar_height:.2f}" fill="{BLUE}"/>'
            "</g>"
        )
    out.append(
        f'<line class="profile-baseline" x1="{left}" y1="{baseline}" '
        f'x2="{right}" y2="{baseline}" stroke="{AXIS}"/>'
    )
    if len(labels) == 1:
        out.append(
            _text(
                (left + right) / 2,
                102,
                _histogram_label(labels[0], right - left),
                anchor="middle",
                size=13,
                tooltip=labels[0],
                kind="profile-bin-label",
            )
        )
    else:
        edges = [(0, left, "start"), (len(labels) - 1, right, "end")]
        for index, x, anchor in edges:
            out.append(
                _text(
                    x,
                    102,
                    _histogram_label(labels[index]),
                    anchor=anchor,
                    size=13,
                    tooltip=labels[index],
                    kind="profile-bin-label",
                )
            )
        if len(labels) > 2:
            middle = len(labels) // 2
            shown = _histogram_label(labels[middle])
            x = left + (middle + 0.5) * span
            # Estimate label widths conservatively; never force a final tick.
            first_end = left + _histogram_label_width(_histogram_label(labels[0]))
            last_start = right - _histogram_label_width(_histogram_label(labels[-1]))
            middle_width = _histogram_label_width(shown)
            if first_end + 8 < x - middle_width / 2 and (
                x + middle_width / 2 < last_start - 8
            ):
                out.append(
                    _text(
                        x,
                        102,
                        shown,
                        anchor="middle",
                        size=13,
                        tooltip=labels[middle],
                        kind="profile-bin-label",
                    )
                )
    return _frame(width, height, "".join(out), title, "profile-histogram")


def _correlation(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    if abs(value) > 1 + 1e-12:
        raise ValueError("Correlations must lie between -1 and +1.")
    return min(1.0, max(-1.0, float(value)))


def _colour(value: float | None) -> str:
    if value is None:
        return GREY
    endpoint = BLUE if value < 0 else ORANGE
    components = [int(endpoint[index : index + 2], 16) for index in (1, 3, 5)]
    return "#" + "".join(
        f"{round(255 + abs(value) * (component - 255)):02x}" for component in components
    )


def _foreground(colour: str) -> str:
    channels = [int(colour[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    luminance = sum(
        value * factor
        for value, factor in zip(linear, (0.2126, 0.7152, 0.0722), strict=True)
    )
    return WHITE if luminance < 0.179 else "#000000"


def correlation_matrix(
    names: list[str],
    values: list[list[float | None]],
    counts: list[list[int]],
    *,
    title: str,
) -> str:
    """Draw up to 20 numeric variables with fixed -1/0/+1 colour meaning.

    Columns use one-based indices; indexed row labels supply their key. Long
    row names are shortened with full native tooltips. At 20 variables the
    chart is 776 × 676, keeping cells legible in a printable report. Undefined
    correlations (including non-finite values) stay grey and display an em dash.
    Every cell tooltip includes both full names, r and its paired row count.
    """
    n = len(names)
    if n > 20 or len(values) != n or len(counts) != n:
        raise ValueError("Supply matching square matrices for at most 20 variables.")
    if any(len(row) != n for row in values + counts):
        raise ValueError("Correlation values and counts must be square matrices.")
    names = [str(name) for name in names]
    checked_values = [[_correlation(value) for value in row] for row in values]
    checked_counts = [[_count(value) for value in row] for row in counts]
    if not n:
        return _frame(
            320,
            110,
            _text(160, 57, "No numeric variables", anchor="middle"),
            title,
            "profile-correlation",
        )
    cell = 28 if n > 10 else 38
    left, top = 200, 46
    width, height = max(340, left + n * cell + 16), top + n * cell + 70
    out = [_text(24, 13, "Column numbers match the row labels", size=10)]
    for index, name in enumerate(names):
        number = str(index + 1)
        out.append(
            _text(
                left + (index + 0.5) * cell,
                top - 10,
                number,
                anchor="middle",
                tooltip=f"{number} · {name}",
                kind="profile-column-label",
            )
        )
        out.append(
            _text(
                left - 8,
                top + (index + 0.5) * cell + 3.5,
                f"{number} · {_short(name, 25)}",
                anchor="end",
                tooltip=name,
                kind="profile-row-label",
            )
        )
        for column in range(n):
            value = checked_values[index][column]
            colour = _colour(value)
            shown = "—" if value is None else f"{value:.2f}".replace("-0.00", "0.00")
            exact = (
                "undefined"
                if value is None
                else f"{value:.3f}".replace("-0.000", "0.000")
            )
            tooltip = (
                f"{name} × {names[column]}\nr = {exact}\n"
                f"Paired rows: {checked_counts[index][column]:,}"
            )
            x, y = left + column * cell, top + index * cell
            out.append(
                f'<g class="profile-correlation-cell" data-row="{index}" '
                f'data-column="{column}"><title>{escape(tooltip)}</title>'
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{colour}" stroke="{GRID}" stroke-width="1"/>'
                + _text(
                    x + cell / 2,
                    y + cell / 2 + 3.5,
                    shown,
                    anchor="middle",
                    size=9 if n > 10 else 10,
                    colour=_foreground(colour),
                    kind="profile-correlation-value",
                )
                + "</g>"
            )
    legend_x, legend_y, legend_width = 24, top + n * cell + 24, 180
    steps = 21
    for index in range(steps):
        out.append(
            f'<rect x="{legend_x + index * legend_width / steps:.2f}" '
            f'y="{legend_y}" width="{legend_width / steps + 0.05:.2f}" height="10" '
            f'fill="{_colour(-1 + index / 10)}"/>'
        )
    for legend_label, position in (("−1", 0), ("0", 0.5), ("+1", 1)):
        out.append(
            _text(
                legend_x + position * legend_width,
                legend_y + 24,
                legend_label,
                anchor="middle",
            )
        )
    out += [
        f'<rect x="230" y="{legend_y}" width="12" height="10" fill="{GREY}"/>',
        _text(249, legend_y + 9, "— Undefined"),
    ]
    return _frame(width, height, "".join(out), title, "profile-correlation")
