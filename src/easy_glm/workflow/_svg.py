"""Small SVG chart writer used by the HTML report (:mod:`easy_glm.workflow.report`).

The report has to be **one self-contained file** that a browser opens with no
network access and no console errors, and that is small enough to email. Plotly
would mean inlining its 4.8 MB script, so the report draws its own charts:
plain SVG elements, no JavaScript at all. Hover text is a native SVG
``<title>``, which every browser shows as a tooltip.

Everything here is a pure function returning an SVG string, so the charts are
testable without a browser. Colours match the workbench (``easy_glm.app.charts``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from html import escape
from typing import Any

BLUE = "#1f5f99"
ORANGE = "#e07b39"
GREY = "#c9cfd6"
GREEN = "#2e8b57"
RED = "#c0392b"
AXIS = "#5b6570"
GRID = "#e8ecef"

Number = float | int | None


# --------------------------------------------------------------------------
# numbers and ticks
# --------------------------------------------------------------------------
def fmt(value: Number, digits: int = 3) -> str:
    """Compact human number for a tick or a tooltip."""
    if value is None or not math.isfinite(float(value)):
        return "—"
    v = float(value)
    if v != 0 and abs(v) < 1e-3:
        return f"{v:.2e}"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:,.1f}"
    return f"{v:.{digits}f}"


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """``n``-ish round tick values covering ``[lo, hi]``."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return [0.0, 1.0]
    if hi <= lo:
        hi = lo + (abs(lo) or 1.0) * 0.1
    raw = (hi - lo) / max(n, 1)
    mag = 10.0 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    step = mag
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = m * mag
        if step >= raw:
            break
    start = math.floor(lo / step) * step
    ticks: list[float] = []
    v = start
    while v < hi + step * 0.5 and len(ticks) < 40:
        ticks.append(round(v, 12))
        v += step
    return ticks or [lo, hi]


def _finite(values: Sequence[Number]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


class _Axis:
    """Linear value → pixel mapping with round ticks."""

    def __init__(
        self,
        values: Sequence[Number],
        px_lo: float,
        px_hi: float,
        *,
        from_zero: bool = False,
        include: Sequence[float] = (),
    ) -> None:
        vals = _finite(values) + [float(v) for v in include]
        lo = min(vals) if vals else 0.0
        hi = max(vals) if vals else 1.0
        if from_zero:
            lo = min(lo, 0.0)
        if hi == lo:
            hi = lo + (abs(lo) or 1.0) * 0.1
        pad = (hi - lo) * 0.06
        self.lo = lo - pad
        if lo >= 0 > self.lo:  # never pad a non-negative quantity below zero
            self.lo = 0.0
        self.hi = hi + pad
        # round ticks inside the range; the range itself stays tight to the data
        self.ticks = [
            t for t in nice_ticks(self.lo, self.hi) if self.lo <= t <= self.hi
        ]
        self.px_lo, self.px_hi = px_lo, px_hi

    def px(self, value: Number) -> float | None:
        if value is None or not math.isfinite(float(value)):
            return None
        share = (float(value) - self.lo) / (self.hi - self.lo or 1.0)
        return self.px_lo + share * (self.px_hi - self.px_lo)


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------
def _dash(rest: list[Any]) -> str:
    """``stroke-dasharray`` for a line whose optional ``dashed`` flag was
    passed as a fourth (fifth for a curve) tuple element."""
    return ' stroke-dasharray="7 4"' if rest and rest[0] else ""


def _legend(entries: list[tuple[str, str, bool, str]], x: float, y: float) -> str:
    """``[(name, colour, is_line, dash)]`` as swatches on one row."""
    out: list[str] = []
    cur = x
    for name, colour, is_line, dash in entries:
        if is_line:
            out.append(
                f'<line x1="{cur:.1f}" y1="{y:.1f}" x2="{cur + 16:.1f}" '
                f'y2="{y:.1f}" stroke="{colour}" stroke-width="2.5"{dash}/>'
            )
        else:
            out.append(
                f'<rect x="{cur:.1f}" y="{y - 5:.1f}" width="16" height="10" '
                f'fill="{colour}"/>'
            )
        out.append(
            f'<text x="{cur + 21:.1f}" y="{y + 4:.1f}" class="lg">{escape(name)}</text>'
        )
        cur += 30 + 6.6 * len(name)
    return "".join(out)


def _y_axis(
    axis: _Axis, x: float, *, right: bool, title: str, grid_to: float | None = None
) -> str:
    out: list[str] = []
    anchor = "start" if right else "end"
    dx = 6 if right else -6
    for t in axis.ticks:
        y = axis.px(t)
        if y is None or y < axis.px_hi - 0.5 or y > axis.px_lo + 0.5:
            continue
        if grid_to is not None:
            out.append(
                f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{grid_to:.1f}" y2="{y:.1f}" '
                f'stroke="{GRID}" stroke-width="1"/>'
            )
        out.append(
            f'<text x="{x + dx:.1f}" y="{y + 4:.1f}" text-anchor="{anchor}" '
            f'class="ax">{escape(fmt(t))}</text>'
        )
    if title:
        rot = 90 if right else -90
        tx = x + (46 if right else -46)
        ty = (axis.px_lo + axis.px_hi) / 2
        out.append(
            f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" class="ax" '
            f'transform="rotate({rot} {tx:.1f} {ty:.1f})">{escape(title)}</text>'
        )
    return "".join(out)


def _x_labels(labels: list[str], centres: list[float], y: float) -> str:
    """Band labels under the axis: upright when they are all short (bin
    numbers), rotated when they are band names that would otherwise collide."""
    step = max(1, math.ceil(len(labels) / 26))
    upright = all(len(lab) <= 4 for lab in labels)
    out = []
    for i, (lab, cx) in enumerate(zip(labels, centres, strict=True)):
        if i % step:
            continue
        text = lab if len(lab) <= 22 else lab[:21] + "…"
        if upright:
            out.append(
                f'<text x="{cx:.1f}" y="{y + 2:.1f}" text-anchor="middle" '
                f'class="ax">{escape(text)}</text>'
            )
        else:
            out.append(
                f'<text x="{cx:.1f}" y="{y:.1f}" text-anchor="end" class="ax" '
                f'transform="rotate(-40 {cx:.1f} {y:.1f})">{escape(text)}</text>'
            )
    return "".join(out)


def _frame(width: int, height: int, body: str, title: str = "") -> str:
    """The ``<svg>`` wrapper. ``title`` becomes the chart's accessible name (a
    ``<title>`` first child, which is what ``role="img"`` is announced as), so
    a screen reader and a PDF bookmark say which variable and which chart this
    is instead of "image"."""
    label = f"<title>{escape(title)}</title>" if title else ""
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="100%" '
        f'height="{height}" role="img" xmlns="http://www.w3.org/2000/svg">'
        f"{label}{body}</svg>"
    )


# --------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------
def category_chart(
    labels: Sequence[str],
    *,
    bars: Sequence[Number] | None = None,
    bar_name: str = "exposure",
    lines: Sequence[tuple] = (),
    left_title: str = "exposure",
    right_title: str = "rate",
    hline: float | None = None,
    right_from_zero: bool = True,
    title: str = "",
    width: int = 900,
    height: int = 340,
) -> str:
    """Bars on a left axis and lines on a right axis over categorical bands.

    Used for A/E by variable (exposure bars, actual / expected lines), lift and
    double lift. ``lines`` is ``[(name, values, colour)]``, optionally with a
    fourth element ``dashed``: the challenger's line is dashed so it never
    hides the champion's where the two agree, exactly as the workbench draws
    it. Hovering a band shows every value of that band; ``title`` names the
    chart for a screen reader.
    """
    labels = [str(x) for x in labels]
    if not labels:  # nothing to draw: an empty frame beats a broken chart
        return _frame(width, height, "", title)
    n = len(labels)
    ml, mr, mt, mb = 66, 66, 28, 96
    x0, x1 = ml, width - mr
    y0, y1 = height - mb, mt
    bw = (x1 - x0) / n
    centres = [x0 + (i + 0.5) * bw for i in range(n)]

    drawn = [(name, vals, colour, _dash(rest)) for name, vals, colour, *rest in lines]
    left = _Axis(bars or [0.0], y0, y1, from_zero=True)
    right_values = [v for _n, vals, _c, _d in drawn for v in vals]
    right = _Axis(
        right_values or [0.0, 1.0],
        y0,
        y1,
        from_zero=right_from_zero,
        include=[hline] if hline is not None else [],
    )

    out = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>']
    if bars is None:
        # nothing to put on the left axis: the lines get it, and the width back
        out.append(_y_axis(right, x0, right=False, title=right_title, grid_to=x1))
    else:
        out.append(_y_axis(right, x1, right=True, title=right_title, grid_to=x0))
        out.append(_y_axis(left, x0, right=False, title=left_title))
        base = left.px(0.0) or y0
        for cx, v in zip(centres, bars, strict=False):
            y = left.px(v)
            if y is None:
                continue
            top, bot = min(y, base), max(y, base)
            out.append(
                f'<rect x="{cx - bw * 0.35:.1f}" y="{top:.1f}" '
                f'width="{bw * 0.7:.1f}" height="{max(bot - top, 0.5):.1f}" '
                f'fill="{GREY}" opacity="0.7"/>'
            )
    out.append(
        f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y0:.1f}" '
        f'stroke="{AXIS}" stroke-width="1"/>'
    )
    if hline is not None:
        hy = right.px(hline)
        if hy is not None:
            out.append(
                f'<line x1="{x0:.1f}" y1="{hy:.1f}" x2="{x1:.1f}" y2="{hy:.1f}" '
                f'stroke="{AXIS}" stroke-width="1" stroke-dasharray="4 3"/>'
            )
    for _name, values, colour, dash in drawn:
        pts = [
            (cx, right.px(v))
            for cx, v in zip(centres, values, strict=False)
            if right.px(v) is not None
        ]
        if not pts:
            continue
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        out.append(
            f'<polyline points="{path}" fill="none" stroke="{colour}" '
            f'stroke-width="2.5" stroke-linejoin="round"{dash}/>'
        )
        for x, y in pts:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{colour}"/>')
    # one transparent column per band carrying the hover text
    for i, cx in enumerate(centres):
        parts = [labels[i] if i < len(labels) else ""]
        if bars is not None and i < len(bars):
            parts.append(f"{bar_name}: {fmt(bars[i])}")
        for name, values, _c, _d in drawn:
            if i < len(values):
                parts.append(f"{name}: {fmt(values[i], 4)}")
        out.append(
            f'<rect x="{cx - bw / 2:.1f}" y="{y1:.1f}" width="{bw:.1f}" '
            f'height="{y0 - y1:.1f}" fill="transparent">'
            f"<title>{escape(chr(10).join(parts))}</title></rect>"
        )
    out.append(_x_labels(labels, centres, y0 + 14))
    entries: list[tuple[str, str, bool, str]] = []
    if bars is not None:
        entries.append((bar_name, GREY, False, ""))
    entries += [(name, colour, True, dash) for name, _v, colour, dash in drawn]
    out.append(_legend(entries, x0, 14))
    return _frame(width, height, "".join(out), title)


def curve_chart(
    series: Sequence[tuple],
    *,
    x_title: str = "value",
    y_title: str = "relativity",
    hline: float | None = 1.0,
    marks: Sequence[tuple[float, str]] = (),
    title: str = "",
    width: int = 900,
    height: int = 340,
) -> str:
    """Continuous curves on a numeric x axis (piecewise-linear relativities).

    ``series`` is ``[(name, xs, ys, colour)]``, optionally with a fifth element
    ``dashed``; ``marks`` draws labelled vertical lines (the clamp points, the
    base point); ``title`` names the chart for a screen reader.
    """
    ml, mr, mt, mb = 66, 30, 28, 56
    x0, x1 = ml, width - mr
    y0, y1 = height - mb, mt
    drawn = [
        (name, xs, ys, colour, _dash(rest)) for name, xs, ys, colour, *rest in series
    ]
    xs_all = [x for _n, xs, _y, _c, _d in drawn for x in xs]
    ys_all = [y for _n, _x, ys, _c, _d in drawn for y in ys]
    xa = _Axis(xs_all or [0.0, 1.0], x0, x1, include=[m[0] for m in marks])
    ya = _Axis(
        ys_all or [0.0, 1.0],
        y0,
        y1,
        from_zero=True,
        include=[hline] if hline is not None else [],
    )
    out = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>']
    out.append(_y_axis(ya, x0, right=False, title=y_title, grid_to=x1))
    out.append(
        f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y0:.1f}" '
        f'stroke="{AXIS}" stroke-width="1"/>'
    )
    for t in xa.ticks:
        x = xa.px(t)
        if x is None or x < x0 - 0.5 or x > x1 + 0.5:
            continue
        out.append(
            f'<text x="{x:.1f}" y="{y0 + 16:.1f}" text-anchor="middle" '
            f'class="ax">{escape(fmt(t))}</text>'
        )
    out.append(
        f'<text x="{(x0 + x1) / 2:.1f}" y="{height - 8:.1f}" text-anchor="middle" '
        f'class="ax">{escape(x_title)}</text>'
    )
    if hline is not None:
        hy = ya.px(hline)
        if hy is not None:
            out.append(
                f'<line x1="{x0:.1f}" y1="{hy:.1f}" x2="{x1:.1f}" y2="{hy:.1f}" '
                f'stroke="{AXIS}" stroke-width="1" stroke-dasharray="4 3"/>'
            )
    for value, label in marks:
        x = xa.px(value)
        if x is None:
            continue
        out.append(
            f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y0:.1f}" '
            f'stroke="{GREEN}" stroke-width="1" stroke-dasharray="5 4"/>'
            f'<text x="{x + 4:.1f}" y="{y1 + 12:.1f}" class="ax">{escape(label)}</text>'
        )
    for name, xs, ys, colour, dash in drawn:
        pts = [
            (xa.px(x), ya.px(y))
            for x, y in zip(xs, ys, strict=False)
            if xa.px(x) is not None and ya.px(y) is not None
        ]
        if not pts:
            continue
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        out.append(
            f'<polyline points="{path}" fill="none" stroke="{colour}" '
            f'stroke-width="2.5" stroke-linejoin="round"{dash}>'
            f"<title>{escape(name)}</title></polyline>"
        )
    out.append(
        _legend(
            [(name, colour, True, dash) for name, _x, _y, colour, dash in drawn],
            x0,
            14,
        )
    )
    return _frame(width, height, "".join(out), title)


def _ratio_colour(value: float | None) -> str:
    """Red above 1, blue below, on a log scale saturating at ×2 / ÷2."""
    if value is None or value <= 0 or not math.isfinite(value):
        return "#dfe4e8"  # no value: a grey that cannot be read as "1.00"
    t = max(-1.0, min(1.0, math.log(value) / math.log(2.0)))
    if t >= 0:  # white -> red
        r, g, b = 255, int(255 - 137 * t), int(255 - 175 * t)
    else:  # white -> blue
        s = -t
        r, g, b = int(255 - 224 * s), int(255 - 160 * s), int(255 - 102 * s)
    return f"rgb({r},{g},{b})"


def heatmap(
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    values: Sequence[Sequence[float | None]],
    *,
    row_name: str = "",
    col_name: str = "",
    hover: dict[str, Sequence[Sequence[float | None]]] | None = None,
    title: str = "",
    width: int = 900,
) -> str:
    """Matrix of multiplicative values (relativities, A/E) centred on 1.00.

    Blank cells are the ones with no value. ``hover`` adds named matrices to
    each cell's tooltip (exposure, actual, expected).
    """
    rows = [str(r) for r in row_labels]
    cols = [str(c) for c in col_labels]
    ml, mr, mt, mb = 150, 30, 74, 40
    cell_w = max(28.0, min(90.0, (width - ml - mr) / max(len(cols), 1)))
    cell_h = 26.0
    height = int(mt + mb + cell_h * max(len(rows), 1))
    out = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>']
    for j, c in enumerate(cols):
        cx = ml + (j + 0.5) * cell_w
        text = c if len(c) <= 18 else c[:17] + "…"
        out.append(
            f'<text x="{cx:.1f}" y="{mt - 8:.1f}" text-anchor="start" class="ax" '
            f'transform="rotate(-45 {cx:.1f} {mt - 8:.1f})">{escape(text)}</text>'
        )
    for i, r in enumerate(rows):
        cy = mt + (i + 0.5) * cell_h
        text = r if len(r) <= 22 else r[:21] + "…"
        out.append(
            f'<text x="{ml - 8:.1f}" y="{cy + 4:.1f}" text-anchor="end" '
            f'class="ax">{escape(text)}</text>'
        )
        for j in range(len(cols)):
            v = values[i][j] if i < len(values) and j < len(values[i]) else None
            x = ml + j * cell_w
            y = mt + i * cell_h
            parts = [f"{row_name or 'row'}: {r}", f"{col_name or 'column'}: {cols[j]}"]
            parts.append(f"value: {fmt(v, 4)}")
            for name, matrix in (hover or {}).items():
                try:
                    parts.append(f"{name}: {fmt(matrix[i][j])}")
                except (IndexError, TypeError):  # pragma: no cover - ragged input
                    pass
            out.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w - 1:.1f}" '
                f'height="{cell_h - 1:.1f}" fill="{_ratio_colour(v)}" '
                f'stroke="#ffffff" stroke-width="1">'
                f"<title>{escape(chr(10).join(parts))}</title></rect>"
            )
    if row_name:
        out.append(f'<text x="6" y="{mt - 8:.1f}" class="ax">{escape(row_name)}</text>')
    if col_name:
        out.append(
            f'<text x="{ml:.1f}" y="{height - 12:.1f}" class="ax">'
            f"{escape(col_name)}</text>"
        )
    # colour key
    key_x = width - mr - 190
    for k, ratio in enumerate((0.5, 0.7, 0.85, 1.0, 1.2, 1.4, 2.0)):
        out.append(
            f'<rect x="{key_x + k * 24:.1f}" y="{height - 26:.1f}" width="23" '
            f'height="10" fill="{_ratio_colour(ratio)}" stroke="#ffffff"/>'
            f'<text x="{key_x + k * 24 + 11:.1f}" y="{height - 4:.1f}" '
            f'text-anchor="middle" class="lg">{ratio:g}</text>'
        )
    return _frame(width, height, "".join(out), title)
