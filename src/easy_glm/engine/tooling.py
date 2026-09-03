"""Relativity tooling (D5): smooth, cap / floor and round one rate table.

Pure functions on a single :class:`~easy_glm.engine.models.VariableConfig`. Each
returns a :class:`ToolResult` holding **one relativity per row of the table, in
table order** — nothing is mutated, so a page can draw the result as a preview
and only then turn it into adjustments.

Three rules hold for every tool:

* **The null / Other row is never touched.** It is not part of the curve (a
  numeric table's missing-value row) or of any order (a categorical's catch-all
  bucket), so smoothing it would mix it with real bands and capping it would
  quietly change how unknown risks are rated. It keeps its value and its own
  editor row.
* **Smoothing preserves the exposure-weighted mean of the *log* relativities**
  (§R6 of the 0.4 plan). The base rate is not refitted when a table is edited,
  so a smoothing that moved that mean would move the overall premium level;
  keeping the mean of the *logs* is what keeps the level, because relativities
  multiply. The moving average is re-centred to achieve it; the weighted
  isotonic fit preserves it on its own (each pooled block is replaced by its
  weighted mean).
* **Cap / floor and round are idempotent**: applying either twice changes
  nothing the second time. Neither is re-centred — a cap that was then shifted
  back up would not be a cap.

The weights are the **training exposure per band** carried by the table rows
(``FromToRow.exposure`` / ``BandRow.exposure``, filled in by ``to_rate_model``
from ``GLMFit.row_exposure``). A table with no exposure recorded (hand-built, or
read back from a file written before 0.4) falls back to equal weights and says
so in :attr:`ToolResult.uniform_weights`.

**What a "band" is per table type**

* ``numeric`` (step) — one group per bin, in ascending order of the lower edge.
* ``categorical`` — one group per level, in the table's own order. That order is
  the encoder's (most exposed level first), which is *not* an order of the risk,
  so the two smoothers refuse a categorical unless the caller passes
  ``ordered=True`` to say the levels do read in order (e.g. "small / medium /
  large" after a recode).
* ``linear`` — one group per **node** of the curve, in ascending order. A node is
  a point the curve passes through: the value at the lower clamp (shared by the
  ``(None, lo)`` row and the first sloped band, which move together), the value
  at each interior knot, and the value at the upper clamp. Tools set the node
  values and the slopes are re-derived from them, so the curve stays continuous.
* ``interaction`` — refused: a cell adjustment has no neighbours along one axis,
  and the editor's grid is where cells are changed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .models import BandRow, VariableConfig, level_label
from .rate_model import derive_slopes

#: default window of the moving average, in bands (must be odd, at least 3)
DEFAULT_WINDOW = 3

Direction = Literal["increasing", "decreasing"]


class ToolingError(ValueError):
    """A tool that refuses to run, with a message a user can act on."""


@dataclass
class ToolResult:
    """The relativities a tool would set, before anything is applied.

    ``values`` has one entry per row of the table, in table order: the rows the
    tool did not touch (the null / Other row, and anything a tool leaves alone)
    keep the value they have now, so ``values`` can be handed straight to the
    editor's row-edit rule.
    """

    variable: str
    tool: str
    #: new relativity per table row, in table order
    values: list[float]
    #: current relativity per table row, in table order
    before: list[float]
    #: exposure weight per *group* (see the module docstring), in curve order
    weights: list[float]
    #: exposure-weighted mean of the log relativities, before and after
    log_mean_before: float
    log_mean_after: float
    #: True when the table carries no exposure and equal weights were used
    uniform_weights: bool
    #: plain-language description of what the tool did
    note: str
    #: labels of the rows whose relativity changed
    changed_labels: list[str] = field(default_factory=list)

    @property
    def changed(self) -> int:
        return len(self.changed_labels)

    @property
    def level_shift(self) -> float:
        """``exp(mean of logs after − before) − 1``: the change in the overall
        level this result would make. 0.0 for a smoothing (that is the point)."""
        return float(np.exp(self.log_mean_after - self.log_mean_before) - 1.0)


# --------------------------------------------------------------------------
# the rows a tool may touch
# --------------------------------------------------------------------------
def _check_main_effect(cfg: VariableConfig, variable: str) -> None:
    if cfg.type == "interaction":
        raise ToolingError(
            f"{variable!r} is an interaction: its cells have no neighbouring band "
            "to be smoothed, capped or rounded against. Edit the cells in the grid."
        )
    if cfg.type not in ("numeric", "categorical", "linear"):
        raise ToolingError(
            f"{variable!r} has table type {cfg.type!r}, which has no tools"
        )


def _is_null_row(row: Any) -> bool:
    return row.from_ is None and row.to_ is None


def groups(cfg: VariableConfig) -> list[list[int]]:
    """Table-row indices that move together, in curve / level order.

    One group per band, level or curve node; the null / Other row is in no
    group, which is how every tool leaves it alone. See the module docstring for
    what a group is per table type.
    """
    body = [i for i, r in enumerate(cfg.table) if not _is_null_row(cfg.table[i])]
    if cfg.type == "categorical":
        return [[i] for i in body]
    body.sort(
        key=lambda i: (
            cfg.table[i].from_ is not None,
            float(cfg.table[i].from_) if cfg.table[i].from_ is not None else 0.0,
        )
    )
    if cfg.type == "numeric":
        return [[i] for i in body]
    # linear: the (None, lo) row and the first sloped band are one node
    if len(body) < 3:  # pragma: no cover - _validate_linear_rows refuses this
        raise ToolingError("A piecewise-linear table needs at least three rows")
    return [[body[0], body[1]], *([i] for i in body[2:])]


def group_weights(cfg: VariableConfig) -> tuple[np.ndarray, bool]:
    """``(weight per group, fell back to equal weights)``.

    A group's weight is the training exposure of its rows. When no row of the
    table carries exposure — a hand-built table, or one read back from a file
    written before exposure was recorded — every group weighs 1 instead, and the
    flag says so, so a page can tell the user the level check is unweighted.
    """
    w = np.array(
        [sum(float(cfg.table[i].exposure) for i in g) for g in groups(cfg)],
        dtype=float,
    )
    if not np.all(np.isfinite(w)) or w.sum() <= 0:
        return np.ones(len(w)), True
    return w, False


def group_values(cfg: VariableConfig) -> np.ndarray:
    """Current relativity per group (the first row of each group holds it)."""
    return np.array([float(cfg.table[g[0]].relativity) for g in groups(cfg)], float)


def weighted_log_mean(cfg: VariableConfig, values: np.ndarray | None = None) -> float:
    """Exposure-weighted mean of the **log** relativities over the groups.

    The number a smoothing must leave alone: because relativities multiply, it
    is the average log premium effect of the factor, and moving it moves every
    premium the factor touches (the base rate is not refitted after an edit).
    ``values`` defaults to the table's current relativities.
    """
    v = group_values(cfg) if values is None else np.asarray(values, dtype=float)
    w, _ = group_weights(cfg)
    if np.any(v <= 0):
        return float("nan")
    return float(np.sum(w * np.log(v)) / np.sum(w))


# --------------------------------------------------------------------------
# result assembly
# --------------------------------------------------------------------------
def _result(
    cfg: VariableConfig,
    variable: str,
    tool: str,
    new_values: np.ndarray,
    note: str,
) -> ToolResult:
    gs = groups(cfg)
    w, uniform = group_weights(cfg)
    before_groups = group_values(cfg)
    if not np.all(np.isfinite(new_values)) or np.any(new_values <= 0):
        bad = [
            level_label(cfg.table[g[0]], cfg.other_label)
            for g, v in zip(gs, new_values, strict=True)
            if not np.isfinite(v) or v <= 0
        ]
        raise ToolingError(
            f"{tool} on {variable!r} would set a relativity of zero or less on "
            f"{', '.join(bad[:4])}"
            + (" and others" if len(bad) > 4 else "")
            + "; a relativity has to be above 0, so nothing was changed."
        )
    values = [float(r.relativity) for r in cfg.table]
    before = list(values)
    changed: list[str] = []
    for g, v in zip(gs, new_values, strict=True):
        for i in g:
            values[i] = float(v)
        if abs(float(v) - float(before[g[0]])) > 1e-12:
            changed.append(level_label(cfg.table[g[0]], cfg.other_label))
    return ToolResult(
        variable=variable,
        tool=tool,
        values=values,
        before=before,
        weights=[float(x) for x in w],
        log_mean_before=weighted_log_mean(cfg, before_groups),
        log_mean_after=weighted_log_mean(cfg, new_values),
        uniform_weights=uniform,
        note=note,
        changed_labels=changed,
    )


def _recentre(new_log: np.ndarray, old_log: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Shift ``new_log`` so its weighted mean equals ``old_log``'s — the level
    the table had before the smoothing."""
    shift = float(np.sum(w * old_log) - np.sum(w * new_log)) / float(np.sum(w))
    return new_log + shift


# --------------------------------------------------------------------------
# smoothing
# --------------------------------------------------------------------------
def _smoothable(cfg: VariableConfig, variable: str, ordered: bool) -> None:
    _check_main_effect(cfg, variable)
    if cfg.type == "categorical" and not ordered:
        raise ToolingError(
            f"{variable!r} is a categorical factor. Its rows are in the order the "
            "encoder built them (most exposed level first), which is not an order "
            "of the risk, so smoothing would average unrelated levels together. "
            "If the levels really do read in order (a banded or graded factor), "
            "confirm that and the smoother will use the order shown in the table."
        )
    if len(groups(cfg)) < 2:
        raise ToolingError(
            f"{variable!r} has only one band, so there is nothing to smooth."
        )


def smooth_moving_average(
    cfg: VariableConfig,
    variable: str,
    *,
    window: int = DEFAULT_WINDOW,
    ordered: bool = False,
) -> ToolResult:
    """Moving average of the **log** relativities over ``window`` bands.

    Each band is replaced by the exposure-weighted average of the logs of itself
    and its neighbours (the window is centred and shrinks at the two ends, so no
    band is dropped), and the whole curve is then shifted so the
    exposure-weighted mean of the logs is exactly what it was: the smoothing
    changes the *shape* of the factor, never its level. Weighting by exposure
    inside the window is what stops a thin band with a wild relativity from
    dragging its well-populated neighbours.
    """
    _smoothable(cfg, variable, ordered)
    window = int(window)
    if window < 3 or window % 2 == 0:
        raise ToolingError(
            f"The moving average needs an odd window of 3 or more bands (got "
            f"{window}); an even window would sit between two bands instead of on one."
        )
    v = group_values(cfg)
    if np.any(v <= 0):
        raise ToolingError(
            f"{variable!r} has a relativity of zero or less, which has no logarithm; "
            "fix that row before smoothing."
        )
    w, uniform = group_weights(cfg)
    y = np.log(v)
    half = window // 2
    out = np.empty_like(y)
    for i in range(len(y)):
        lo, hi = max(0, i - half), min(len(y), i + half + 1)
        ww = w[lo:hi]
        out[i] = (
            float(np.sum(ww * y[lo:hi]) / np.sum(ww))
            if np.sum(ww) > 0
            else float(np.mean(y[lo:hi]))
        )
    out = _recentre(out, y, w)
    note = (
        f"Moving average over {window} bands in log space, "
        f"{'unweighted (no exposure recorded)' if uniform else 'weighted by exposure'}, "
        "re-centred so the exposure-weighted mean of the log relativities is unchanged."
    )
    return _result(cfg, variable, "Smooth (moving average)", np.exp(out), note)


def smooth_isotonic(
    cfg: VariableConfig,
    variable: str,
    *,
    direction: Direction = "increasing",
    ordered: bool = False,
) -> ToolResult:
    """Isotonic (monotone) fit of the log relativities by pool-adjacent-violators.

    The result is the closest exposure-weighted set of relativities that never
    turns back: each run of bands that breaks the direction is pooled into one
    value, their exposure-weighted average. Pooling replaces a block by its own
    weighted mean, so the exposure-weighted mean of the log relativities is
    preserved exactly — no re-centring is needed.
    """
    _smoothable(cfg, variable, ordered)
    if direction not in ("increasing", "decreasing"):
        raise ToolingError(
            f"direction must be 'increasing' or 'decreasing', got {direction!r}"
        )
    v = group_values(cfg)
    if np.any(v <= 0):
        raise ToolingError(
            f"{variable!r} has a relativity of zero or less, which has no logarithm; "
            "fix that row before smoothing."
        )
    w, uniform = group_weights(cfg)
    y = np.log(v)
    fitted = _pava(
        y if direction == "increasing" else y[::-1],
        w if direction == "increasing" else w[::-1],
    )
    if direction == "decreasing":
        fitted = fitted[::-1]
    fitted = _recentre(fitted, y, w)  # a no-op up to rounding; see the docstring
    pooled = int(np.sum(np.abs(np.diff(fitted)) <= 1e-12))
    note = (
        f"Isotonic ({direction}) fit of the log relativities, "
        f"{'unweighted (no exposure recorded)' if uniform else 'weighted by exposure'}: "
        f"{pooled} pair(s) of neighbouring bands pooled to the same value. The "
        "exposure-weighted mean of the log relativities is unchanged."
    )
    return _result(
        cfg, variable, f"Smooth (isotonic, {direction})", np.exp(fitted), note
    )


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators: the closest non-decreasing fit to ``y``.

    Blocks with no exposure at all (a band no policy reached) are pooled by
    their plain mean, so a zero weight never divides; the merged value always
    lies between the two it replaces, which is what keeps the result monotone.
    """
    blocks: list[dict[str, float]] = []
    for value, weight in zip(y, w, strict=True):
        blocks.append(
            {
                "w": float(weight),
                "wy": float(weight) * float(value),
                "n": 1.0,
                "y": float(value),
            }
        )
        while len(blocks) > 1 and _block_value(blocks[-2]) > _block_value(blocks[-1]):
            b = blocks.pop()
            a = blocks[-1]
            for key in ("w", "wy", "n", "y"):
                a[key] += b[key]
    out: list[float] = []
    for b in blocks:
        out.extend([_block_value(b)] * int(b["n"]))
    return np.array(out, dtype=float)


def _block_value(block: dict[str, float]) -> float:
    return block["wy"] / block["w"] if block["w"] > 0 else block["y"] / block["n"]


# --------------------------------------------------------------------------
# cap / floor and round
# --------------------------------------------------------------------------
def cap_floor(
    cfg: VariableConfig,
    variable: str,
    *,
    floor: float | None = None,
    cap: float | None = None,
) -> ToolResult:
    """Clamp every band's relativity to ``[floor, cap]`` (either may be omitted).

    The level is deliberately *not* restored afterwards: capping a tail and then
    shifting the whole factor back up would put the capped bands above the cap
    again. Clamping twice changes nothing the second time.
    """
    _check_main_effect(cfg, variable)
    if floor is None and cap is None:
        raise ToolingError("Give a floor, a cap, or both.")
    if floor is not None and not floor > 0:
        raise ToolingError(f"The floor must be above 0 (got {floor:g}).")
    if cap is not None and not cap > 0:
        raise ToolingError(f"The cap must be above 0 (got {cap:g}).")
    if floor is not None and cap is not None and floor > cap:
        raise ToolingError(
            f"The floor ({floor:g}) is above the cap ({cap:g}); nothing would be left."
        )
    v = group_values(cfg)
    out = np.clip(
        v, floor if floor is not None else -np.inf, cap if cap is not None else np.inf
    )
    bits = []
    if floor is not None:
        bits.append(f"floor {floor:g}")
    if cap is not None:
        bits.append(f"cap {cap:g}")
    note = (
        f"Relativities clamped to {' and '.join(bits)}. The overall level moves with "
        "the clamped bands (a cap that was shifted back up would not be a cap)."
    )
    return _result(cfg, variable, "Cap / floor", out, note)


def round_relativities(
    cfg: VariableConfig,
    variable: str,
    *,
    decimals: int | None = None,
    step: float | None = None,
) -> ToolResult:
    """Round every band's relativity to ``decimals`` places or to a ``step``
    (e.g. 0.05, the way published rate tables are printed).

    Exactly one of the two is given. Rounding twice changes nothing the second
    time, and the level is not restored afterwards — a rounded table whose bands
    were then shifted would no longer be round.
    """
    _check_main_effect(cfg, variable)
    if (decimals is None) == (step is None):
        raise ToolingError("Round to a number of decimals or to a step, not both.")
    v = group_values(cfg)
    if decimals is not None:
        if not 0 <= int(decimals) <= 10:
            raise ToolingError(f"Round to between 0 and 10 decimals (got {decimals}).")
        out = np.round(v, int(decimals))
        note = f"Relativities rounded to {int(decimals)} decimal place(s)."
    else:
        if not step > 0:  # type: ignore[operator]
            raise ToolingError(f"The rounding step must be above 0 (got {step}).")
        # the second rounding only tidies the float (0.05 x 28 = 1.4000000000000001)
        out = np.round(np.round(v / float(step)) * float(step), 12)
        note = f"Relativities rounded to the nearest {float(step):g}."
    note += (
        " The overall level moves by the rounding (restoring it would un-round the "
        "table)."
    )
    return _result(cfg, variable, "Round", out, note)


# --------------------------------------------------------------------------
# applying a result
# --------------------------------------------------------------------------
def apply_values(
    cfg: VariableConfig, values: list[float] | np.ndarray
) -> VariableConfig:
    """A **copy** of ``cfg`` with ``values`` as its relativities (one per table
    row, in table order); a piecewise-linear table's slopes are re-derived from
    the new node values, so the previewed curve is the one an edit would give.

    Used to draw a preview; the table the model scores from is only ever changed
    through the project's adjustments and ``rebuild_rate_model``.
    """
    out = copy.deepcopy(cfg)
    if len(values) != len(out.table):
        raise ToolingError(
            f"Expected one relativity per table row ({len(out.table)}), got {len(values)}"
        )
    for row, value in zip(out.table, values, strict=True):
        row.relativity = float(value)
    if out.type == "linear":
        bands: list[BandRow] = sorted(
            (r for r in out.table if not _is_null_row(r)),
            key=lambda r: (r.from_ is not None, r.from_ or 0.0),
        )
        derive_slopes(bands)
    # the precomputed lookups belong to the old values
    out.breakpoints = None
    out.relativities = None
    out.cat_map = None
    out.level_index = None
    out.slopes = None
    out.starts = None
    out.null_relativity = None
    out.fallback = 1.0
    from .rate_model import RateModel

    RateModel._precompute_variables({"preview": out})
    return out
