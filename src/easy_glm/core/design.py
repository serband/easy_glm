"""Design specification: how raw columns become GLM features.

A :class:`DesignSpec` is the single source of truth for a model's features.
It maps each predictor to an :class:`Encoder` that knows how to turn a raw
polars column into design-matrix columns *and* what every column means
(variable, kind, knot or level). Because that metadata is structured, rate
tables can be read straight off the fitted coefficients (see
:mod:`easy_glm.core.tables`) and the spec round-trips through JSON.

Four encoders exist:

* :class:`StepEncoder` -- AGLM-style "O-dummies": one column ``1{x >= k}`` per
  knot. With an L1 penalty each coefficient is the *increment* of the effect
  at that knot, so unchanged increments are shrunk to exactly zero and the
  fitted curve is a piecewise-constant step function with data-driven bands.
  Nulls get all-zero step columns (they sit in the lowest bin) plus, by
  default, a dedicated ``is null`` column so they can carry their own effect.
* :class:`CategoricalEncoder` -- one column per kept level except the
  reference (the most frequent level), plus an ``Other`` column for lumped,
  unseen and null values.
* :class:`LinearEncoder` -- AGLM-style "L-dummies": hinge columns
  ``max(x - k, 0)`` at the lower clamp and every knot, so the fitted effect is a
  continuous piecewise-linear curve (on the log scale) with data-driven bends,
  exactly flat outside the training range.
* :class:`InteractionEncoder` -- a two-way interaction ``A × B`` on top of the
  two mains: one 0/1 column per *kept cell* of the parents' rate-table rows,
  so each coefficient is one cell's multiplicative adjustment.

Every encoder also knows its **rate-table rows** (:meth:`Encoder.rows`) and can
map a raw value to its row index (:meth:`Encoder.row_index`); that shared rule
is what keeps the rate tables, the interaction cells and the scorer aligned.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import polars as pl

from easy_glm.engine.models import INTERACTION_SEP, NULL_LABEL

NUMERIC_DTYPES = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
)

FeatureKind = Literal["step", "hinge", "null", "level", "other", "cell"]


@dataclass(frozen=True)
class Feature:
    """One design-matrix column and what it means."""

    name: str
    variable: str
    kind: FeatureKind
    knot: float | None = None
    level: str | None = None
    #: interaction columns only: ``(row_a, row_b)`` rate-table row indices
    cell: tuple[int, int] | None = None


def format_knot(value: float) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def row_label(row: tuple[Any, Any]) -> str:
    """Human-readable label of a rate-table row ``(from, to)``."""
    lo, hi = row
    if lo is None and hi is None:
        return NULL_LABEL
    if lo is None:
        return f"< {hi}"
    if hi is None:
        return f"≥ {lo}"
    if lo == hi:
        return str(lo)
    return f"[{lo}, {hi})"


class Encoder(ABC):
    """Turns one raw column (or two, for interactions) into design columns."""

    kind: ClassVar[str]
    variable: str

    @abstractmethod
    def features(self) -> list[Feature]: ...

    @abstractmethod
    def transform(self, series: pl.Series) -> np.ndarray:
        """Return an ``(n_rows, n_features)`` float64 array."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    # -- rate-table rows ----------------------------------------------------
    @abstractmethod
    def rows(self) -> list[tuple[Any, Any]]:
        """``(from, to)`` per rate-table row, in table order; the last row is
        the null (numeric) / Other (categorical) row."""

    @abstractmethod
    def row_index(self, series: pl.Series) -> np.ndarray:
        """Rate-table row index (0-based, into :meth:`rows`) of every value."""

    @property
    def n_rows(self) -> int:
        return len(self.rows())

    @property
    def n_features(self) -> int:
        return len(self.features())

    @property
    def required_columns(self) -> list[str]:
        return [self.variable]

    def transform_frame(self, data: pl.DataFrame) -> np.ndarray:
        """Design columns from a frame (default: :meth:`transform` on the column)."""
        return self.transform(data[self.variable])


@dataclass
class StepEncoder(Encoder):
    """Step-function (O-dummy) encoding of a numeric variable.

    Parameters
    ----------
    variable : str
        Column name.
    knots : list[float]
        Sorted, distinct thresholds. Column ``j`` is ``1{x >= knots[j]}``.
        The bins are ``(-inf, k0), [k0, k1), ..., [k_last, inf)``.
    null_indicator : bool
        Add a ``1{x is null}`` column so nulls get their own effect on top
        of the lowest bin.
    """

    kind: ClassVar[str] = "step"

    variable: str
    knots: list[float]
    null_indicator: bool = True

    def __post_init__(self) -> None:
        knots = sorted({float(k) for k in self.knots})
        if not knots:
            raise ValueError(f"StepEncoder({self.variable!r}) needs at least one knot")
        if any(not np.isfinite(k) for k in knots):
            raise ValueError(f"StepEncoder({self.variable!r}) knots must be finite")
        self.knots = knots

    def features(self) -> list[Feature]:
        out = [
            Feature(f"{self.variable}>={format_knot(k)}", self.variable, "step", knot=k)
            for k in self.knots
        ]
        if self.null_indicator:
            out.append(Feature(f"{self.variable} is null", self.variable, "null"))
        return out

    def bins(self) -> list[tuple[float | None, float | None]]:
        """``(from, to)`` per bin, ``None`` = open end; ``len(knots) + 1`` bins."""
        edges: list[float | None] = [None, *self.knots, None]
        return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    def rows(self) -> list[tuple[Any, Any]]:
        return [*self.bins(), (None, None)]

    def band_edges(self) -> list[float]:
        """Edges that band a raw value into this encoder's rate-table rows (for
        step encoders: the knots) — pass to ``ae_by_variable(knots=...)`` to get
        labels identical to the table's."""
        return list(self.knots)

    def row_index(self, series: pl.Series) -> np.ndarray:
        x = series.cast(pl.Float64).to_numpy()
        idx = np.searchsorted(np.asarray(self.knots), x, side="right")
        return np.where(np.isnan(x), len(self.knots) + 1, idx).astype(np.int64)

    def transform(self, series: pl.Series) -> np.ndarray:
        x = series.cast(pl.Float64).to_numpy()
        knots = np.asarray(self.knots)
        # NaN >= k is False, so nulls land in the lowest bin.
        with np.errstate(invalid="ignore"):
            cols = (x[:, None] >= knots[None, :]).astype(np.float64)
        if self.null_indicator:
            cols = np.hstack([cols, np.isnan(x)[:, None].astype(np.float64)])
        return cols

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "variable": self.variable,
            "knots": list(self.knots),
            "null_indicator": self.null_indicator,
        }


@dataclass
class CategoricalEncoder(Encoder):
    """One-hot encoding with a reference level and an ``Other`` bucket.

    Parameters
    ----------
    variable : str
        Column name. Values are compared as strings.
    levels : list[str]
        Kept levels. ``levels[0]`` is the reference and gets no column
        (its effect is the intercept); use the most frequent level.
    other_label : str
        Label of the catch-all bucket for lumped, unseen and null values.
    """

    kind: ClassVar[str] = "categorical"

    variable: str
    levels: list[str]
    other_label: str = "Other"

    def __post_init__(self) -> None:
        levels = [str(lvl) for lvl in self.levels]
        if not levels:
            raise ValueError(
                f"CategoricalEncoder({self.variable!r}) needs at least one level"
            )
        if len(set(levels)) != len(levels):
            raise ValueError(f"CategoricalEncoder({self.variable!r}) levels not unique")
        if self.other_label in levels:
            raise ValueError(
                f"CategoricalEncoder({self.variable!r}): other_label "
                f"{self.other_label!r} clashes with a level"
            )
        self.levels = levels

    @property
    def reference(self) -> str:
        return self.levels[0]

    def features(self) -> list[Feature]:
        out = [
            Feature(f"{self.variable}={lvl}", self.variable, "level", level=lvl)
            for lvl in self.levels[1:]
        ]
        out.append(
            Feature(
                f"{self.variable}={self.other_label}",
                self.variable,
                "other",
                level=self.other_label,
            )
        )
        return out

    def rows(self) -> list[tuple[Any, Any]]:
        return [*((lvl, lvl) for lvl in self.levels), (None, None)]

    def row_index(self, series: pl.Series) -> np.ndarray:
        vals = series.cast(pl.Utf8)
        other = len(self.levels)
        idx = vals.replace_strict(
            self.levels, list(range(other)), default=other, return_dtype=pl.Int64
        )
        return idx.fill_null(other).to_numpy().astype(np.int64)

    def transform(self, series: pl.Series) -> np.ndarray:
        vals = series.cast(pl.Utf8)
        cols = [(vals == lvl).fill_null(False).to_numpy() for lvl in self.levels[1:]]
        other = (~vals.is_in(self.levels)).fill_null(True) | vals.is_null()
        cols.append(other.to_numpy())
        return np.column_stack(cols).astype(np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "variable": self.variable,
            "levels": list(self.levels),
            "other_label": self.other_label,
        }


def _hinge_name(variable: str, knot: float) -> str:
    k = format_knot(knot)
    return (
        f"max({variable}+{k[1:]},0)" if k.startswith("-") else f"max({variable}-{k},0)"
    )


@dataclass
class LinearEncoder(Encoder):
    """Piecewise-linear (L-dummy) encoding of a numeric variable.

    ``x`` is first clipped to ``clamp = (lo, hi)`` and then expanded into hinge
    columns ``max(x_clipped - k, 0)`` at ``lo`` **and** at every interior knot.
    The fitted effect is therefore a continuous piecewise-linear curve on the
    linear-predictor scale whose slope can change at each knot, is fitted in the
    first band too (the hinge at ``lo``), and is exactly flat outside
    ``[lo, hi]`` — beyond the training range the relativity stays at its value at
    the nearer clamp. With an L1 penalty each coefficient is a *change of slope*,
    so the lasso keeps few bends. Nulls get all-zero hinge columns (the value at
    ``lo``) plus, by default, an ``is null`` column so they can carry their own
    effect.

    Rate-table rows (:meth:`rows`): ``(None, lo)`` flat, one band per pair of
    consecutive edges ``[lo, k1, ..., km, hi]`` (log-linear inside), ``(hi, None)``
    flat, then the null row.

    Parameters
    ----------
    variable : str
        Column name.
    knots : list[float]
        Interior knots, strictly inside ``(lo, hi)``; the slope may change there.
    clamp : (lo, hi)
        Range the raw value is clipped to before the hinges; the curve is flat
        outside it. ``DesignSpec.from_data`` uses the training minimum/maximum.
    null_indicator : bool
        Add a ``1{x is null}`` column.
    """

    kind: ClassVar[str] = "linear"

    variable: str
    knots: list[float]
    clamp: tuple[float, float]
    null_indicator: bool = True

    def __post_init__(self) -> None:
        if len(self.clamp) != 2:
            raise ValueError(f"LinearEncoder({self.variable!r}) clamp must be (lo, hi)")
        lo, hi = (float(v) for v in self.clamp)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(f"LinearEncoder({self.variable!r}) clamp must be finite")
        if not lo < hi:
            raise ValueError(
                f"LinearEncoder({self.variable!r}) clamp needs lo < hi, got ({lo}, {hi})"
            )
        knots = sorted({float(k) for k in self.knots})
        if any(not np.isfinite(k) for k in knots):
            raise ValueError(f"LinearEncoder({self.variable!r}) knots must be finite")
        bad = [k for k in knots if not lo < k < hi]
        if bad:
            raise ValueError(
                f"LinearEncoder({self.variable!r}) knots {bad} must lie strictly "
                f"inside the clamp range ({lo}, {hi})"
            )
        self.knots = knots
        self.clamp = (lo, hi)

    @property
    def lo(self) -> float:
        return self.clamp[0]

    @property
    def hi(self) -> float:
        return self.clamp[1]

    @property
    def hinges(self) -> list[float]:
        """Knots of the hinge columns: ``lo`` then the interior knots."""
        return [self.lo, *self.knots]

    def band_edges(self) -> list[float]:
        """``[lo, k1, ..., km, hi]`` — the edges of the sloped bands. Passing them
        to ``ae_by_variable(knots=...)`` gives labels identical to the table's."""
        return [self.lo, *self.knots, self.hi]

    def features(self) -> list[Feature]:
        out = [
            Feature(_hinge_name(self.variable, k), self.variable, "hinge", knot=k)
            for k in self.hinges
        ]
        if self.null_indicator:
            out.append(Feature(f"{self.variable} is null", self.variable, "null"))
        return out

    def bins(self) -> list[tuple[float | None, float | None]]:
        """``(None, lo)``, the sloped bands between consecutive edges, ``(hi, None)``."""
        edges: list[float | None] = [None, *self.band_edges(), None]
        return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    def rows(self) -> list[tuple[Any, Any]]:
        return [*self.bins(), (None, None)]

    def row_index(self, series: pl.Series) -> np.ndarray:
        x = series.cast(pl.Float64).to_numpy()
        idx = np.searchsorted(np.asarray(self.band_edges()), x, side="right")
        return np.where(np.isnan(x), len(self.band_edges()) + 1, idx).astype(np.int64)

    def transform(self, series: pl.Series) -> np.ndarray:
        x = series.cast(pl.Float64).to_numpy()
        nan = np.isnan(x)
        xc = np.clip(np.where(nan, self.lo, x), self.lo, self.hi)
        hinges = np.asarray(self.hinges)
        cols = np.maximum(xc[:, None] - hinges[None, :], 0.0)
        cols[nan, :] = 0.0  # nulls: the value at lo (all hinges zero)
        if self.null_indicator:
            cols = np.hstack([cols, nan[:, None].astype(np.float64)])
        return cols

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "variable": self.variable,
            "knots": list(self.knots),
            "clamp": [self.lo, self.hi],
            "null_indicator": self.null_indicator,
        }


def interaction_name(a: str, b: str) -> str:
    return f"{a}{INTERACTION_SEP}{b}"


@dataclass
class InteractionEncoder(Encoder):
    """Two-way interaction ``A × B`` on top of the two main effects.

    A *cell* is a pair of rate-table rows of the parents (see
    :meth:`Encoder.rows`: bins + null row for numerics, levels + Other row for
    categoricals). One 0/1 column is created per **kept** cell; kept cells are
    decided once from training data (``from_data``) and stored explicitly, so
    the design is reproducible without data. Cells that were not kept (too
    little exposure) get no column and therefore relativity 1.00.

    Parameters
    ----------
    a, b : Encoder
        The parent encoders (must be in the same :class:`DesignSpec`).
    cells : list[tuple[int, int]]
        Kept cells as ``(row_a, row_b)`` indices, in column order.
    exposure : list[list[float]]
        Training exposure per cell, shape ``(a.n_rows, b.n_rows)``.
    min_cell_exposure : float
        Share of the interaction's total training exposure a cell needed to
        be kept (recorded for the record and for scripts).
    penalty_weight : float
        Multiplier on the L1 penalty of the cell columns (1.0 = same as the
        mains, on the unstandardised scale).
    """

    kind: ClassVar[str] = "interaction"

    a: Encoder
    b: Encoder
    cells: list[tuple[int, int]]
    exposure: list[list[float]]
    min_cell_exposure: float = 0.005
    penalty_weight: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.a, InteractionEncoder) or isinstance(
            self.b, InteractionEncoder
        ):
            raise ValueError("Interactions of interactions are not supported")
        if self.a.variable == self.b.variable:
            raise ValueError("An interaction needs two different variables")
        for parent in (self.a, self.b):
            if INTERACTION_SEP in parent.variable:
                raise ValueError(
                    f"Variable name {parent.variable!r} contains the interaction "
                    f"separator {INTERACTION_SEP!r}; rename it before using it in an "
                    "interaction"
                )
        na, nb = self.a.n_rows, self.b.n_rows
        cells = [(int(i), int(j)) for i, j in self.cells]
        if len(set(cells)) != len(cells):
            raise ValueError(f"InteractionEncoder({self.variable!r}) cells not unique")
        for i, j in cells:
            if not (0 <= i < na and 0 <= j < nb):
                raise ValueError(
                    f"InteractionEncoder({self.variable!r}) cell {(i, j)} outside "
                    f"the parents' {na}×{nb} rows"
                )
        self.cells = cells
        exp = np.asarray(self.exposure, dtype=float)
        if exp.shape != (na, nb):
            raise ValueError(
                f"InteractionEncoder({self.variable!r}) exposure must be "
                f"{na}×{nb}, got {exp.shape}"
            )
        self.exposure = exp.tolist()
        if not 0.0 <= float(self.min_cell_exposure) < 1.0:
            raise ValueError("min_cell_exposure must be in [0, 1)")
        if float(self.penalty_weight) <= 0:
            raise ValueError("penalty_weight must be positive")

    # -- construction --------------------------------------------------------
    @classmethod
    def from_data(
        cls,
        a: Encoder,
        b: Encoder,
        data: pl.DataFrame,
        *,
        weights: np.ndarray | pl.Series | None = None,
        min_cell_exposure: float = 0.005,
        penalty_weight: float = 1.0,
    ) -> InteractionEncoder:
        """Decide the kept cells from training ``data``: a cell is kept when
        its share of the interaction's total exposure is at least
        ``min_cell_exposure`` (and it has any exposure at all)."""
        ia = a.row_index(data[a.variable])
        ib = b.row_index(data[b.variable])
        w = (
            np.ones(data.height)
            if weights is None
            else np.asarray(
                weights.to_numpy() if isinstance(weights, pl.Series) else weights,
                dtype=float,
            )
        )
        na, nb = a.n_rows, b.n_rows
        exposure = np.zeros((na, nb))
        np.add.at(exposure, (ia, ib), w)
        total = exposure.sum()
        share = exposure / total if total > 0 else exposure
        keep = (share >= min_cell_exposure) & (exposure > 0)
        cells = [(int(i), int(j)) for i, j in zip(*np.nonzero(keep), strict=True)]
        return cls(
            a,
            b,
            cells,
            exposure.tolist(),
            min_cell_exposure=min_cell_exposure,
            penalty_weight=penalty_weight,
        )

    # -- Encoder interface ---------------------------------------------------
    @property
    def variable(self) -> str:  # type: ignore[override]
        return interaction_name(self.a.variable, self.b.variable)

    @property
    def parents(self) -> tuple[str, str]:
        return self.a.variable, self.b.variable

    @property
    def required_columns(self) -> list[str]:
        return [self.a.variable, self.b.variable]

    def features(self) -> list[Feature]:
        rows_a, rows_b = self.a.rows(), self.b.rows()
        return [
            Feature(
                f"{self.variable}[{row_label(rows_a[i])} | {row_label(rows_b[j])}]",
                self.variable,
                "cell",
                level=f"{row_label(rows_a[i])} | {row_label(rows_b[j])}",
                cell=(i, j),
            )
            for i, j in self.cells
        ]

    def rows(self) -> list[tuple[Any, Any]]:
        """All cells (kept or not) as ``((from_a, to_a), (from_b, to_b))``,
        row-major over the parents' rows. Note the elements are *pairs of rows*,
        not edges: use :meth:`cell_labels` for text, not :func:`row_label`."""
        rows_a, rows_b = self.a.rows(), self.b.rows()
        return [(ra, rb) for ra in rows_a for rb in rows_b]

    def cell_labels(self) -> list[tuple[str, str]]:
        """``(label_a, label_b)`` per cell, in :meth:`rows` order."""
        rows_a, rows_b = self.a.rows(), self.b.rows()
        return [(row_label(ra), row_label(rb)) for ra in rows_a for rb in rows_b]

    def cell_index(self, data: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.a.row_index(data[self.a.variable]), self.b.row_index(
            data[self.b.variable]
        )

    def row_index(self, series: pl.Series) -> np.ndarray:  # pragma: no cover
        raise TypeError("InteractionEncoder needs two columns; use cell_index(frame)")

    def transform(self, series: pl.Series) -> np.ndarray:  # pragma: no cover
        raise TypeError("InteractionEncoder needs two columns; use transform_frame")

    def transform_frame(self, data: pl.DataFrame) -> np.ndarray:
        ia, ib = self.cell_index(data)
        nb = self.b.n_rows
        flat = ia * nb + ib
        out = np.zeros((data.height, len(self.cells)), dtype=np.float64)
        for col, (i, j) in enumerate(self.cells):
            out[:, col] = flat == (i * nb + j)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "a": self.a.variable,
            "b": self.b.variable,
            "cells": [list(c) for c in self.cells],
            "exposure": self.exposure,
            "min_cell_exposure": self.min_cell_exposure,
            "penalty_weight": self.penalty_weight,
        }


_ENCODERS: dict[str, type[Encoder]] = {
    StepEncoder.kind: StepEncoder,
    CategoricalEncoder.kind: CategoricalEncoder,
    LinearEncoder.kind: LinearEncoder,
}


def encoder_from_dict(
    raw: dict[str, Any], parents: dict[str, Encoder] | None = None
) -> Encoder:
    raw = dict(raw)
    kind = raw.pop("kind")
    if kind == InteractionEncoder.kind:
        parents = parents or {}
        try:
            a, b = parents[raw.pop("a")], parents[raw.pop("b")]
        except KeyError as exc:
            raise ValueError(
                f"Interaction refers to an unknown parent encoder {exc}"
            ) from exc
        raw["cells"] = [tuple(c) for c in raw.get("cells", [])]
        return InteractionEncoder(a, b, **raw)
    try:
        cls = _ENCODERS[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown encoder kind {kind!r}") from exc
    return cls(**raw)


def quantile_knots(
    series: pl.Series, n_bins: int = 20, *, round_to: float | None = None
) -> list[float]:
    """Knots at the interior quantiles of ``series`` (nulls ignored).

    Uses observed values (``interpolation="nearest"``) so integer variables get
    integer knots, drops duplicates and the minimum (``x >= min`` is all ones).
    Returns at most ``n_bins - 1`` knots.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    x = series.drop_nulls().cast(pl.Float64)
    x = x.filter(x.is_finite())
    if x.is_empty():
        return []
    probs = np.linspace(0, 1, n_bins + 1)[1:-1]
    knots = np.array([x.quantile(p, interpolation="nearest") for p in probs])
    if round_to is not None:
        knots = np.round(knots / round_to) * round_to
    knots = np.unique(knots)
    knots = knots[knots > x.min()]
    return [float(k) for k in knots]


def frequent_levels(
    series: pl.Series,
    *,
    min_share: float = 0.0025,
    max_levels: int | None = None,
    weights: pl.Series | None = None,
) -> list[str]:
    """Levels ordered by (weighted) frequency, most frequent first.

    Levels whose share of the non-null total is below ``min_share`` are dropped
    (they fall into ``Other``). Nulls never count as a level.
    """
    frame = pl.DataFrame({"level": series.cast(pl.Utf8)})
    if weights is not None:
        frame = frame.with_columns(pl.Series("w", weights).cast(pl.Float64))
    else:
        frame = frame.with_columns(pl.lit(1.0).alias("w"))
    counts = (
        frame.drop_nulls("level")
        .group_by("level")
        .agg(pl.col("w").sum())
        .sort(["w", "level"], descending=[True, False])
    )
    if counts.is_empty():
        return []
    total = counts["w"].sum()
    kept = counts.filter(pl.col("w") / total >= min_share)
    if max_levels is not None:
        kept = kept.head(max_levels)
    return kept["level"].to_list()


def round_range_outward(lo: float, hi: float) -> tuple[float, float]:
    """``(lo, hi)`` rounded outward to a multiple of ``10**(floor(log10(hi-lo))-2)``
    — integers stay integers; ``(0.0038, 99.997)`` becomes ``(0.0, 100.0)``."""
    if not hi > lo:
        return lo, hi
    step = 10.0 ** (int(np.floor(np.log10(hi - lo))) - 2)
    lo_r = float(np.floor(lo / step + 1e-9) * step)
    hi_r = float(np.ceil(hi / step - 1e-9) * step)
    # keep integers exact and avoid float noise like 79.99999999
    return float(round(lo_r, 10)), float(round(hi_r, 10))


def linear_encoder_from_data(
    variable: str,
    series: pl.Series,
    *,
    knots: list[float] | None = None,
    n_bins: int = 20,
    clamp: tuple[float, float] | None = None,
    null_indicator: bool = True,
) -> LinearEncoder:
    """A :class:`LinearEncoder` clamped to the training range (or ``clamp``),
    with ``knots`` (default: quantile knots) restricted to the open range.

    The default clamp is the training minimum / maximum rounded *outward* to a
    round number (two decades below the range, e.g. 18–80 stays 18–80, 0.0038–99.997
    becomes 0–100), so every training value stays inside it and the table edges
    read well."""
    x = series.drop_nulls().cast(pl.Float64)
    x = x.filter(x.is_finite())
    if x.is_empty():
        raise ValueError(f"Cannot build a linear term for {variable!r}: all null")
    if clamp is None:
        lo, hi = round_range_outward(float(x.min()), float(x.max()))
    else:
        lo, hi = (float(clamp[0]), float(clamp[1]))
    if not lo < hi:
        raise ValueError(
            f"Cannot build a linear term for {variable!r}: the clamp range "
            f"({lo}, {hi}) is empty (constant on train?)"
        )
    ks = list(knots) if knots is not None else quantile_knots(series, n_bins)
    ks = [float(k) for k in ks if lo < float(k) < hi]
    return LinearEncoder(variable, ks, (lo, hi), null_indicator=null_indicator)


@dataclass
class DesignSpec:
    """Ordered collection of encoders; builds the design matrix.

    Main effects come first, interactions after them (``add_interaction``
    enforces this), so the column layout is mains then cells.
    """

    encoders: dict[str, Encoder] = field(default_factory=dict)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_data(
        cls,
        data: pl.DataFrame,
        predictors: list[str],
        *,
        n_bins: int = 20,
        min_level_share: float = 0.0025,
        max_levels: int | None = None,
        null_indicator: bool = True,
        knots: dict[str, list[float]] | None = None,
        categorical: list[str] | None = None,
        weight_col: str | None = None,
        interactions: list[tuple[str, str]] | None = None,
        min_cell_exposure: float = 0.005,
        interaction_penalty_weight: float = 1.0,
        linear: list[str] | None = None,
        clamp: dict[str, tuple[float, float]] | None = None,
    ) -> DesignSpec:
        """Infer an encoder per predictor from (training) data.

        Numeric columns become :class:`StepEncoder` with quantile knots
        (override per variable via ``knots``, or force a numeric column to be
        treated as categorical via ``categorical``). Numeric columns listed in
        ``linear`` become :class:`LinearEncoder` instead, clamped to the training
        minimum/maximum (override via ``clamp``); knots outside the clamp range
        are dropped. Everything else becomes a :class:`CategoricalEncoder`
        keeping levels with at least ``min_level_share`` of the (optionally
        ``weight_col``-weighted) rows. ``interactions`` adds ``A × B`` terms
        (both must be predictors) whose kept cells are decided from the same data.
        """
        knots = knots or {}
        categorical = set(categorical or [])
        linear_vars = set(linear or [])
        clamp = clamp or {}
        weights = data[weight_col] if weight_col else None
        encoders: dict[str, Encoder] = {}
        for var in predictors:
            if var not in data.columns:
                raise KeyError(f"Predictor {var!r} not found in data")
            s = data[var]
            is_numeric = s.dtype in NUMERIC_DTYPES and var not in categorical
            if var in linear_vars and not is_numeric:
                raise ValueError(f"{var!r} is not numeric; it cannot be a linear term")
            if is_numeric and var in linear_vars:
                encoders[var] = linear_encoder_from_data(
                    var,
                    s,
                    knots=knots.get(var),
                    n_bins=n_bins,
                    clamp=clamp.get(var),
                    null_indicator=null_indicator,
                )
            elif is_numeric:
                ks = list(knots[var]) if var in knots else quantile_knots(s, n_bins)
                if not ks:
                    raise ValueError(
                        f"Cannot derive knots for {var!r}: it has fewer than two "
                        "distinct non-null values. Drop it or pass knots=... ."
                    )
                encoders[var] = StepEncoder(var, ks, null_indicator=null_indicator)
            else:
                levels = frequent_levels(
                    s, min_share=min_level_share, max_levels=max_levels, weights=weights
                )
                if not levels:
                    raise ValueError(f"Cannot derive levels for {var!r}: all null.")
                encoders[var] = CategoricalEncoder(var, levels)
        spec = cls(encoders)
        for a, b in interactions or []:
            for parent in (a, b):
                if parent not in spec.encoders:
                    raise ValueError(
                        f"Interaction ({a!r}, {b!r}): {parent!r} is not one of the "
                        f"predictors {list(spec.encoders)}; both parents must be "
                        "main effects of the same design"
                    )
            spec.add_interaction(
                InteractionEncoder.from_data(
                    spec[a],
                    spec[b],
                    data,
                    weights=weights,
                    min_cell_exposure=min_cell_exposure,
                    penalty_weight=interaction_penalty_weight,
                )
            )
        return spec

    def add_interaction(self, enc: InteractionEncoder) -> InteractionEncoder:
        """Append an interaction whose parents are already in the spec."""
        for parent in (enc.a, enc.b):
            if self.encoders.get(parent.variable) is not parent:
                raise ValueError(
                    f"Interaction parent {parent.variable!r} is not (the same object "
                    "as) a main effect of this spec"
                )
        if enc.variable in self.encoders:
            raise ValueError(f"Interaction {enc.variable!r} already in the spec")
        self.encoders[enc.variable] = enc
        return enc

    # -- introspection ----------------------------------------------------
    @property
    def variables(self) -> list[str]:
        return list(self.encoders)

    @property
    def main_effects(self) -> list[str]:
        return [
            v for v, e in self.encoders.items() if not isinstance(e, InteractionEncoder)
        ]

    @property
    def interactions(self) -> list[InteractionEncoder]:
        return [e for e in self.encoders.values() if isinstance(e, InteractionEncoder)]

    @property
    def features(self) -> list[Feature]:
        return [f for enc in self.encoders.values() for f in enc.features()]

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def n_features(self) -> int:
        return sum(enc.n_features for enc in self.encoders.values())

    @property
    def required_columns(self) -> list[str]:
        seen: dict[str, None] = {}
        for enc in self.encoders.values():
            for c in enc.required_columns:
                seen.setdefault(c, None)
        return list(seen)

    def slices(self) -> dict[str, slice]:
        """Column range of each variable in the design matrix."""
        out: dict[str, slice] = {}
        start = 0
        for var, enc in self.encoders.items():
            stop = start + enc.n_features
            out[var] = slice(start, stop)
            start = stop
        return out

    def __getitem__(self, variable: str) -> Encoder:
        return self.encoders[variable]

    def __contains__(self, variable: object) -> bool:
        return variable in self.encoders

    def __len__(self) -> int:
        return len(self.encoders)

    # -- matrix -----------------------------------------------------------
    def build(self, data: pl.DataFrame) -> np.ndarray:
        """Dense ``(n_rows, n_features)`` float64 design matrix for ``data``."""
        missing = [c for c in self.required_columns if c not in data.columns]
        if missing:
            raise KeyError(f"Data is missing predictor columns: {missing}")
        n = data.height
        out = np.empty((n, self.n_features), dtype=np.float64)
        for var, sl in self.slices().items():
            out[:, sl] = self.encoders[var].transform_frame(data)
        return out

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"encoders": [enc.to_dict() for enc in self.encoders.values()]}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DesignSpec:
        mains: dict[str, Encoder] = {}
        pending: list[dict[str, Any]] = []
        for e in raw["encoders"]:
            if e.get("kind") == InteractionEncoder.kind:
                pending.append(e)
            else:
                enc = encoder_from_dict(e)
                mains[enc.variable] = enc
        spec = cls(mains)
        for e in pending:
            spec.add_interaction(encoder_from_dict(e, parents=mains))  # type: ignore[arg-type]
        return spec

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> DesignSpec:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def __repr__(self) -> str:
        parts = []
        for var, enc in self.encoders.items():
            if isinstance(enc, StepEncoder):
                parts.append(f"{var}: step({len(enc.knots)} knots)")
            elif isinstance(enc, CategoricalEncoder):
                parts.append(f"{var}: categorical({len(enc.levels)} levels)")
            elif isinstance(enc, LinearEncoder):
                parts.append(
                    f"{var}: linear({len(enc.knots)} knots, clamp {enc.lo:g}–{enc.hi:g})"
                )
            elif isinstance(enc, InteractionEncoder):
                parts.append(
                    f"{var}: interaction({len(enc.cells)} of "
                    f"{enc.a.n_rows * enc.b.n_rows} cells)"
                )
            else:  # pragma: no cover
                parts.append(f"{var}: {enc.kind}")
        return f"DesignSpec({', '.join(parts)})"
