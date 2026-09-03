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
* :class:`LinearEncoder` -- piecewise-linear ("L-dummy") columns, one per
  *band* between consecutive knots: ``clip(x - k_j, 0, k_{j+1} - k_j)``, the
  amount of ``x`` that falls inside band ``j``. Each coefficient is therefore
  the *slope inside that band*, and with an L1 penalty the lasso sets slopes to
  exactly zero -- the curve is flat wherever the data does not insist on a
  slope. The curve is continuous by construction and exactly flat outside the
  training range. A term with no interior knots (``kind="continuous"`` on the
  Design page) is one band: a single slope on the raw clamped value.
* :class:`InteractionEncoder` -- a two-way interaction ``A × B`` on top of the
  two mains: one 0/1 column per *kept cell* of the parents' rate-table rows,
  so each coefficient is one cell's multiplicative adjustment.

Every encoder also knows its **rate-table rows** (:meth:`Encoder.rows`) and can
map a raw value to its row index (:meth:`Encoder.row_index`); that shared rule
is what keeps the rate tables, the interaction cells and the scorer aligned.

Two ways to build the matrix
----------------------------
:meth:`DesignSpec.build` returns either a dense float64 numpy array (small
books, and what every existing caller got) or a **tabmat**
:class:`~tabmat.SplitMatrix` that stores one integer code per row per variable
instead of the columns those codes stand for (big books). The two are the same
matrix: ``build(df, sparse=True).toarray()`` equals ``build(df, sparse=False)``
exactly, and a fit on either gives the same coefficients (piece G;
``tests/test_scale.py``). Which one you get by default is decided by the row
count — see :data:`SPARSE_ROW_THRESHOLD`.

The codes are the same integers the rate tables index by
(:meth:`Encoder.row_index`), which is why scoring never needs a matrix at all:
:meth:`DesignSpec.linear_predictor` adds up one lookup per variable in row
chunks (:meth:`Encoder.contribution`).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import polars as pl

from easy_glm.engine.models import (
    DEFAULT_OTHER_LABEL,
    INTERACTION_SEP,
    NULL_LABEL,
)

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

FeatureKind = Literal["step", "band", "null", "level", "other", "cell"]

#: Row count at or above which :meth:`DesignSpec.build` returns the compact
#: tabmat ``SplitMatrix`` instead of a dense float64 array. Below it the dense
#: matrix is at most ~0.3 GB for a 200-column design and the extra machinery
#: buys nothing; above it the dense matrix grows by 1.6 GB per million rows.
#: A 50,000-row book — the golden fit and every test fixture — is dense.
SPARSE_ROW_THRESHOLD = 200_000

#: Rows scored at a time by :meth:`DesignSpec.linear_predictor` (and therefore
#: by ``GLMFit.predict``). Big enough that the per-chunk overhead disappears,
#: small enough that the working set is a few tens of megabytes.
SCORING_CHUNK_ROWS = 500_000


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


def _check_penalty_weight(enc: Any) -> None:
    """A penalty weight must be a finite number ``>= 0`` (0 = unpenalised)."""
    value = float(enc.penalty_weight)
    if not np.isfinite(value) or value < 0:
        raise ValueError(
            f"{type(enc).__name__}({enc.variable!r}) penalty_weight must be a "
            f"finite number >= 0 (0 = unpenalised), got {enc.penalty_weight!r}"
        )
    enc.penalty_weight = value


class Encoder(ABC):
    """Turns one raw column (or two, for interactions) into design columns.

    Every encoder carries a ``penalty_weight``: how hard this variable's columns
    are penalised relative to the rest of the design (glum's ``P1``; see
    :func:`easy_glm.core.fit.penalty_weights`). 1.0 is the default, a larger
    number shrinks the variable harder, and **0 leaves it unpenalised** — the
    usual reason being a categorical main effect (e.g. territory) an actuary
    wants kept in full while the lasso thins everything else. It multiplies the
    per-column rules that already exist (band rises, interaction cells), never
    replaces them.
    """

    kind: ClassVar[str]
    variable: str
    penalty_weight: float = 1.0

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

    # -- integer codes: the compact form of a design row --------------------
    def codes(self, data: pl.DataFrame) -> np.ndarray:
        """One ``int32`` code per row — the compact stand-in for this term's
        columns. For everything but an interaction it *is* the rate-table row
        index (:meth:`row_index`), which is what lets the same integers drive
        the design matrix, the rate tables and the scorer."""
        return self.row_index(data[self.variable]).astype(np.int32, copy=False)

    @property
    def n_codes(self) -> int:
        """Number of distinct codes (``max code + 1``)."""
        return self.n_rows

    def contribution(self, data: pl.DataFrame, coef: np.ndarray) -> np.ndarray:
        """This term's part of the linear predictor, ``columns @ coef``,
        computed **without building the columns** (a table lookup per row).

        Equals ``transform_frame(data) @ coef`` to floating-point noise; it is
        how :meth:`DesignSpec.linear_predictor`, and therefore
        ``GLMFit.predict``, scores without ever materialising a design matrix.
        """
        table = self.lookup_table(np.asarray(coef, dtype=np.float64))
        return table[self.codes(data)]

    def lookup_table(self, coef: np.ndarray) -> np.ndarray:
        """Linear-predictor contribution of every code, as a float64 array of
        length :attr:`n_codes` (relative to the term's reference row)."""
        raise NotImplementedError(
            f"{type(self).__name__} has no code lookup table"
        )  # pragma: no cover


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
    penalty_weight: float = 1.0

    def __post_init__(self) -> None:
        _check_penalty_weight(self)
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

    def lookup_table(self, coef: np.ndarray) -> np.ndarray:
        """``[0, cumsum(step coefficients), null coefficient]`` — one entry per
        bin plus the null row, indexed by :meth:`codes`. The bin coefficients
        are *increments*, so the effect of bin ``j`` is their partial sum."""
        n_knots = len(self.knots)
        null = float(coef[n_knots]) if self.null_indicator else 0.0
        return np.concatenate([[0.0], np.cumsum(coef[:n_knots]), [null]])

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "variable": self.variable,
            "knots": list(self.knots),
            "null_indicator": self.null_indicator,
            "penalty_weight": self.penalty_weight,
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
    other_label: str = DEFAULT_OTHER_LABEL
    penalty_weight: float = 1.0

    def __post_init__(self) -> None:
        _check_penalty_weight(self)
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

    def lookup_table(self, coef: np.ndarray) -> np.ndarray:
        """``[0, one coefficient per non-reference level, Other]``, indexed by
        :meth:`codes` (0 = the reference level, last = Other / unseen / null)."""
        return np.concatenate([[0.0], coef])

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "variable": self.variable,
            "levels": list(self.levels),
            "other_label": self.other_label,
            "penalty_weight": self.penalty_weight,
        }


def _band_name(variable: str, lo: float, hi: float) -> str:
    """Name of the per-band column: the amount of ``variable`` inside the band."""
    return f"{variable} in [{format_knot(lo)}, {format_knot(hi)})"


@dataclass
class LinearEncoder(Encoder):
    """Piecewise-linear (L-dummy) encoding of a numeric variable.

    ``x`` is first clipped to ``clamp = (lo, hi)`` and then expanded into one
    column per **band** between consecutive edges ``[lo, k1, ..., km, hi]``:
    ``clip(x_clipped - k_j, 0, k_{j+1} - k_j)``, i.e. how much of ``x`` falls
    inside band ``j``. Coefficient ``beta_j`` is therefore the *slope inside
    band j* on the linear-predictor scale, and with an L1 penalty the lasso
    drives slopes to exactly zero: the curve is **flat unless the data insists
    on a slope** (the actuary's answer to Q3), not merely free of bends. The
    curve is continuous by construction (the bands tile ``[lo, hi]`` and each
    column is capped at its band width) and exactly flat outside ``[lo, hi]`` —
    beyond the training range the relativity stays at its value at the nearer
    clamp. A monotone constraint is a sign bound on these coefficients
    (see :func:`easy_glm.core.fit.monotone_bounds`).

    Nulls get all-zero band columns (the value at ``lo``) plus, by default, an
    ``is null`` column so they can carry their own effect.

    With no interior knots there is a single band ``[lo, hi]``: one slope on the
    raw clamped value — what the Design page offers as ``kind="continuous"``.

    Rate-table rows (:meth:`rows`): ``(None, lo)`` flat, one band per pair of
    consecutive edges ``[lo, k1, ..., km, hi]`` (log-linear inside), ``(hi, None)``
    flat, then the null row.

    Parameters
    ----------
    variable : str
        Column name.
    knots : list[float]
        Interior knots, strictly inside ``(lo, hi)``; the slope may change there.
        Empty = one band, a single slope ("continuous").
    clamp : (lo, hi)
        Range the raw value is clipped to before the band columns; the curve is
        flat outside it. ``DesignSpec.from_data`` uses the training
        minimum/maximum.
    null_indicator : bool
        Add a ``1{x is null}`` column.
    """

    kind: ClassVar[str] = "linear"

    variable: str
    knots: list[float]
    clamp: tuple[float, float]
    null_indicator: bool = True
    penalty_weight: float = 1.0

    def __post_init__(self) -> None:
        _check_penalty_weight(self)
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
        # a band narrower than a billionth of the range is not a band: its slope
        # is unidentifiable, its column name collides with its neighbour's and
        # ``from_rate_tables`` would read the table back with an infinite slope
        edges = [lo, *knots, hi]
        thin = [
            (edges[i], edges[i + 1])
            for i in range(len(edges) - 1)
            if edges[i + 1] - edges[i] < (hi - lo) * 1e-9
        ]
        if thin:
            raise ValueError(
                f"LinearEncoder({self.variable!r}) knots are too close together: "
                f"band(s) {thin} are narrower than a billionth of the clamp range "
                f"({lo}, {hi}); drop or move the knot"
            )
        self.knots = knots
        self.clamp = (lo, hi)

    @property
    def lo(self) -> float:
        return self.clamp[0]

    @property
    def hi(self) -> float:
        return self.clamp[1]

    def band_edges(self) -> list[float]:
        """``[lo, k1, ..., km, hi]`` — the edges of the sloped bands. Passing them
        to ``ae_by_variable(knots=...)`` gives labels identical to the table's."""
        return [self.lo, *self.knots, self.hi]

    def band_starts(self) -> list[float]:
        """Lower edge of each band — the knot each slope column starts from."""
        return [self.lo, *self.knots]

    def band_widths(self) -> list[float]:
        """Width of each band; the column for band ``j`` is capped at it."""
        edges = self.band_edges()
        return [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]

    @property
    def n_bands(self) -> int:
        return len(self.knots) + 1

    def features(self) -> list[Feature]:
        edges = self.band_edges()
        out = [
            Feature(
                _band_name(self.variable, edges[i], edges[i + 1]),
                self.variable,
                "band",
                knot=edges[i],
            )
            for i in range(len(edges) - 1)
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
        starts = np.asarray(self.band_starts())
        widths = np.asarray(self.band_widths())
        # amount of x inside each band: 0 below it, its width above it
        cols = np.clip(xc[:, None] - starts[None, :], 0.0, widths[None, :])
        cols[nan, :] = 0.0  # nulls: the value at lo (all bands empty)
        if self.null_indicator:
            cols = np.hstack([cols, nan[:, None].astype(np.float64)])
        return cols

    def contribution(self, data: pl.DataFrame, coef: np.ndarray) -> np.ndarray:
        """The log relativity at every row's value.

        A piecewise-linear term is the one term that is **not** a pure lookup:
        the effect moves *inside* a band, so the band index alone does not
        determine it. It is still computed without any design columns — the
        slope of each band times the amount of the (clamped) value that falls
        inside it, which is the same continuous curve the rate table carries.
        """
        coef = np.asarray(coef, dtype=np.float64)
        x = data[self.variable].cast(pl.Float64).to_numpy()
        nan = np.isnan(x)
        xc = np.clip(np.where(nan, self.lo, x), self.lo, self.hi)
        out = np.zeros(len(x), dtype=np.float64)
        for slope, start, width in zip(
            coef[: self.n_bands], self.band_starts(), self.band_widths(), strict=True
        ):
            if slope != 0.0:
                out += slope * np.clip(xc - start, 0.0, width)
        out[nan] = float(coef[self.n_bands]) if self.null_indicator else 0.0
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "variable": self.variable,
            "knots": list(self.knots),
            "clamp": [self.lo, self.hi],
            "null_indicator": self.null_indicator,
            "penalty_weight": self.penalty_weight,
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
        codes = self.codes(data)
        out = np.zeros((data.height, len(self.cells)), dtype=np.float64)
        for col in range(len(self.cells)):
            np.copyto(out[:, col], codes == col + 1, casting="unsafe")
        return out

    def codes(self, data: pl.DataFrame) -> np.ndarray:
        """Kept-cell code per row: ``0`` = "no kept cell" (every column zero,
        relativity 1.00), ``1 + k`` = the ``k``-th kept cell, in column order.

        Because "no cell" is a code of its own the whole interaction is one
        integer per row, whatever the number of cells — which is what keeps a
        big interaction from costing ``8 * n_rows * n_cells`` bytes."""
        ia, ib = self.cell_index(data)
        nb = self.b.n_rows
        lookup = np.zeros(self.a.n_rows * nb, dtype=np.int32)
        for col, (i, j) in enumerate(self.cells):
            lookup[i * nb + j] = col + 1
        return lookup[ia * nb + ib]

    @property
    def n_codes(self) -> int:
        return len(self.cells) + 1

    def lookup_table(self, coef: np.ndarray) -> np.ndarray:
        """``[0, one coefficient per kept cell]``, indexed by :meth:`codes`."""
        return np.concatenate([[0.0], coef])

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
    penalty_weight: float = 1.0,
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
        # min()/max() of a non-empty Float64 series are floats; polars types them
        # as "any scalar", so the cast is spelled out for the type checker too
        lo, hi = round_range_outward(float(x.min()), float(x.max()))  # type: ignore[arg-type]
    else:
        lo, hi = (float(clamp[0]), float(clamp[1]))
    if not lo < hi:
        raise ValueError(
            f"Cannot build a linear term for {variable!r}: the clamp range "
            f"({lo}, {hi}) is empty (constant on train?)"
        )
    ks = list(knots) if knots is not None else quantile_knots(series, n_bins)
    ks = [float(k) for k in ks if lo < float(k) < hi]
    return LinearEncoder(
        variable,
        ks,
        (lo, hi),
        null_indicator=null_indicator,
        penalty_weight=penalty_weight,
    )


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
        penalty_weight: dict[str, float] | None = None,
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
        ``penalty_weight`` sets a variable's own L1 weight (1.0 = as the rest of
        the design, 0 = unpenalised; see
        :func:`easy_glm.core.fit.penalty_weights`).
        """
        knots = knots or {}
        pweight = penalty_weight or {}
        categorical_vars = set(categorical or [])
        linear_vars = set(linear or [])
        clamp = clamp or {}
        weights = data[weight_col] if weight_col else None
        encoders: dict[str, Encoder] = {}
        for var in predictors:
            if var not in data.columns:
                raise KeyError(f"Predictor {var!r} not found in data")
            s = data[var]
            is_numeric = s.dtype in NUMERIC_DTYPES and var not in categorical_vars
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
                    penalty_weight=float(pweight.get(var, 1.0)),
                )
            elif is_numeric:
                ks = list(knots[var]) if var in knots else quantile_knots(s, n_bins)
                if not ks:
                    raise ValueError(
                        f"Cannot derive knots for {var!r}: it has fewer than two "
                        "distinct non-null values. Drop it or pass knots=... ."
                    )
                encoders[var] = StepEncoder(
                    var,
                    ks,
                    null_indicator=null_indicator,
                    penalty_weight=float(pweight.get(var, 1.0)),
                )
            else:
                levels = frequent_levels(
                    s, min_share=min_level_share, max_levels=max_levels, weights=weights
                )
                if not levels:
                    raise ValueError(f"Cannot derive levels for {var!r}: all null.")
                other = DEFAULT_OTHER_LABEL
                while other in levels:  # a real level called "Other" (e.g. a recode)
                    other += " (lumped)"
                encoders[var] = CategoricalEncoder(
                    var,
                    levels,
                    other_label=other,
                    penalty_weight=float(pweight.get(var, 1.0)),
                )
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

    # -- the two stages of an interaction fit ------------------------------
    def main_effects_spec(self) -> DesignSpec:
        """The main effects of this spec on their own — **stage 1** of a
        two-stage interaction fit (:func:`~easy_glm.core.fit.fit_two_stage`).

        It is exactly the design the same model *without* any interaction would
        build, which is what lets stage 1 (and therefore every main rate table
        and the base rate) be identical with and without the interaction. The
        encoder objects are shared, not copied."""
        return DesignSpec({v: self.encoders[v] for v in self.main_effects})

    def interactions_spec(self) -> DesignSpec:
        """The interaction cell columns on their own — **stage 2** of a
        two-stage fit. The parent encoders are shared with
        :meth:`main_effects_spec`, so the cell columns are the same columns the
        joint design would have built, in the same order."""
        return DesignSpec({e.variable: e for e in self.interactions})

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
    def _require_columns(self, data: pl.DataFrame) -> None:
        missing = [c for c in self.required_columns if c not in data.columns]
        if missing:
            raise KeyError(f"Data is missing predictor columns: {missing}")

    def codes(self, data: pl.DataFrame) -> dict[str, np.ndarray]:
        """One ``int32`` code per row per variable (:meth:`Encoder.codes`).

        This is the whole design in ``4`` bytes per row per variable — the form
        both :meth:`build` and :meth:`linear_predictor` work from."""
        self._require_columns(data)
        return {var: enc.codes(data) for var, enc in self.encoders.items()}

    def build(
        self, data: pl.DataFrame, *, sparse: bool | None = None
    ) -> np.ndarray | Any:
        """The design matrix for ``data``.

        Parameters
        ----------
        sparse : bool, optional
            ``False`` returns the dense ``(n_rows, n_features)`` float64 numpy
            array. ``True`` returns a tabmat :class:`~tabmat.SplitMatrix` that
            holds integer codes instead of columns — the same matrix, a small
            fraction of the memory (see :meth:`build_sparse`). The default is
            **by row count**: dense below :data:`SPARSE_ROW_THRESHOLD` rows
            (200,000), the split matrix at or above it. The threshold is where
            the dense matrix starts to be worth avoiding but small enough that
            an ordinary book still takes the path every 0.3 model took; a
            50,000-row book — the golden fit, every test fixture — is dense.

        Both forms are float64 and produce the same fit: coefficients and
        predictions agree to 1e-10 with the same non-zero set
        (``tests/test_scale.py``).
        """
        if sparse is None:
            sparse = data.height >= SPARSE_ROW_THRESHOLD
        return self.build_sparse(data) if sparse else self.build_dense(data)

    def build_dense(self, data: pl.DataFrame) -> np.ndarray:
        """Dense ``(n_rows, n_features)`` float64 design matrix for ``data``.

        Written column by column straight into the output array, from the
        integer codes, so nothing bigger than the result is ever allocated.
        """
        self._require_columns(data)
        n = data.height
        out = np.zeros((n, self.n_features), dtype=np.float64)
        for var, sl in self.slices().items():
            enc = self.encoders[var]
            if isinstance(enc, LinearEncoder):
                out[:, sl] = enc.transform_frame(data)
                continue
            code = enc.codes(data)
            if isinstance(enc, StepEncoder):
                n_knots = len(enc.knots)
                for j in range(n_knots):
                    np.copyto(
                        out[:, sl.start + j],
                        (code >= j + 1) & (code <= n_knots),
                        casting="unsafe",
                    )
                if enc.null_indicator:
                    np.copyto(
                        out[:, sl.start + n_knots],
                        code == n_knots + 1,
                        casting="unsafe",
                    )
            else:  # categorical levels[1:] + Other, or interaction cells
                for j in range(enc.n_features):
                    np.copyto(out[:, sl.start + j], code == j + 1, casting="unsafe")
        return out

    def build_sparse(self, data: pl.DataFrame) -> Any:
        """A tabmat :class:`~tabmat.SplitMatrix` for ``data`` (float64).

        The blocks, **in this order**:

        1. one :class:`~easy_glm.core.stepmatrix.StepMatrix` per step variable
           — a bin index per row, ``4`` bytes whatever the number of knots;
        2. one :class:`~tabmat.CategoricalMatrix` per categorical variable —
           a level index per row, the reference level dropped;
        3. one :class:`~tabmat.CategoricalMatrix` per interaction — the
           kept-cell code per row, with "no cell" as the dropped category, so
           a row in no kept cell is a row of zeros;
        4. one dense float64 block holding what is left: the ``is null``
           indicators and the piecewise-linear band columns.

        Step blocks come first because ``SplitMatrix`` only ever asks the
        *earlier* block of a pair for their cross product, and tabmat's own
        blocks do not know ours (see :mod:`easy_glm.core.stepmatrix`).

        The linear bands stay **dense on purpose**. Their columns are
        real-valued, not 0/1, so the cumulative-sum trick that makes the step
        blocks free does not apply; a ``(band index, amount into the band)``
        pair per row would work — the row is the band's widths up to the band,
        then the overlap, then zeros — but it needs its own sandwich kernel for
        a term that has few columns to begin with (a piecewise-linear term is
        an explicit per-variable choice, and 8 to 20 bands is typical). The
        cost is stated rather than hidden: ``8 * n_rows`` bytes per band, which
        at 5M rows and 20 bands is 0.8 GB for that one term. If that ever
        binds, the pair representation is the fix.
        """
        import tabmat as tm

        from .stepmatrix import StepMatrix

        self._require_columns(data)
        if self.n_features == 0:
            raise ValueError("Cannot build a design matrix with no features")
        n = data.height
        if n == 0:
            # tabmat's CategoricalMatrix cannot be built from zero rows; an
            # empty dense block has the right shape and nothing to save.
            return tm.DenseMatrix(
                np.asarray(self.build(data, sparse=False), dtype=np.float64)
            )
        codes = self.codes(data)
        slices = self.slices()
        step_blocks: list[tuple[Any, np.ndarray]] = []
        cat_blocks: list[tuple[Any, np.ndarray]] = []
        cell_blocks: list[tuple[Any, np.ndarray]] = []
        dense_cols: list[int] = []
        dense_values: list[np.ndarray] = []

        for var, sl in slices.items():
            enc = self.encoders[var]
            code = codes[var]
            if isinstance(enc, StepEncoder):
                n_knots = len(enc.knots)
                step_blocks.append(
                    (
                        StepMatrix(code, n_knots, dtype=np.float64, name=var),
                        np.arange(sl.start, sl.start + n_knots),
                    )
                )
                if enc.null_indicator:
                    dense_cols.append(sl.start + n_knots)
                    dense_values.append(code == n_knots + 1)
            elif isinstance(enc, LinearEncoder):
                block = enc.transform_frame(data)
                for j in range(enc.n_features):
                    dense_cols.append(sl.start + j)
                    dense_values.append(block[:, j])
            elif isinstance(enc, CategoricalEncoder):
                cat_blocks.append(
                    (
                        tm.CategoricalMatrix(
                            code,
                            categories=np.arange(enc.n_codes),
                            drop_first=True,
                            dtype=np.float64,
                            column_name=var,
                        ),
                        np.arange(sl.start, sl.stop),
                    )
                )
            elif isinstance(enc, InteractionEncoder):
                cell_blocks.append(
                    (
                        tm.CategoricalMatrix(
                            code,
                            categories=np.arange(enc.n_codes),
                            drop_first=True,
                            dtype=np.float64,
                            column_name=var,
                        ),
                        np.arange(sl.start, sl.stop),
                    )
                )
            else:  # pragma: no cover - a new encoder must choose a block
                raise NotImplementedError(
                    f"No sparse design block for {type(enc).__name__}; build "
                    "with sparse=False or add one in DesignSpec.build_sparse"
                )

        blocks = [*step_blocks, *cat_blocks, *cell_blocks]
        if dense_cols:
            dense = np.empty((n, len(dense_cols)), dtype=np.float64, order="F")
            for j, values in enumerate(dense_values):
                np.copyto(dense[:, j], values, casting="unsafe")
            blocks.append((tm.DenseMatrix(dense), np.asarray(dense_cols)))
        matrix = tm.SplitMatrix([m for m, _ in blocks], [i for _, i in blocks])
        _check_step_blocks_first(matrix)
        return matrix

    def expected_design_bytes(self, n_rows: int) -> int:
        """Bytes :meth:`build_sparse` should need for ``n_rows`` rows.

        ``n * (4 * step variables + 4 * categoricals + 4 * interactions
        + 8 * dense columns)``, where the dense columns are the ``is null``
        indicators and the piecewise-linear bands. The benchmark
        (``scripts/bench_scale.py``) asserts the real matrix against this.
        """
        per_row = 0
        for enc in self.encoders.values():
            if isinstance(enc, StepEncoder):
                per_row += 4 + (8 if enc.null_indicator else 0)
            elif isinstance(enc, LinearEncoder):
                per_row += 8 * enc.n_features
            else:
                per_row += 4
        return int(n_rows) * per_row

    # -- scoring without a matrix -----------------------------------------
    def linear_predictor(
        self,
        data: pl.DataFrame,
        coef: np.ndarray,
        intercept: float = 0.0,
        *,
        chunk_rows: int = SCORING_CHUNK_ROWS,
    ) -> np.ndarray:
        """``intercept + design @ coef``, computed from the codes in row chunks.

        One table lookup per variable per chunk, in float64 — the same
        arithmetic ``RateModel`` does, which is why the two agree exactly.
        Nothing the size of a design matrix is ever allocated, so scoring a
        5M-row book costs one float64 vector, not a second copy of the design.
        """
        self._require_columns(data)
        coef = np.asarray(coef, dtype=np.float64)
        if coef.shape != (self.n_features,):
            raise ValueError(
                f"coef must have one value per design column ({self.n_features}); "
                f"got shape {coef.shape}"
            )
        n = data.height
        out = np.full(n, float(intercept), dtype=np.float64)
        slices = self.slices()
        step = max(int(chunk_rows), 1)
        for start in range(0, n, step):
            part = data.slice(start, step)
            block = out[start : start + part.height]
            for var, sl in slices.items():
                block += self.encoders[var].contribution(part, coef[sl])
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


# --------------------------------------------------------------------------
# helpers for the compact (tabmat) design matrix
# --------------------------------------------------------------------------
def _check_step_blocks_first(matrix: Any) -> None:
    """Fail loudly if a ``StepMatrix`` block is not before every other block.

    ``SplitMatrix.sandwich`` computes the cross product of blocks ``i`` and
    ``j`` (``i < j``) by asking block ``i``, and tabmat's own blocks raise
    ``TypeError`` on a block type they have never heard of. Ordering is
    therefore a correctness condition, not a preference, and it is cheap to
    assert once at construction rather than to debug inside a Hessian.
    """
    from .stepmatrix import StepMatrix

    seen_other = False
    for block in matrix.matrices:
        if isinstance(block, StepMatrix):
            if seen_other:
                raise RuntimeError(
                    "StepMatrix blocks must come before every other block of "
                    "the SplitMatrix (cross-sandwich dispatch); got "
                    f"{[type(m).__name__ for m in matrix.matrices]}"
                )
        else:
            seen_other = True


def design_bytes(design: Any) -> int:
    """Bytes a built design matrix actually holds (dense array or SplitMatrix).

    Counts the payload of every block — the codes of a ``StepMatrix``, the
    indices of a ``CategoricalMatrix``, the values of a dense block — so the
    benchmark can compare it with
    :meth:`DesignSpec.expected_design_bytes`.
    """
    import tabmat as tm

    from .stepmatrix import StepMatrix

    if isinstance(design, np.ndarray):
        return int(design.nbytes)
    if isinstance(design, tm.SplitMatrix):
        return sum(design_bytes(block) for block in design.matrices)
    if isinstance(design, StepMatrix):
        return design.nbytes
    if isinstance(design, tm.CategoricalMatrix):
        return int(np.asarray(design.indices).nbytes)
    if isinstance(design, tm.DenseMatrix):
        return int(np.asarray(design.unpack()).nbytes)
    raise TypeError(f"design_bytes does not know {type(design).__name__}")
