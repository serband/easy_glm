"""Design specification: how raw columns become GLM features.

A :class:`DesignSpec` is the single source of truth for a model's features.
It maps each predictor to an :class:`Encoder` that knows how to turn a raw
polars column into design-matrix columns *and* what every column means
(variable, kind, knot or level). Because that metadata is structured, rate
tables can be read straight off the fitted coefficients (see
:mod:`easy_glm.core.tables`) and the spec round-trips through JSON.

Two encoders exist:

* :class:`StepEncoder` -- AGLM-style "O-dummies": one column ``1{x >= k}`` per
  knot. With an L1 penalty each coefficient is the *increment* of the effect
  at that knot, so unchanged increments are shrunk to exactly zero and the
  fitted curve is a piecewise-constant step function with data-driven bands.
  Nulls get all-zero step columns (they sit in the lowest bin) plus, by
  default, a dedicated ``is null`` column so they can carry their own effect.
* :class:`CategoricalEncoder` -- one column per kept level except the
  reference (the most frequent level), plus an ``Other`` column for lumped,
  unseen and null values.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import polars as pl

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

FeatureKind = Literal["step", "null", "level", "other"]


@dataclass(frozen=True)
class Feature:
    """One design-matrix column and what it means."""

    name: str
    variable: str
    kind: FeatureKind
    knot: float | None = None
    level: str | None = None


def format_knot(value: float) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


class Encoder(ABC):
    """Turns one raw column into design-matrix columns."""

    kind: ClassVar[str]
    variable: str

    @abstractmethod
    def features(self) -> list[Feature]: ...

    @abstractmethod
    def transform(self, series: pl.Series) -> np.ndarray:
        """Return an ``(n_rows, n_features)`` float64 array."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @property
    def n_features(self) -> int:
        return len(self.features())


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


_ENCODERS: dict[str, type[Encoder]] = {
    StepEncoder.kind: StepEncoder,
    CategoricalEncoder.kind: CategoricalEncoder,
}


def encoder_from_dict(raw: dict[str, Any]) -> Encoder:
    raw = dict(raw)
    kind = raw.pop("kind")
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


@dataclass
class DesignSpec:
    """Ordered collection of encoders; builds the design matrix."""

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
    ) -> DesignSpec:
        """Infer an encoder per predictor from (training) data.

        Numeric columns become :class:`StepEncoder` with quantile knots
        (override per variable via ``knots``, or force a numeric column to be
        treated as categorical via ``categorical``). Everything else becomes a
        :class:`CategoricalEncoder` keeping levels with at least
        ``min_level_share`` of the (optionally ``weight_col``-weighted) rows.
        """
        knots = knots or {}
        categorical = set(categorical or [])
        weights = data[weight_col] if weight_col else None
        encoders: dict[str, Encoder] = {}
        for var in predictors:
            if var not in data.columns:
                raise KeyError(f"Predictor {var!r} not found in data")
            s = data[var]
            is_numeric = s.dtype in NUMERIC_DTYPES and var not in categorical
            if is_numeric:
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
        return cls(encoders)

    # -- introspection ----------------------------------------------------
    @property
    def variables(self) -> list[str]:
        return list(self.encoders)

    @property
    def features(self) -> list[Feature]:
        return [f for enc in self.encoders.values() for f in enc.features()]

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def n_features(self) -> int:
        return sum(enc.n_features for enc in self.encoders.values())

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
        missing = [v for v in self.encoders if v not in data.columns]
        if missing:
            raise KeyError(f"Data is missing predictor columns: {missing}")
        n = data.height
        out = np.empty((n, self.n_features), dtype=np.float64)
        for var, sl in self.slices().items():
            out[:, sl] = self.encoders[var].transform(data[var])
        return out

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"encoders": [enc.to_dict() for enc in self.encoders.values()]}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DesignSpec:
        encs = [encoder_from_dict(e) for e in raw["encoders"]]
        return cls({e.variable: e for e in encs})

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
            else:
                parts.append(f"{var}: categorical({len(enc.levels)} levels)")
        return f"DesignSpec({', '.join(parts)})"
