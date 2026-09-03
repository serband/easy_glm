"""Fit a penalised GLM on the design matrix of a :class:`DesignSpec`.

Three things here are about **size** rather than statistics (piece G):

* the design handed to glum is whatever :meth:`DesignSpec.build` returns — a
  dense float64 array for a small book, a compact tabmat ``SplitMatrix`` for a
  big one. Both give the same fit; ``sparse=`` forces either.
* **scoring never builds a design matrix.** ``GLMFit.predict`` and
  ``GLMFit.linear_predictor`` add up one rate-table lookup per variable in row
  chunks (:meth:`DesignSpec.linear_predictor`), which is exactly what
  ``RateModel`` does — so the two agree by construction and diagnostics on a
  5M-row book never materialise a second copy of the design.
* ``aggregate=True`` fits one row per *distinct design row* with the summed
  weights. Exact, opt-in, and worth it only on a coarse design.
"""

from __future__ import annotations

import threading
import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from glum import (
    GeneralizedLinearRegressor,
    GeneralizedLinearRegressorCV,
    TweedieDistribution,
)

from .design import (
    SCORING_CHUNK_ROWS,
    DesignSpec,
    InteractionEncoder,
    LinearEncoder,
    StepEncoder,
)
from .stepmatrix import install_glum_shim

_FAMILY_ALIASES: dict[str, str] = {
    "poisson": "poisson",
    "gamma": "gamma",
    "gaussian": "normal",
    "normal": "normal",
    "binomial": "binomial",
    "inverse_gaussian": "inverse.gaussian",
    "inverse.gaussian": "inverse.gaussian",
    "tweedie": "tweedie",
}

Direction = str  # "increasing" | "decreasing"


def resolve_family(family: Any) -> tuple[Any, str, str]:
    """Return ``(glum_family, canonical_name, default_link)``.

    Strings are case-insensitive (``"Poisson"``, ``"gaussian"``, ``"tweedie"``
    = Tweedie with power 1.5). glum distribution objects pass through. Every
    family defaults to the log link except binomial (logit), because rate
    tables are multiplicative.
    """
    if isinstance(family, str):
        key = family.strip().lower()
        if key not in _FAMILY_ALIASES:
            raise ValueError(
                f"Unknown family {family!r}. Use one of "
                f"{sorted(_FAMILY_ALIASES)} or a glum distribution object."
            )
        name = _FAMILY_ALIASES[key]
        fam: Any = TweedieDistribution(1.5) if name == "tweedie" else name
    else:
        fam = family
        name = type(family).__name__.replace("Distribution", "").lower()
    link = "logit" if name == "binomial" else "log"
    return fam, name, link


def monotone_bounds(
    spec: DesignSpec, monotone: Mapping[str, Direction]
) -> tuple[np.ndarray, np.ndarray]:
    """Coefficient bounds that make a numeric effect monotone in the variable.

    * **Step** terms: each column is the *increment* at a knot, so a monotone
      curve just needs every increment to share a sign.
    * **Piecewise-linear** terms: each column is the *slope inside one band*
      (see :class:`~easy_glm.core.design.LinearEncoder`), so a monotone curve
      needs every slope to share a sign. Nothing is implied about the *change*
      of slope, so the curve is monotone without being forced convex — that is
      why the constraint is available for linear terms.

    Either way: ``lower = 0`` for increasing, ``upper = 0`` for decreasing.
    The null-indicator column is never bounded (nulls are not on the curve).
    Works with the L1 penalty: a lasso'd slope is 0, which both bounds allow.

    The bound binds the **factor's own curve**, not any interaction cell sitting
    on top of it: ``A × B`` can still turn the combined effect round for some
    level of ``B``. That has always been true for step terms; it matters more
    now that the constraint is offered on the smooth curves an actuary is most
    likely to constrain.
    """
    p = spec.n_features
    lower = np.full(p, -np.inf)
    upper = np.full(p, np.inf)
    slices = spec.slices()
    features = spec.features
    for var, direction in monotone.items():
        if var not in spec:
            raise KeyError(f"monotone: {var!r} is not a predictor in the spec")
        if not isinstance(spec[var], StepEncoder | LinearEncoder):
            raise ValueError(
                f"monotone: {var!r} is categorical or an interaction; only numeric "
                "(step or piecewise-linear) variables can be constrained"
            )
        if direction not in ("increasing", "decreasing"):
            raise ValueError(
                f"monotone[{var!r}] must be 'increasing' or 'decreasing', "
                f"got {direction!r}"
            )
        sl = slices[var]
        idx = [
            i for i in range(sl.start, sl.stop) if features[i].kind in ("step", "band")
        ]
        if direction == "increasing":
            lower[idx] = 0.0
        else:
            upper[idx] = 0.0
    return lower, upper


def _validate_target(y: np.ndarray, family: str) -> None:
    if not np.all(np.isfinite(y)):
        raise ValueError("Target contains NaN or infinite values.")
    if family == "poisson" and np.any(y < 0):
        raise ValueError("Poisson target must be non-negative.")
    if family in ("gamma", "inverse.gaussian") and np.any(y <= 0):
        raise ValueError(f"{family} target must be strictly positive.")
    if family == "binomial" and (np.any(y < 0) or np.any(y > 1)):
        raise ValueError("Binomial target must lie in [0, 1].")


@dataclass
class GLMFit:
    """A fitted penalised GLM together with the spec that built its features."""

    spec: DesignSpec
    model: GeneralizedLinearRegressor | GeneralizedLinearRegressorCV
    family: str
    link: str
    target: str
    weight_col: str | None = None
    offset_col: str | None = None
    divide_target_by_weight: bool = False
    monotone: dict[str, Direction] = field(default_factory=dict)
    #: index (into the variable's table rows) of the most exposed bin/level in
    #: the training data; used as the default base risk for rate tables.
    modal_bins: dict[str, int] = field(default_factory=dict)
    n_train_rows: int = 0

    # -- coefficients -----------------------------------------------------
    @property
    def intercept(self) -> float:
        return float(self.model.intercept_)

    @property
    def coef(self) -> np.ndarray:
        return np.asarray(self.model.coef_, dtype=float)

    @property
    def alpha(self) -> float:
        value = getattr(self.model, "alpha_", None)
        if value is None:
            value = getattr(self.model, "alpha", None)
        return float("nan") if value is None else float(value)

    @property
    def feature_names(self) -> list[str]:
        return self.spec.feature_names

    def coef_table(self, *, drop_zero: bool = False) -> pl.DataFrame:
        """One row per coefficient with structured feature metadata."""
        rows = [("(intercept)", None, "intercept", None, None, self.intercept)]
        for f, c in zip(self.spec.features, self.coef, strict=True):
            rows.append((f.name, f.variable, f.kind, f.knot, f.level, float(c)))
        out = pl.DataFrame(
            rows,
            schema={
                "feature": pl.Utf8,
                "variable": pl.Utf8,
                "kind": pl.Utf8,
                "knot": pl.Float64,
                "level": pl.Utf8,
                "coef": pl.Float64,
            },
            orient="row",
        ).with_columns(pl.col("coef").exp().alias("exp_coef"))
        if drop_zero:
            out = out.filter((pl.col("coef") != 0) | (pl.col("kind") == "intercept"))
        return out

    # -- prediction -------------------------------------------------------
    def design_matrix(self, data: pl.DataFrame, *, sparse: bool | None = None):
        """The design matrix for ``data`` (see :meth:`DesignSpec.build`).

        Nothing in scoring needs this — :meth:`linear_predictor` and
        :meth:`predict` work from the codes — so it is here for inspection and
        for callers that want the columns themselves.
        """
        return self.spec.build(data, sparse=sparse)

    def _link_inverse(self, lp: np.ndarray) -> np.ndarray:
        """The mean for a linear predictor, using glum's own link object so the
        answer is identical to ``model.predict``."""
        link = getattr(self.model, "_link_instance", None)
        if link is None:  # pragma: no cover - only before a fit
            link = getattr(self.model, "link_instance", None)
        if link is not None:
            return np.asarray(link.inverse(lp), dtype=float)
        if self.link == "log":  # pragma: no cover - defensive
            return np.exp(lp)
        raise RuntimeError(f"Cannot invert the {self.link!r} link without glum")

    def linear_predictor(
        self,
        data: pl.DataFrame,
        *,
        offset: np.ndarray | None = None,
        chunk_rows: int = SCORING_CHUNK_ROWS,
    ) -> np.ndarray:
        """``intercept + design @ coef`` (+ ``offset``), computed from the
        integer codes in row chunks — never from a design matrix.

        The arithmetic is one float64 table lookup per variable per chunk, the
        same thing :class:`~easy_glm.engine.rate_model.RateModel` does, so a
        book of any size costs one float64 vector rather than a second copy of
        the design.
        """
        lp = self.spec.linear_predictor(
            data, self.coef, self.intercept, chunk_rows=chunk_rows
        )
        if offset is not None:
            lp = lp + np.asarray(offset, dtype=float)
        return lp

    def scoring_offset(
        self, data: pl.DataFrame, offset: np.ndarray | None = None
    ) -> np.ndarray | None:
        """The offset to score ``data`` with: the one passed in, else the stored
        offset column read off ``data`` (with a warning if it is not there)."""
        if offset is not None or not self.offset_col:
            return offset
        if self.offset_col in data.columns:
            return data[self.offset_col].cast(pl.Float64).to_numpy()
        warnings.warn(
            f"Offset column '{self.offset_col}' not found in data — "
            "predictions exclude the offset",
            stacklevel=3,
        )
        return None

    def predict(
        self, data: pl.DataFrame, *, offset: np.ndarray | None = None
    ) -> np.ndarray:
        """Predictions on the response scale (per unit of weight if the target
        was divided by the weight). Uses the stored offset column if present.

        Computed by :meth:`linear_predictor` (rate-table lookups, chunked) and
        glum's own link, **not** by ``model.predict(X)``: no design matrix is
        built, so scoring a book costs the same whether it has fifty thousand
        rows or five million.
        """
        offset = self.scoring_offset(data, offset)
        return self._link_inverse(self.linear_predictor(data, offset=offset))

    def __repr__(self) -> str:
        nnz = int((self.coef != 0).sum())
        return (
            f"GLMFit(family={self.family!r}, link={self.link!r}, target="
            f"{self.target!r}, alpha={self.alpha:.4g}, features={len(self.coef)}, "
            f"non_zero={nnz}, variables={self.spec.variables})"
        )


#: A one-band ("continuous") term bases at the **upper** clamp only when the
#: exposure-weighted median is past this fraction of the range; below it the
#: lower clamp wins. Off centre on purpose: at 0.5 a factor whose median sits
#: near the middle would flip between clamps on sampling noise.
CONTINUOUS_BASE_AT_HI = 0.6


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Exposure-weighted median of ``values``, ignoring NaN. NaN if all null."""
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(x) & (w > 0)
    x, w = x[ok], w[ok]
    if x.size == 0:
        return float("nan")
    order = np.argsort(x, kind="stable")
    x, w = x[order], np.cumsum(w[order])
    return float(x[min(int(np.searchsorted(w, 0.5 * w[-1])), x.size - 1)])


def _continuous_base_row(enc: LinearEncoder, x: np.ndarray, w: np.ndarray) -> int:
    """Base row of a **one-band** (``kind="continuous"``) linear term.

    Q2 puts relativity 1.00 where the exposure is, which for a multi-band term
    is the lower edge of the most exposed band. A continuous term has a single
    band, so that rule would always give the lower clamp — the bottom of the
    range, where hardly any business is. The 1.00 point must still be an edge of
    a table row (that is how the rate table, Excel and ``from_rate_tables``
    carry it), and a one-band term has exactly two: the lower clamp and the
    upper one. So: **1.00 sits at the lower clamp unless the bulk of the business
    is well up the range** — the exposure-weighted median past
    :data:`CONTINUOUS_BASE_AT_HI` of the way from ``lo`` to ``hi``. Row 1 is the
    band (base at ``lo``), row 2 the ``(hi, None)`` row (base at ``hi``).

    The threshold is deliberately off centre. At the midpoint a factor whose
    median sits near the middle of its range would flip between the two clamps
    on sampling noise — a refit on next month's data would move every relativity
    and the base rate, for no reason anyone could explain. 0.6 makes the lower
    clamp the default answer and reserves the upper one for factors that are
    plainly top heavy. Because a continuous curve has one slope, the choice only
    rescales the base rate: the ratios between relativities are unchanged, and
    ``base="reference"`` or a base-rate override put 1.00 anywhere else.
    """
    median = weighted_median(np.clip(x, enc.lo, enc.hi), w)
    if not np.isfinite(median):
        return 1
    cut = enc.lo + CONTINUOUS_BASE_AT_HI * (enc.hi - enc.lo)
    return 2 if median > cut else 1


def _modal_bins(
    spec: DesignSpec, data: pl.DataFrame, weights: np.ndarray | None
) -> dict[str, int]:
    """Index of the most exposed table row per main effect (see ``rate_tables``).
    Uses the encoders' shared ``row_index`` rule; interactions have no base row.
    One-band linear terms follow :func:`_continuous_base_row`."""
    w = np.ones(data.height) if weights is None else np.asarray(weights, dtype=float)
    out: dict[str, int] = {}
    for var, enc in spec.encoders.items():
        if isinstance(enc, InteractionEncoder):
            continue
        if isinstance(enc, LinearEncoder) and enc.n_bands == 1:
            x = data[var].cast(pl.Float64).to_numpy()
            out[var] = _continuous_base_row(enc, x, w)
            continue
        idx = enc.row_index(data[var])
        counts = np.bincount(idx, weights=w, minlength=enc.n_rows)
        if isinstance(enc, LinearEncoder) and len(counts) > 1:
            # the base of a linear term is a point on the curve (x_base), so the
            # null row is never the base even when it carries the most exposure
            counts[-1] = -1.0
        out[var] = int(np.argmax(counts))
    return out


def penalty_weights(
    spec: DesignSpec,
    design: Any,
    weights: np.ndarray | None,
    *,
    scale_predictors: bool,
) -> np.ndarray | None:
    """Per-column L1 weights (glum ``P1``) for piecewise-linear bands and
    interaction cells.

    Both need one for the same reason: with ``scale_predictors=True`` glum
    penalises the *standardised* coefficient ``alpha * P1_j * |beta_j| * sd_j``,
    so a column with little spread buys a large **effect** for a small penalty —
    the opposite of what a pricing model wants.

    **Piecewise-linear bands.** The effect an actuary reads off a band is its
    *rise*: the change in log relativity from one end of the band to the other,
    ``beta_j * width_j``. Unweighted, one unit of rise costs
    ``alpha * sd_j / width_j``, which collapses in a wide band that few rows
    reach: on the French motor set the top bonus-malus band ``[95, 230)`` bought
    its rise for **4 %** of what the first band paid, so the thin tail — the part
    of a curve an actuary trusts least — was the *least* penalised part of it.
    Writing ``u_j = column_j / width_j`` (the share of band ``j`` a row has
    used, in ``[0, 1]``), ``P1_j = 0.5 / sd(u_j)`` makes one unit of rise cost
    ``0.5 * alpha`` in **every** band. The 0.5 is the same normalisation the
    cells use: a band that half the exposure has passed has ``sd(u) = 0.5`` and
    so ``P1 = 1``, exactly like a 50/50 step column. Note what this also does at
    the ends of the range: the first and last bands of a wide design are nearly
    constant columns (almost every row is past the first band, almost none
    reaches the last), so they are penalised hardest — a rise placed there is
    close to a shift of the whole curve and now has to pay for itself.

    **This raises a linear term's penalty as well as levelling it.** ``u_j`` lives
    in ``[0, 1]``, so ``sd(u_j) <= 0.5`` and ``P1_j >= 1`` always: no band is
    penalised less than before and most are penalised more. Measured on the
    French motor set, the mean weight over a term's bands is **1.6x** (Density,
    20 bands) to **4.1x** (BonusMalus, 9 bands), and a **one-band (continuous)
    term, which has nothing to level, is simply penalised 3.6x (Density) to 6.3x
    (BonusMalus) harder** than before. That is what turns a weak continuous trend
    into a flat term: the penalty is now on the rise, and a rise of a few per
    cent across the whole range does not pay for itself. It is a real change in
    how strong a given ``alpha`` is on these terms, not a redistribution.

    Without standardisation the raw coefficient is the *slope*, so a wide band
    is still cheap per unit of rise; ``P1_j = width_j * n_bands / (hi - lo)``
    restores equality there. That form **is** a pure redistribution: the weights
    average to 1 over the term's bands, so the overall strength of ``alpha`` on
    the term is unchanged.

    **Interaction cells** get ``P1 = penalty_weight * 0.5 / sd`` under
    standardisation (thin cells shrunk harder, fat cells like the mains) and
    ``penalty_weight * 0.5`` without it. The two are the *same* penalty: glum
    multiplies a standardised column's ``P1`` by that column's ``sd``, so
    ``0.5 / sd`` times ``sd`` is ``0.5`` — one unit of log adjustment costs
    ``0.5 * alpha * penalty_weight`` in every cell either way, which is what a
    50/50 step column (``sd = 0.5``, ``P1 = 1``) costs in the mains. That
    equality is what lets the **second stage** of an interaction fit
    (:func:`fit_two_stage`) penalise cells exactly as a joint fit did: glum
    refuses to standardise without an intercept, and stage 2 has no intercept.

    The columns themselves are never rescaled, so ``beta_j`` stays band ``j``'s
    slope and the rate table reads it off the coefficients unchanged.

    ``design`` may be the dense float64 array or the compact tabmat
    ``SplitMatrix`` :meth:`DesignSpec.build` returns; the standard deviations
    come from the matrix's own weighted-column-statistics method in the second
    case, so no column is ever expanded. The two differ only by floating-point
    noise (they sum the same numbers in a different order), which is why the
    two paths' fitted coefficients agree to 1e-10 rather than exactly.

    Returns ``None`` when the spec has neither linear terms nor interactions
    (glum's default applies).
    """
    linears = [(v, e) for v, e in spec.encoders.items() if isinstance(e, LinearEncoder)]
    if not linears and not spec.interactions:
        return None
    p1 = np.ones(spec.n_features)
    w = np.ones(design.shape[0]) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    slices = spec.slices()
    cached_stds: list[np.ndarray] = []

    def _sd(idx: np.ndarray | slice, scale: np.ndarray | None = None) -> np.ndarray:
        """Weighted sd of the design columns ``idx``, each divided by ``scale``."""
        if isinstance(design, np.ndarray):
            cols = design[:, idx]
            if scale is not None:
                cols = cols / scale
            mean = w @ cols
            var = w @ (cols**2) - mean**2
            sd = np.sqrt(np.clip(var, 0.0, None))
        else:
            if not cached_stds:
                cached_stds.append(
                    np.asarray(
                        design._get_col_stds(w, design.transpose_matvec(w)), dtype=float
                    )
                )
            sd = cached_stds[0][idx]
            if scale is not None:
                sd = sd / scale
        # a constant column has no spread to standardise by; glum leaves it
        # alone, and 0.5 is the weight of a column half the exposure shares
        return np.where(sd > 0, sd, 0.5)

    for var, enc in linears:
        start = slices[var].start
        idx = np.arange(start, start + enc.n_bands)
        widths = np.asarray(enc.band_widths(), dtype=float)
        if scale_predictors:
            p1[idx] = 0.5 / _sd(idx, widths)
        else:
            p1[idx] = widths * enc.n_bands / (enc.hi - enc.lo)
    for enc in spec.interactions:
        sl = slices[enc.variable]
        if scale_predictors:
            p1[sl] = enc.penalty_weight * 0.5 / _sd(sl)
        else:
            p1[sl] = enc.penalty_weight * 0.5
    return p1


# --------------------------------------------------------------------------
# progress
# --------------------------------------------------------------------------
#: Seconds between progress messages while a fit is running.
PROGRESS_INTERVAL_SECONDS = 1.0

Progress = Callable[[str], None]


class _ElapsedProgress:
    """Report elapsed time while a fit runs, from a background thread.

    glum 3.4.1 offers no hook: its ``verbose`` flag prints to the console and
    there is no callback per alpha on the path or per cross-validation fold, so
    there is no honest fraction to show. What *can* be shown is what stage the
    fit is in and how long it has been there, which is what a long fit's
    watcher actually wants. A daemon thread ticks every
    :data:`PROGRESS_INTERVAL_SECONDS`; the callback is called from that thread,
    so a caller that draws something must be ready for that (the workbench
    attaches Streamlit's script context to it).

    Any exception the callback raises is swallowed: a progress display must
    never be able to fail a fit.
    """

    def __init__(self, progress: Progress | None, label: str) -> None:
        self._progress = progress
        self._label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start = 0.0

    def _emit(self, seconds: float) -> None:
        if self._progress is None:
            return
        try:
            self._progress(f"{self._label} — {seconds:.0f}s")
        except Exception:  # pragma: no cover - a display must not fail a fit
            pass

    def _loop(self) -> None:
        while not self._stop.wait(PROGRESS_INTERVAL_SECONDS):
            self._emit(time.monotonic() - self._start)

    def __enter__(self) -> _ElapsedProgress:
        if self._progress is None:
            return self
        self._start = time.monotonic()
        self._emit(0.0)
        self._thread = threading.Thread(
            target=self._loop, name="easy_glm-progress", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=PROGRESS_INTERVAL_SECONDS * 2)
        if exc[0] is None:
            self._emit(time.monotonic() - self._start)


# --------------------------------------------------------------------------
# aggregation by identical design row
# --------------------------------------------------------------------------
def design_row_key(
    spec: DesignSpec, data: pl.DataFrame, offset: np.ndarray | None = None
) -> pl.DataFrame:
    """The columns that decide a row's design row (and so its prediction).

    The integer code of every variable, plus — for a piecewise-linear term,
    whose columns move *inside* a band — the clamped value itself, plus the
    offset when there is one. Two rows agreeing on all of these have identical
    design rows and identical offsets.
    """
    columns: dict[str, pl.Series] = {}
    for var, enc in spec.encoders.items():
        columns[f"c:{var}"] = pl.Series(f"c:{var}", enc.codes(data))
        if isinstance(enc, LinearEncoder):
            x = data[var].cast(pl.Float64).to_numpy()
            columns[f"x:{var}"] = pl.Series(
                f"x:{var}", np.clip(np.where(np.isnan(x), enc.lo, x), enc.lo, enc.hi)
            )
    if offset is not None:
        columns["o:"] = pl.Series("o:", np.asarray(offset, dtype=float))
    return pl.DataFrame(columns)


def aggregate_rows(
    spec: DesignSpec,
    data: pl.DataFrame,
    y: np.ndarray,
    weights: np.ndarray | None,
    offset: np.ndarray | None,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray | None]:
    """Group rows with identical design rows; return one row per group.

    Returns ``(rows, y_bar, weight_sum, offset)`` where ``rows`` is a frame of
    representative raw rows (one per group, so the design built from it is the
    group's design row), ``weight_sum`` is the group's total weight and
    ``y_bar`` its weighted mean target.

    **Why this is exact.** For every exponential-dispersion family the part of
    the deviance that depends on the coefficients is linear in ``y``, so rows
    sharing a design row (and an offset) contribute exactly
    ``W_g * (-ybar_g * theta(mu_g) + b(theta(mu_g)))``. Objective, gradient and
    Hessian are unchanged, the weighted column means and standard deviations
    glum standardises by are unchanged, and the weights still sum to the same
    total so ``alpha`` means the same thing. What is *not* preserved is the
    deviance constant (the ``y log y`` terms) and anything per row.
    """
    n = data.height
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    keys = design_row_key(spec, data, offset)
    key_names = keys.columns
    frame = keys.with_columns(
        pl.Series("_w", w),
        pl.Series("_wy", w * np.asarray(y, dtype=float)),
        pl.int_range(pl.len(), dtype=pl.UInt32).alias("_row"),
    )
    grouped = frame.group_by(key_names, maintain_order=False).agg(
        pl.col("_w").sum(), pl.col("_wy").sum(), pl.col("_row").first()
    )
    rep = grouped["_row"].to_numpy()
    weight_sum = grouped["_w"].to_numpy().astype(float)
    y_bar = grouped["_wy"].to_numpy().astype(float) / weight_sum
    rows = data[rep]
    return rows, y_bar, weight_sum, None if offset is None else offset[rep]


def fit_glm(
    data: pl.DataFrame,
    spec: DesignSpec,
    target: str,
    *,
    family: Any = "poisson",
    weight_col: str | None = None,
    offset_col: str | None = None,
    divide_target_by_weight: bool = False,
    alpha: float | None = None,
    l1_ratio: float | list[float] = 1.0,
    cv: int | None = None,
    n_alphas: int = 20,
    min_alpha_ratio: float | None = None,
    monotone: Mapping[str, Direction] | None = None,
    scale_predictors: bool = True,
    offset: np.ndarray | None = None,
    fit_intercept: bool = True,
    sparse: bool | None = None,
    aggregate: bool = False,
    progress: Progress | None = None,
    **glum_kwargs: Any,
) -> GLMFit:
    """Fit an L1/elastic-net GLM on ``spec.build(data)``.

    Parameters
    ----------
    data : pl.DataFrame
        Training rows only (filter your holdout out first).
    spec : DesignSpec
        Feature definitions, normally ``DesignSpec.from_data(train, predictors)``.
    target : str
        Response column.
    family : str or glum distribution
        ``"poisson"``, ``"gamma"``, ``"gaussian"``, ``"binomial"``,
        ``"tweedie"`` (power 1.5) or e.g. ``TweedieDistribution(1.7)``. The link
        is log (logit for binomial) unless ``link=`` is passed in ``glum_kwargs``.
    weight_col, offset_col : str, optional
        Sample weights (exposure or premium) and an offset already on the
        linear-predictor scale (e.g. ``log(exposure)``).
    divide_target_by_weight : bool
        Model ``target / weight`` (e.g. claim frequency from counts + exposure).
    alpha : float, optional
        Penalty strength. Required unless ``cv`` is given.
    l1_ratio : float or list[float]
        1 = lasso (sparse knots; the AGLM default), 0 = ridge. A list is only
        meaningful with ``cv``.
    cv : int, optional
        Number of folds; the alpha (and l1_ratio) minimising CV deviance over a
        ``n_alphas``-point path is chosen. Overrides ``alpha``.
    monotone : {variable: "increasing" | "decreasing"}, optional
        Sign constraints on step increments / piecewise-linear band slopes
        (see :func:`monotone_bounds`).
    scale_predictors : bool
        Standardise columns before penalising (glmnet/aglm default). glum
        cannot do this without an intercept, so ``fit_intercept=False``
        requires ``scale_predictors=False``.
    offset : np.ndarray, optional
        The whole offset of *this* fit, one value per row on the
        linear-predictor scale, instead of ``offset_col``. Passing both is an
        error. This is how the second stage of an interaction fit receives the
        first stage's linear predictor (see :func:`fit_two_stage`).
    fit_intercept : bool
        ``False`` fits no intercept — the second stage of an interaction fit,
        where the level already sits in the offset.
    sparse : bool, optional
        Force the compact (``True``) or the dense (``False``) design matrix.
        The default decides by row count (:data:`~easy_glm.core.design.
        SPARSE_ROW_THRESHOLD`, 200,000 rows). Both give the same fit; the
        compact one holds an integer per row per variable instead of the
        columns, which is what lets a 5M-row book fit in memory.
    aggregate : bool
        Fit **one row per distinct design row**, carrying the summed weight and
        the weighted mean target. This is exact for every family easy_glm
        offers — the objective, gradient and Hessian are unchanged, and so are
        the coefficients (to 1e-12) — because the part of the deviance that
        depends on the coefficients is linear in the target. Whether it is
        *worth* anything depends entirely on the design: a coarse one (few
        knots, few levels, no continuous term) can collapse a book several
        fold, a fine one barely at all (1.5x on the French motor set), and the
        grouping itself costs a pass over the data. Off by default, and
        refused with ``cv`` (folds must be assigned to rows, not groups) and
        when the fit has a piecewise-linear term with many distinct values.
        Nothing downstream changes: rate tables, predictions and diagnostics
        are still per row.
    progress : callable, optional
        Called with a short status string (``"Fitting 1,000,000 rows x 197
        columns — 12s"``) about once a second while the fit runs, from a
        background thread. glum exposes no per-alpha or per-fold hook, so what
        is reported is the stage and the elapsed time, not a fraction. Any
        exception the callback raises is swallowed.
    glum_kwargs
        Anything else for the glum estimator (``max_iter``, ``P1``, ``link``,
        ``lower_bounds`` ...). Passing your own ``P1`` replaces the per-cell
        per-band / per-cell penalty rule entirely (see ``penalty_weights``).
    """
    if cv is None and alpha is None:
        raise ValueError(
            "Pass alpha=<penalty strength> or cv=<n_folds>. (Without either, "
            "glum would silently return the least-regularised end of its path.)"
        )
    if target not in data.columns:
        raise KeyError(f"Target column {target!r} not found in data")
    if data.is_empty():
        raise ValueError("No training rows.")

    if aggregate and cv is not None:
        raise ValueError(
            "aggregate=True cannot be combined with cv=: cross-validation folds "
            "have to be assigned to rows, and aggregation replaces the rows. "
            "Choose an alpha (or cross-validate once without aggregation and "
            "refit with that alpha)."
        )

    fam, family_name, default_link = resolve_family(family)
    link = glum_kwargs.pop("link", default_link)

    y = data[target].cast(pl.Float64).to_numpy()
    sw = None
    if weight_col:
        if weight_col not in data.columns:
            raise KeyError(f"Weight column {weight_col!r} not found in data")
        sw = data[weight_col].cast(pl.Float64).to_numpy()
        if not np.all(np.isfinite(sw)) or np.any(sw <= 0):
            raise ValueError("Weights must be finite and strictly positive.")
        if divide_target_by_weight:
            y = y / sw
    elif divide_target_by_weight:
        raise ValueError("divide_target_by_weight=True needs weight_col.")
    if offset is not None:
        if offset_col:
            raise ValueError(
                "Pass offset_col=<column> or offset=<array>, not both: the array "
                "is used as the whole offset of the fit."
            )
        offset = np.asarray(offset, dtype=float)
        if offset.shape != (data.height,):
            raise ValueError(
                f"offset must have one value per training row ({data.height}); "
                f"got shape {offset.shape}"
            )
        if not np.all(np.isfinite(offset)):
            raise ValueError("offset contains NaN or infinite values.")
    elif offset_col:
        if offset_col not in data.columns:
            raise KeyError(f"Offset column {offset_col!r} not found in data")
        offset = data[offset_col].cast(pl.Float64).to_numpy()
    if not fit_intercept and scale_predictors:
        raise ValueError(
            "fit_intercept=False needs scale_predictors=False: glum cannot "
            "standardise columns without an intercept. (The cell penalty is "
            "the same either way — see penalty_weights.)"
        )
    _validate_target(y, family_name)

    modal_bins = _modal_bins(spec, data, sw)
    n_train_rows = data.height
    fit_rows = data
    if aggregate:
        fit_rows, y, sw, offset = aggregate_rows(spec, data, y, sw, offset)
    design = spec.build(fit_rows, sparse=sparse)
    if not isinstance(design, np.ndarray):
        # glum validates its input with a private function that only knows
        # tabmat's own block types; teach it about ours (see stepmatrix.py)
        install_glum_shim()

    if "P1" not in glum_kwargs:
        p1 = penalty_weights(spec, design, sw, scale_predictors=scale_predictors)
        if p1 is not None:
            glum_kwargs["P1"] = np.asarray(p1, dtype=np.float64)

    lower = glum_kwargs.pop("lower_bounds", None)
    upper = glum_kwargs.pop("upper_bounds", None)
    monotone = dict(monotone or {})
    if monotone:
        mlo, mup = monotone_bounds(spec, monotone)
        lower = mlo if lower is None else np.maximum(np.asarray(lower, float), mlo)
        upper = mup if upper is None else np.minimum(np.asarray(upper, float), mup)
    # float64 for everything the solver touches: a float32 design silently
    # stops converging past a million rows and segfaults tabmat when the
    # sample weight is not cast to match (docs/spikes/g-scale, §4.2)
    if lower is not None:
        lower = np.asarray(lower, dtype=np.float64)
    if upper is not None:
        upper = np.asarray(upper, dtype=np.float64)
    if sw is not None:
        sw = np.asarray(sw, dtype=np.float64)
    if offset is not None:
        offset = np.asarray(offset, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    common: dict[str, Any] = dict(
        family=fam,
        link=link,
        l1_ratio=l1_ratio,
        fit_intercept=fit_intercept,
        scale_predictors=scale_predictors,
        lower_bounds=lower,
        upper_bounds=upper,
        **glum_kwargs,
    )
    model: GeneralizedLinearRegressor | GeneralizedLinearRegressorCV
    if cv is not None:
        if alpha is not None:
            warnings.warn("alpha is ignored when cv is set", stacklevel=2)
        model = GeneralizedLinearRegressorCV(
            cv=cv, n_alphas=n_alphas, min_alpha_ratio=min_alpha_ratio, **common
        )
    else:
        if isinstance(l1_ratio, list | tuple):
            raise ValueError("A list of l1_ratio values requires cv=...")
        model = GeneralizedLinearRegressor(alpha=alpha, **common)

    shape = f"{design.shape[0]:,} rows x {design.shape[1]:,} columns"
    label = (
        f"Cross-validating {cv} folds over {n_alphas} penalties, {shape}"
        if cv is not None
        else f"Fitting {shape}"
    )
    with _ElapsedProgress(progress, label):
        model.fit(design, y, sample_weight=sw, offset=offset)

    return GLMFit(
        spec=spec,
        model=model,
        family=family_name,
        link=link,
        target=target,
        weight_col=weight_col,
        offset_col=offset_col,
        divide_target_by_weight=divide_target_by_weight,
        monotone=monotone,
        modal_bins=modal_bins,
        n_train_rows=n_train_rows,
    )


# --------------------------------------------------------------------------
# two stages: main effects, then interaction cells on top of them
# --------------------------------------------------------------------------
class TwoStageFit(GLMFit):
    """Main effects and interaction cells fitted in **two stages**.

    The actuary's answer to Q5: when an interaction is added, the main-effect
    tables must not move. So the mains are fitted first, exactly as a model
    without interactions fits them, and the interaction cells are then fitted
    **on top** of that fit — no intercept, and stage 1's linear predictor (plus
    the user's offset column, if any) as the offset. Every cell coefficient is
    therefore a pure adjustment to a frozen main-effect model, and 1.00 means
    "no adjustment".

    It *is* a :class:`GLMFit` — same ``spec`` (mains then cells, the column
    order a joint fit would have used), same ``coef`` (stage 1's coefficients
    followed by stage 2's), same ``intercept``, ``modal_bins``, ``coef_table``,
    ``design_matrix`` and ``linear_predictor``. So ``rate_tables``,
    ``base_rate``, ``to_rate_model`` and the diagnostics need no special case:
    the main tables and the base rate read stage-1 numbers, the cells read
    stage-2 numbers, and ``linear_predictor`` is ``eta1 + eta2`` because the two
    coefficient blocks sit on disjoint columns.

    ``alpha`` is stage 1's penalty and :attr:`alpha_stage2` is stage 2's.
    """

    def __init__(self, stage1: GLMFit, stage2: GLMFit) -> None:
        if isinstance(stage1, TwoStageFit) or isinstance(stage2, TwoStageFit):
            raise ValueError("A TwoStageFit cannot be a stage of another one")
        if stage1.spec.interactions:
            raise ValueError(
                "Stage 1 must hold the main effects only; its spec has "
                f"{[e.variable for e in stage1.spec.interactions]}"
            )
        if not stage2.spec.encoders:
            raise ValueError("Stage 2 has no interaction to fit")
        for var, enc in stage2.spec.encoders.items():
            if not isinstance(enc, InteractionEncoder):
                raise ValueError(
                    f"Stage 2 may only hold interactions; {var!r} is a "
                    f"{type(enc).__name__}"
                )
            for parent in (enc.a, enc.b):
                if stage1.spec.encoders.get(parent.variable) is not parent:
                    raise ValueError(
                        f"Interaction {var!r}: parent {parent.variable!r} is not "
                        "(the same encoder as) a main effect of stage 1"
                    )
        if float(stage2.model.intercept_) != 0.0:
            raise ValueError(
                "Stage 2 must be fitted with fit_intercept=False; its intercept "
                f"is {float(stage2.model.intercept_):g}, which would move the "
                "base rate the mains fixed"
            )
        for field_name in ("family", "link", "target", "weight_col"):
            if getattr(stage1, field_name) != getattr(stage2, field_name):
                raise ValueError(
                    f"The two stages disagree on {field_name}: "
                    f"{getattr(stage1, field_name)!r} vs "
                    f"{getattr(stage2, field_name)!r}"
                )
        super().__init__(
            spec=DesignSpec({**stage1.spec.encoders, **stage2.spec.encoders}),
            model=stage1.model,
            family=stage1.family,
            link=stage1.link,
            target=stage1.target,
            weight_col=stage1.weight_col,
            offset_col=stage1.offset_col,
            divide_target_by_weight=stage1.divide_target_by_weight,
            monotone=dict(stage1.monotone),
            modal_bins=dict(stage1.modal_bins),
            n_train_rows=stage1.n_train_rows,
        )
        self.stage1 = stage1
        self.stage2 = stage2

    # -- coefficients -----------------------------------------------------
    @property
    def coef(self) -> np.ndarray:
        """Stage 1's coefficients followed by stage 2's — one entry per column
        of :attr:`spec`, in the spec's own order."""
        return np.concatenate([self.stage1.coef, self.stage2.coef])

    @property
    def alpha_stage2(self) -> float:
        """The penalty strength the cells were fitted at."""
        return self.stage2.alpha

    # -- prediction -------------------------------------------------------
    def predict(
        self, data: pl.DataFrame, *, offset: np.ndarray | None = None
    ) -> np.ndarray:
        """``link_inverse(eta1 + eta2 + offset)`` — the two stages composed.

        No special case is needed: the composed fit's spec is the mains
        followed by the cells and its coefficients are stage 1's followed by
        stage 2's, on disjoint columns, so the inherited rate-table scoring
        adds up ``eta1 + eta2`` by itself.
        """
        return super().predict(data, offset=offset)

    def __eq__(self, other: object) -> bool:
        """Both stages, not just the fields :class:`GLMFit` declares — ``stage1``
        and ``stage2`` are ordinary attributes, so the inherited dataclass
        equality would call two fits with the same mains and different cells
        equal."""
        if not isinstance(other, TwoStageFit):
            return NotImplemented
        return (self.stage1, self.stage2) == (other.stage1, other.stage2)

    def __repr__(self) -> str:
        nnz = int((self.coef != 0).sum())
        return (
            f"TwoStageFit(family={self.family!r}, link={self.link!r}, target="
            f"{self.target!r}, alpha={self.alpha:.4g}, alpha_stage2="
            f"{self.alpha_stage2:.4g}, features={len(self.coef)}, non_zero={nnz}, "
            f"mains={self.stage1.spec.variables}, "
            f"cells={self.stage2.spec.variables})"
        )


def fit_two_stage(
    data: pl.DataFrame,
    spec: DesignSpec,
    target: str,
    *,
    stage2_alpha: float | None = None,
    **kwargs: Any,
) -> GLMFit:
    """Fit ``spec`` in two stages and return the composed :class:`TwoStageFit`.

    ``kwargs`` are :func:`fit_glm`'s and describe **stage 1** — the main-effect
    fit, which is the fit the same model without any interaction would produce
    (to glum's own run-to-run noise, ~1e-15 on a coefficient). Stage 2 then fits
    the interaction cells alone with ``fit_intercept=False`` and
    ``offset = eta1``, where ``eta1`` is stage 1's whole linear predictor on the
    training rows — the user's offset included, whether it came from
    ``offset_col`` or from an ``offset`` array — keeping the family, link,
    weights and ``divide_target_by_weight`` of stage 1; ``monotone`` does not
    apply to cells and is dropped, and ``fit_intercept``, ``scale_predictors``
    and any ``P1`` of stage 1 do not carry over (stage 2 has its own columns).

    An ``offset`` **array** is a training-time offset only, exactly as in
    :func:`fit_glm`: nothing about it can be stored for scoring, so pass it to
    ``predict`` as well (and note that a ``RateModel`` compiled from such a fit
    cannot apply it — use ``offset_col`` for a model you intend to score).

    The second stage's penalty:

    * ``stage2_alpha`` if given (an explicit strength for the cells, which
      overrides cross-validation for this stage);
    * otherwise the alpha stage 1 used — the cross-validated one if stage 1
      cross-validated *and* ``cv`` is not repeated below;
    * if stage 1 used ``cv``, so does stage 2, on its own path over its own
      columns.

    Per-interaction differences in strength belong in
    ``InteractionEncoder.penalty_weight``, which multiplies that interaction's
    cells' ``P1``; stage 2 is one fit and therefore has one alpha.

    When no cell of any interaction has enough exposure to be rated on its own
    there is nothing for a second stage to fit: a plain :class:`GLMFit` on
    ``spec`` comes back (the same numbers a mains-only fit gives, since the cell
    block has no columns) and every cell reads 1.00. That is why the return type
    is ``GLMFit``: narrow with ``isinstance(fit, TwoStageFit)`` before reading
    :attr:`TwoStageFit.alpha_stage2` or :attr:`TwoStageFit.stage2`.
    """
    if not spec.interactions:
        raise ValueError(
            "fit_two_stage needs a spec with at least one interaction; use "
            "fit_glm for a mains-only design"
        )
    if spec.interactions_spec().n_features == 0:
        # no cell had enough exposure to be rated on its own, so there is
        # nothing for a second stage to fit and the design *is* the main-effect
        # design; a plain GLMFit comes back and every cell reads 1.00
        return fit_glm(data, spec, target, **kwargs)

    outer_progress: Progress | None = kwargs.get("progress")

    def _stage_progress(name: str) -> Progress | None:
        """Prefix the caller's progress messages with the stage they belong to."""
        if outer_progress is None:
            return None
        return lambda message: outer_progress(f"{name} — {message}")

    kwargs = {**kwargs, "progress": _stage_progress("Stage 1, main effects")}
    fit1 = fit_glm(data, spec.main_effects_spec(), target, **kwargs)
    # eta1 must be the *whole* of stage 1's linear predictor, the user's offset
    # included, or stage 2 would see a residual that still contains the offset
    # and would put it in the cells. linear_predictor() adds neither form of
    # offset, so both are added here.
    eta1 = fit1.linear_predictor(data)
    if fit1.offset_col:
        eta1 = eta1 + data[fit1.offset_col].cast(pl.Float64).to_numpy()
    user_offset = kwargs.get("offset")
    if user_offset is not None:
        eta1 = eta1 + np.asarray(user_offset, dtype=float)

    dropped = {
        "monotone",
        "offset_col",
        "offset",  # already inside eta1
        "alpha",
        "cv",
        "fit_intercept",
        "P1",  # stage 1's, one weight per main-effect column
        "progress",  # re-labelled per stage below
    }
    kw2 = {k: v for k, v in kwargs.items() if k not in dropped}
    kw2["progress"] = _stage_progress("Stage 2, interaction cells")
    kw2["scale_predictors"] = False  # glum refuses to standardise with no intercept
    if stage2_alpha is not None:
        kw2["alpha"] = float(stage2_alpha)
    elif kwargs.get("cv") is not None and kwargs.get("alpha") is None:
        kw2["cv"] = kwargs["cv"]
    else:
        kw2["alpha"] = fit1.alpha
    if "alpha" in kw2 and isinstance(kw2.get("l1_ratio"), list | tuple):
        # a list of l1_ratios is only meaningful with cv; take the one chosen
        kw2["l1_ratio"] = float(getattr(fit1.model, "l1_ratio_", 1.0))
    fit2 = fit_glm(
        data,
        spec.interactions_spec(),
        target,
        offset=eta1,
        fit_intercept=False,
        **kw2,
    )
    return TwoStageFit(fit1, fit2)
