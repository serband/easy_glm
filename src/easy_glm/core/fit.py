"""Fit a penalised GLM on the design matrix of a :class:`DesignSpec`."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from glum import (
    GeneralizedLinearRegressor,
    GeneralizedLinearRegressorCV,
    TweedieDistribution,
)

from .design import DesignSpec, InteractionEncoder, LinearEncoder, StepEncoder

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
    def design_matrix(self, data: pl.DataFrame) -> np.ndarray:
        return self.spec.build(data)

    def linear_predictor(
        self, data: pl.DataFrame, *, offset: np.ndarray | None = None
    ) -> np.ndarray:
        lp = self.intercept + self.design_matrix(data) @ self.coef
        if offset is not None:
            lp = lp + np.asarray(offset, dtype=float)
        return lp

    def predict(
        self, data: pl.DataFrame, *, offset: np.ndarray | None = None
    ) -> np.ndarray:
        """Predictions on the response scale (per unit of weight if the target
        was divided by the weight). Uses the stored offset column if present."""
        if offset is None and self.offset_col:
            if self.offset_col in data.columns:
                offset = data[self.offset_col].cast(pl.Float64).to_numpy()
            else:
                warnings.warn(
                    f"Offset column '{self.offset_col}' not found in data — "
                    "predictions exclude the offset",
                    stacklevel=2,
                )
        return np.asarray(
            self.model.predict(self.design_matrix(data), offset=offset), dtype=float
        )

    def __repr__(self) -> str:
        nnz = int((self.coef != 0).sum())
        return (
            f"GLMFit(family={self.family!r}, link={self.link!r}, target="
            f"{self.target!r}, alpha={self.alpha:.4g}, features={len(self.coef)}, "
            f"non_zero={nnz}, variables={self.spec.variables})"
        )


def _modal_bins(
    spec: DesignSpec, data: pl.DataFrame, weights: np.ndarray | None
) -> dict[str, int]:
    """Index of the most exposed table row per main effect (see ``rate_tables``).
    Uses the encoders' shared ``row_index`` rule; interactions have no base row."""
    w = np.ones(data.height) if weights is None else np.asarray(weights, dtype=float)
    out: dict[str, int] = {}
    for var, enc in spec.encoders.items():
        if isinstance(enc, InteractionEncoder):
            continue
        idx = enc.row_index(data[var])
        counts = np.bincount(idx, weights=w, minlength=enc.n_rows)
        if isinstance(enc, LinearEncoder) and len(counts) > 1:
            # the base of a linear term is a point on the curve (x_base), so the
            # null row is never the base even when it carries the most exposure
            counts[-1] = -1.0
        out[var] = int(np.argmax(counts))
    return out


def interaction_penalty_weights(
    spec: DesignSpec,
    design: np.ndarray,
    weights: np.ndarray | None,
    *,
    scale_predictors: bool,
) -> np.ndarray | None:
    """Per-column L1 weights (glum ``P1``) for interaction cells.

    With ``scale_predictors=True`` glum penalises the standardised coefficient
    ``beta * sd(column)``, so a thin cell (small sd) buys a large raw effect for
    little penalty — the opposite of what a pricing model wants. Cell columns
    therefore get ``P1 = penalty_weight * 0.5 / sd``: the penalty an
    *unstandardised* column would carry, scaled so that a 50/50 cell is
    penalised exactly like a 50/50 main effect (sd 0.5). Thin cells are shrunk
    harder, fat cells like the mains. Main-effect columns keep ``P1 = 1``.
    Returns ``None`` when the spec has no interactions (glum's default applies).
    """
    if not spec.interactions:
        return None
    p1 = np.ones(spec.n_features)
    w = np.ones(design.shape[0]) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    slices = spec.slices()
    for enc in spec.interactions:
        sl = slices[enc.variable]
        cols = design[:, sl]
        if scale_predictors:
            mean = w @ cols
            var = w @ (cols**2) - mean**2
            sd = np.sqrt(np.clip(var, 0.0, None))
            sd = np.where(sd > 0, sd, 0.5)
            p1[sl] = enc.penalty_weight * 0.5 / sd
        else:
            p1[sl] = enc.penalty_weight
    return p1


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
        Standardise columns before penalising (glmnet/aglm default).
    glum_kwargs
        Anything else for the glum estimator (``max_iter``, ``P1``, ``link``,
        ``lower_bounds`` ...). Passing your own ``P1`` replaces the per-cell
        interaction penalty rule entirely (see ``interaction_penalty_weights``).
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

    fam, family_name, default_link = resolve_family(family)
    link = glum_kwargs.pop("link", default_link)

    design = spec.build(data)
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
    offset = None
    if offset_col:
        if offset_col not in data.columns:
            raise KeyError(f"Offset column {offset_col!r} not found in data")
        offset = data[offset_col].cast(pl.Float64).to_numpy()
    _validate_target(y, family_name)

    if "P1" not in glum_kwargs:
        p1 = interaction_penalty_weights(
            spec, design, sw, scale_predictors=scale_predictors
        )
        if p1 is not None:
            glum_kwargs["P1"] = p1

    lower = glum_kwargs.pop("lower_bounds", None)
    upper = glum_kwargs.pop("upper_bounds", None)
    monotone = dict(monotone or {})
    if monotone:
        mlo, mup = monotone_bounds(spec, monotone)
        lower = mlo if lower is None else np.maximum(np.asarray(lower, float), mlo)
        upper = mup if upper is None else np.minimum(np.asarray(upper, float), mup)

    common: dict[str, Any] = dict(
        family=fam,
        link=link,
        l1_ratio=l1_ratio,
        fit_intercept=True,
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
        modal_bins=_modal_bins(spec, data, sw),
        n_train_rows=data.height,
    )
