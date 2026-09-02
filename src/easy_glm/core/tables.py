"""Exact rate tables read straight off a fitted GLM's coefficients.

With step (O-dummy) and one-hot features and a log link, the linear
predictor decomposes additively by variable, so every bin's relativity is
``exp`` of a partial sum of coefficients and the product of the per-variable
relativities times the base rate reproduces the GLM prediction *exactly*
(including the lowest bin, nulls and the ``Other`` bucket). No sampling, no
random reference row.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import polars as pl

from easy_glm.engine.models import FromToRow, ModelMetadata, VariableConfig, level_label
from easy_glm.engine.rate_model import RateModel

from .design import CategoricalEncoder, StepEncoder
from .fit import GLMFit

Base = Literal["modal", "reference"]


def _bin_rows(fit: GLMFit, variable: str) -> tuple[list[FromToRow], np.ndarray]:
    """Table rows for ``variable`` and each row's linear-predictor contribution
    relative to the encoder's reference (lowest bin / reference level)."""
    enc = fit.spec[variable]
    coef = fit.coef[fit.spec.slices()[variable]]
    if isinstance(enc, StepEncoder):
        n_knots = len(enc.knots)
        step_coefs = coef[:n_knots]
        contrib = np.concatenate([[0.0], np.cumsum(step_coefs)])  # one per bin
        null_contrib = float(coef[n_knots]) if enc.null_indicator else 0.0
        rows = [FromToRow(lo, hi, 1.0) for lo, hi in enc.bins()]
        rows.append(FromToRow(None, None, 1.0))  # nulls
        contrib = np.append(contrib, null_contrib)
    else:
        assert isinstance(enc, CategoricalEncoder)
        contrib = np.concatenate([[0.0], coef])  # reference, levels[1:], other
        rows = [FromToRow(lvl, lvl, 1.0) for lvl in enc.levels]
        rows.append(FromToRow(None, None, 1.0))  # other / unseen / null
    return rows, contrib


def _check_log_link(fit: GLMFit) -> None:
    if fit.link != "log":
        raise NotImplementedError(
            f"Multiplicative rate tables need a log link; this fit uses "
            f"{fit.link!r}. Use fit.coef_table() / fit.predict() instead."
        )


def rate_tables(fit: GLMFit, *, base: Base = "modal") -> dict[str, pl.DataFrame]:
    """One relativity table per variable.

    Columns: ``from``, ``to`` (Float64 for numeric, Utf8 for categorical;
    null = open end, both null = the null / Other row), ``label``, ``coef``
    (linear-predictor contribution relative to the base row), ``relativity``
    (``exp(coef)``) and ``is_base``.

    ``base="modal"`` puts relativity 1.0 on the most exposed bin of the
    training data; ``"reference"`` uses the lowest bin / reference level.
    """
    _check_log_link(fit)
    out: dict[str, pl.DataFrame] = {}
    for var in fit.spec.variables:
        rows, contrib = _bin_rows(fit, var)
        b = fit.modal_bins.get(var, 0) if base == "modal" else 0
        rel_lp = contrib - contrib[b]
        edge_dtype = pl.Float64 if isinstance(fit.spec[var], StepEncoder) else pl.Utf8
        out[var] = pl.DataFrame(
            {
                "from": pl.Series([r.from_ for r in rows], dtype=edge_dtype),
                "to": pl.Series([r.to_ for r in rows], dtype=edge_dtype),
                "label": [level_label(r) for r in rows],
                "coef": rel_lp,
                "relativity": np.exp(rel_lp),
                "is_base": [i == b for i in range(len(rows))],
            }
        )
    return out


def base_rate(fit: GLMFit, *, base: Base = "modal") -> float:
    """Prediction (per unit weight) for the base risk implied by ``base``."""
    _check_log_link(fit)
    lp = fit.intercept
    for var in fit.spec.variables:
        _, contrib = _bin_rows(fit, var)
        b = fit.modal_bins.get(var, 0) if base == "modal" else 0
        lp += float(contrib[b])
    return math.exp(lp)


def to_rate_model(
    fit: GLMFit,
    *,
    base: Base = "modal",
    base_rate_override: float | None = None,
    exposure_col: str | None = None,
    train_test_col: str | None = None,
    model_type: str | None = None,
) -> RateModel:
    """Compile the GLM into a :class:`RateModel` that reproduces it exactly.

    ``rm.predict(data, exposure_col=None)`` equals ``fit.predict(data)`` for
    any data. Pass ``base_rate_override`` to rescale (e.g. for a target loss
    ratio); the relativities are unaffected. ``model_type`` is a label stored
    in the metadata (defaults to the canonical family name).
    """
    _check_log_link(fit)
    variables: dict[str, VariableConfig] = {}
    for var in fit.spec.variables:
        rows, contrib = _bin_rows(fit, var)
        b = fit.modal_bins.get(var, 0) if base == "modal" else 0
        rel = np.exp(contrib - contrib[b])
        for row, r in zip(rows, rel, strict=True):
            row.relativity = float(r)
        kind = "numeric" if isinstance(fit.spec[var], StepEncoder) else "categorical"
        variables[var] = VariableConfig(type=kind, table=rows)
    RateModel._precompute_variables(variables)

    metadata = ModelMetadata(
        model_type=model_type or fit.family,
        target=fit.target,
        weight_col=fit.weight_col,
        exposure_col=exposure_col,
        train_test_col=train_test_col,
        predictor_variables=list(variables),
    )
    rate = (
        base_rate(fit, base=base)
        if base_rate_override is None
        else float(base_rate_override)
    )
    rm = RateModel(base_rate=rate, variables=variables, metadata=metadata)
    rm.create_snapshot("Base model")
    return rm
