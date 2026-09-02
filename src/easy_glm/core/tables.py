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
from typing import Any, Literal

import numpy as np
import polars as pl

from easy_glm.engine.models import (
    CellRow,
    FromToRow,
    ModelMetadata,
    VariableConfig,
    level_label,
)
from easy_glm.engine.rate_model import RateModel

from .design import CategoricalEncoder, InteractionEncoder, StepEncoder
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
    elif isinstance(enc, CategoricalEncoder):
        contrib = np.concatenate([[0.0], coef])  # reference, levels[1:], other
        rows = [FromToRow(lvl, lvl, 1.0) for lvl in enc.levels]
        rows.append(FromToRow(None, None, 1.0))  # other / unseen / null
    else:
        raise NotImplementedError(f"No rate-table rule for {type(enc).__name__}")
    return rows, contrib


def _cell_rows(fit: GLMFit, variable: str) -> tuple[list[CellRow], np.ndarray]:
    """All cells of an interaction (row-major over the parents' rows) and each
    cell's coefficient (0 for cells that were not kept)."""
    enc = fit.spec[variable]
    if not isinstance(enc, InteractionEncoder):
        raise TypeError(f"{variable!r} is not an interaction")
    coef = fit.coef[fit.spec.slices()[variable]]
    na, nb = enc.a.n_rows, enc.b.n_rows
    contrib = np.zeros((na, nb))
    for c, (i, j) in zip(coef, enc.cells, strict=True):
        contrib[i, j] = c
    exposure = np.asarray(enc.exposure, dtype=float)
    rows_a, rows_b = enc.a.rows(), enc.b.rows()
    rows = [
        CellRow(ra[0], ra[1], rb[0], rb[1], 1.0, float(exposure[i, j]))
        for i, ra in enumerate(rows_a)
        for j, rb in enumerate(rows_b)
    ]
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
        enc = fit.spec[var]
        if isinstance(enc, InteractionEncoder):
            out[var] = _interaction_table(fit, var)
            continue
        rows, contrib = _bin_rows(fit, var)
        b = fit.modal_bins.get(var, 0) if base == "modal" else 0
        rel_lp = contrib - contrib[b]
        edge_dtype = pl.Float64 if isinstance(enc, StepEncoder) else pl.Utf8
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


def _edge_dtype(enc) -> Any:
    return pl.Float64 if isinstance(enc, StepEncoder) else pl.Utf8


def _interaction_table(fit: GLMFit, variable: str) -> pl.DataFrame:
    """Long table of an interaction: one row per cell (kept or not), with
    parent edges, labels, training exposure, ``kept``, ``coef`` and the
    multiplicative adjustment ``relativity`` (1.0 for cells not kept). Cells
    are *not* re-based: 1.0 always means "no adjustment"."""
    enc = fit.spec[variable]
    assert isinstance(enc, InteractionEncoder)
    rows, contrib = _cell_rows(fit, variable)
    kept = np.zeros(contrib.shape, dtype=bool)
    for i, j in enc.cells:
        kept[i, j] = True
    flat = contrib.ravel()
    labels = enc.cell_labels()
    return pl.DataFrame(
        {
            "from_a": pl.Series([r.from_a for r in rows], dtype=_edge_dtype(enc.a)),
            "to_a": pl.Series([r.to_a for r in rows], dtype=_edge_dtype(enc.a)),
            "from_b": pl.Series([r.from_b for r in rows], dtype=_edge_dtype(enc.b)),
            "to_b": pl.Series([r.to_b for r in rows], dtype=_edge_dtype(enc.b)),
            "label": [level_label(r) for r in rows],
            "label_a": [la for la, _ in labels],
            "label_b": [lb for _, lb in labels],
            "exposure": [r.exposure for r in rows],
            "kept": kept.ravel().tolist(),
            "coef": flat,
            "relativity": np.exp(flat),
            "is_base": [False] * len(rows),
        }
    )


def base_rate(fit: GLMFit, *, base: Base = "modal") -> float:
    """Prediction (per unit weight) for the base risk implied by ``base``."""
    _check_log_link(fit)
    lp = fit.intercept
    for var in fit.spec.main_effects:
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
    any data, including fits with an offset column (the RateModel stores the
    offset column name and applies ``exp(offset)`` at scoring) and with
    interactions (mains × a cell adjustment matrix; the base rate is the
    prediction for the base risk *before* interaction adjustments, so an
    interaction cell of 1.0 always means "no adjustment"). Pass
    ``base_rate_override`` to rescale (e.g. for a target loss ratio); the
    relativities are unaffected. ``model_type`` is a label stored in the
    metadata (defaults to the canonical family name).
    """
    _check_log_link(fit)
    variables: dict[str, VariableConfig] = {}
    for var in fit.spec.variables:
        enc = fit.spec[var]
        if isinstance(enc, InteractionEncoder):
            cells, contrib = _cell_rows(fit, var)
            for row, c in zip(cells, contrib.ravel(), strict=True):
                row.relativity = float(math.exp(c))
            variables[var] = VariableConfig(
                type="interaction", table=cells, parents=enc.parents
            )
            continue
        rows, contrib = _bin_rows(fit, var)
        b = fit.modal_bins.get(var, 0) if base == "modal" else 0
        rel = np.exp(contrib - contrib[b])
        for row, r in zip(rows, rel, strict=True):
            row.relativity = float(r)
        kind = "numeric" if isinstance(enc, StepEncoder) else "categorical"
        variables[var] = VariableConfig(type=kind, table=rows)
    RateModel._precompute_variables(variables)

    metadata = ModelMetadata(
        model_type=model_type or fit.family,
        target=fit.target,
        weight_col=fit.weight_col,
        exposure_col=exposure_col,
        train_test_col=train_test_col,
        predictor_variables=list(variables),
        offset_col=fit.offset_col,
        offset_is_log=True,
        link=fit.link,
        divide_target_by_weight=fit.divide_target_by_weight,
    )
    rate = (
        base_rate(fit, base=base)
        if base_rate_override is None
        else float(base_rate_override)
    )
    rm = RateModel(base_rate=rate, variables=variables, metadata=metadata)
    rm.create_snapshot("Base model")
    return rm
