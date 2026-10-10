"""Training-only residual discovery for an accepted logical-model checkpoint.

This module deliberately has no server or job-manager dependency.  The route
chooses an *already accepted* saved run and passes it here; discovery then uses
that run's complete scorer, including any accepted lookup tables.  A candidate
under review is never an input to this calculation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import polars as pl

from easy_glm.workflow.diagnostics import (
    pearson_dispersion,
    residual_factor_search,
    residual_pair_search,
    totals,
)
from easy_glm.workflow.prep import train_holdout
from easy_glm.workflow.project import Project

SearchAction = Literal["factors", "interactions"]


def eligible_columns(project: Project, frame: pl.DataFrame) -> list[str]:
    """Raw predictor/unassigned columns that may be investigated.

    The role grid is shared across the project.  Discovery must therefore not
    promote an unassigned column or let control columns enter a search merely
    because a recipe happens to mention them.
    """
    protected = {
        value
        for value in (
            project.target,
            project.weight,
            project.exposure,
            project.offset,
            project.current_premium,
            project.offset_column,
            project.data.split.column,
        )
        if isinstance(value, str) and value
    }
    derived = {item.name for item in project.data.derived}
    return sorted(
        column
        for column in frame.columns
        if project.data.roles.get(column) in (None, "predictor")
        and column not in protected
        and column not in derived
    )


def _accepted_pair_sets(
    run: Any, supplied: Iterable[tuple[str, str]]
) -> set[frozenset[str]]:
    """Collect pair tables already in the accepted scorer, in either order."""
    pairs = {frozenset(pair) for pair in supplied if len(pair) == 2}
    config = getattr(run, "config", None)
    for interaction in getattr(config, "interactions", ()):
        pairs.add(frozenset((interaction.a, interaction.b)))
    for stage in getattr(config, "pair_stages", ()):
        pairs.add(frozenset((stage.a, stage.b)))
    stack = getattr(run, "stack", None)
    for artifact in getattr(stack, "components", ()):
        recipe = getattr(artifact, "recipe", None)
        factors = getattr(recipe, "factors", ())
        if getattr(recipe, "kind", None) == "interaction" and len(factors) == 2:
            pairs.add(frozenset(factors))
    return pairs


def _n_parameters(run: Any) -> int:
    """A conservative dispersion adjustment for ordinary and stack runs."""
    fit = getattr(run, "fit", None)
    coefficients = getattr(fit, "coef", None)
    return int(len(coefficients)) if coefficients is not None else 0


def _dispersion_and_note(
    run: Any, actual: np.ndarray, expected: np.ndarray
) -> tuple[float, str]:
    family = str(getattr(getattr(run, "config", None), "family", "poisson")).lower()
    if family == "gaussian":
        raise ValueError(
            "Residual search is not available for Gaussian models. Its Pearson "
            "count-scale signal is not valid for Gaussian outcomes; inspect A/E "
            "or use a Gaussian residual analysis instead."
        )
    if family == "poisson":
        return (
            1.0,
            "Poisson signal is a noise-adjusted Pearson excess ranking. "
            "Use it to choose what to review, then evaluate any refit separately.",
        )
    dispersion = pearson_dispersion(actual, expected, _n_parameters(run))
    if not np.isfinite(dispersion) or dispersion <= 0:
        dispersion = 1.0
    return (
        float(dispersion),
        "Signal is an exploratory residual ranking scaled by Pearson dispersion "
        f"({dispersion:.4g}) for this {family} model. It is not a calibrated "
        "count-model z-score.",
    )


def discover(
    project: Project,
    prepared: pl.DataFrame,
    run: Any,
    *,
    action: SearchAction,
    main_factors: Iterable[str] = (),
    accepted_pairs: Iterable[tuple[str, str]] = (),
    n_bins: int = 8,
    min_expected: float = 3.0,
    min_cell_share: float = 0.0,
    knots: dict[str, list[float]] | None = None,
    levels: dict[str, list[str]] | None = None,
    top: int = 20,
) -> dict[str, Any]:
    """Rank missing factors or interactions on the accepted scorer's training rows.

    ``run`` must be the selected accepted checkpoint.  Its :meth:`predict`
    method is intentionally called once on training rows and that exact output
    supplies the expected totals.  The helper has no fitting or persistence
    side effects.
    """
    if action not in {"factors", "interactions"}:
        raise ValueError("Search action must be 'factors' or 'interactions'")
    if n_bins < 2:
        raise ValueError("Search bands must be at least 2")
    if min_expected <= 0:
        raise ValueError("Minimum expected must be positive")
    if not 0 <= min_cell_share < 1:
        raise ValueError("Minimum cell share must be at least 0 and below 1")

    train, _holdout = train_holdout(prepared, project.data.split)
    if train.is_empty():
        raise ValueError("No training rows are available for residual search")

    prediction = np.asarray(run.predict(train), dtype=float)
    if prediction.shape != (train.height,) or not np.all(np.isfinite(prediction)):
        raise ValueError("The accepted scorer returned invalid training predictions")
    actual, expected, weight = totals(train, run.config, prediction)
    dispersion, note = _dispersion_and_note(run, actual, expected)
    eligible = eligible_columns(project, train)
    current_mains = set(main_factors)

    if action == "factors":
        candidates = [column for column in eligible if column not in current_mains]
        rows = residual_factor_search(
            train,
            candidates,
            actual,
            expected,
            weight,
            n_bins=n_bins,
            min_expected=min_expected,
            dispersion=dispersion,
        ).to_dicts()
    else:
        accepted = _accepted_pair_sets(run, accepted_pairs)
        pairs = [
            (left, right)
            for index, left in enumerate(eligible)
            for right in eligible[index + 1 :]
            if frozenset((left, right)) not in accepted
        ]
        rows = residual_pair_search(
            train,
            eligible,
            actual,
            expected,
            weight,
            knots=knots,
            levels=levels,
            n_bins=n_bins,
            min_expected=min_expected,
            min_cell_share=min_cell_share,
            pairs=pairs,
            top=top,
            dispersion=dispersion,
        ).to_dicts()

    return {
        "rows": rows,
        "subset": "train",
        "eligible_variables": eligible,
        "source": "Complete accepted scorer on training rows",
        "dispersion": dispersion,
        "note": note,
    }
