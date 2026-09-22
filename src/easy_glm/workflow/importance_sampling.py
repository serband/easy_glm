"""Deterministic, pre-declared scoring samples for permutation importance."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl


def validate_importance_sample_pct(value: float) -> float:
    """Return a valid importance scoring percentage, rejecting booleans."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 < float(value) <= 100
    ):
        raise ValueError("importance_sample_pct must be a finite number in (0, 100].")
    return float(value)


def _effective_rows(weights: np.ndarray) -> float:
    total = float(weights.sum())
    squared = float(weights @ weights)
    return total * total / squared if total > 0 and squared > 0 else 0.0


def _constant_after_sampling(
    frame: pl.DataFrame, indices: np.ndarray, variables: Sequence[str]
) -> list[str]:
    changed: list[str] = []
    sampled = frame[indices]
    for variable in variables:
        if variable not in frame.columns:
            continue
        if frame[variable].n_unique() > 1 and sampled[variable].n_unique() <= 1:
            changed.append(variable)
    return changed


def importance_sample_indices(
    frame: pl.DataFrame,
    *,
    importance_sample_pct: float = 30.0,
    seed: int = 42,
    outcome: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    family: str | None = None,
    variables: Sequence[str] = (),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose one sorted scoring sample and describe any full-data fallback.

    The decision uses only fixed coverage checks, before any importance value is
    calculated. Model fitting, cross-validation and encoder construction remain
    on the complete training frame.
    """
    percentage = validate_importance_sample_pct(importance_sample_pct)
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 2**32 - 1
    ):
        raise ValueError("seed must be an integer in [0, 2**32 - 1].")
    n_rows = frame.height
    if n_rows < 1:
        raise ValueError("Variable importance needs at least one training row.")
    all_indices = np.arange(n_rows, dtype=np.int64)
    if weights is None:
        full_weights = np.ones(n_rows, dtype=np.float64)
    else:
        full_weights = np.asarray(weights, dtype=np.float64)
        if full_weights.shape != (n_rows,):
            raise ValueError("weights must have one value per training row.")
    if not np.all(np.isfinite(full_weights)) or np.any(full_weights < 0):
        raise ValueError("Variable importance weights must be finite and nonnegative.")
    total_weight = float(full_weights.sum())
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("Variable importance needs positive finite total weight.")

    reasons: list[str] = []
    if percentage == 100:
        indices = all_indices
    else:
        proposed_rows = max(1, math.floor(percentage * n_rows / 100))
        indices = np.sort(
            np.random.default_rng(seed).choice(n_rows, proposed_rows, replace=False)
        ).astype(np.int64, copy=False)
        if proposed_rows < 1_000:
            reasons.append("The proposed scoring sample had fewer than 1,000 rows.")
        sampled_weights = full_weights[indices]
        if _effective_rows(sampled_weights) < 100:
            reasons.append(
                "The proposed scoring sample had fewer than 100 effective rows."
            )
        if outcome is not None:
            values = np.asarray(outcome, dtype=np.float64)
            if values.shape != (n_rows,):
                raise ValueError("outcome must have one value per training row.")
            contributing = np.isfinite(values[indices]) & (sampled_weights > 0)
            sampled_outcome = values[indices]
            family_name = (family or "").strip().lower()
            full_contributing = np.isfinite(values) & (full_weights > 0)
            if (
                np.unique(sampled_outcome[contributing]).size <= 1
                and np.unique(values[full_contributing]).size > 1
            ):
                reasons.append(
                    "The proposed scoring sample made the outcome constant although it varies on full training data."
                )
            if family_name in {"poisson", "tweedie"}:
                positive = int(np.sum(contributing & (sampled_outcome > 0)))
                if positive < 20:
                    reasons.append(
                        "The proposed scoring sample had fewer than 20 positive-outcome rows."
                    )
            elif family_name == "binomial":
                successes = int(np.sum(contributing & (sampled_outcome > 0)))
                failures = int(np.sum(contributing & (sampled_outcome < 1)))
                if successes < 20 or failures < 20:
                    reasons.append(
                        "The proposed scoring sample had fewer than 20 contributing success or failure rows."
                    )
        constant = _constant_after_sampling(frame, indices, variables)
        if constant:
            preview = ", ".join(constant[:5])
            suffix = "" if len(constant) <= 5 else f" and {len(constant) - 5} more"
            reasons.append(
                "The proposed scoring sample made a variable constant that varies "
                f"on full training data: {preview}{suffix}."
            )
        if reasons:
            indices = all_indices

    selected_weight = float(full_weights[indices].sum())
    final_weights = full_weights[indices]
    effective_rows = _effective_rows(final_weights)
    support_warnings: list[str] = []
    if effective_rows < 100:
        support_warnings.append(
            "The scored data have fewer than 100 effective rows; importance may be unstable."
        )
    positive_outcome_rows: int | None = None
    success_rows: int | None = None
    failure_rows: int | None = None
    if outcome is not None:
        values = np.asarray(outcome, dtype=np.float64)[indices]
        contributing = np.isfinite(values) & (final_weights > 0)
        family_name = (family or "").strip().lower()
        if family_name in {"poisson", "tweedie"}:
            positive_outcome_rows = int(np.sum(contributing & (values > 0)))
            if positive_outcome_rows < 20:
                support_warnings.append(
                    "The scored data have fewer than 20 positive-outcome rows; importance may be unstable."
                )
        elif family_name == "binomial":
            success_rows = int(np.sum(contributing & (values > 0)))
            failure_rows = int(np.sum(contributing & (values < 1)))
            if success_rows < 20 or failure_rows < 20:
                support_warnings.append(
                    "The scored data have fewer than 20 contributing success or failure rows; importance may be unstable."
                )
    rows = int(indices.size)
    metadata = {
        "requested_pct": percentage,
        "importance_sample_pct": percentage,
        "full_training_rows": n_rows,
        "importance_rows": rows,
        "actual_pct": 100.0 * rows / n_rows,
        "weight_share": selected_weight / total_weight,
        "effective_rows": effective_rows,
        "positive_outcome_rows": positive_outcome_rows,
        "success_rows": success_rows,
        "failure_rows": failure_rows,
        "seed": seed,
        "fallback_reasons": reasons,
        "support_warnings": support_warnings,
    }
    return indices, metadata
