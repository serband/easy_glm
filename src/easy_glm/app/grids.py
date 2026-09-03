"""Pure helpers behind the workbench's editable grids (no Streamlit).

Kept free of Streamlit so the edit rules can be unit-tested directly:
* :func:`apply_row_edits` — step / categorical / linear tables (one value per row);
* :func:`apply_cell_edits` — interaction matrices (one value per cell);
* :func:`cell_grid` / :func:`pair_matrices` — matrices for the heatmaps.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from easy_glm.core.excel import rate_model_tables
from easy_glm.engine.models import level_label
from easy_glm.engine.rate_model import RateModel
from easy_glm.workflow import Adjustment
from easy_glm.workflow.project import ModelConfig

TOL = 1e-9


def _positive(value: float) -> bool:
    return value is not None and value == value and value > 0


def apply_row_edits(
    cfg: ModelConfig,
    var: str,
    rows: list[Any],
    fitted: list[float],
    edited: list[float],
    *,
    require_positive: bool,
) -> tuple[bool, list[str]]:
    """Turn edited row values into adjustments on ``cfg``.

    For each row whose edited value differs from the current one, any existing
    adjustment on that row is replaced; a value equal to the fitted one removes
    the adjustment instead of recording it. Returns ``(changed, errors)`` —
    errors are human-readable and the offending rows are left untouched."""
    changed = False
    errors: list[str] = []
    for row, fit_val, new in zip(rows, fitted, edited, strict=True):
        try:
            new = float(new)
        except (TypeError, ValueError):
            errors.append(f"{level_label(row)!r}: not a number; change not saved")
            continue
        if new != new:  # NaN (an emptied cell)
            errors.append(f"{level_label(row)!r}: empty; change not saved")
            continue
        if abs(new - row.relativity) <= TOL:
            continue
        if require_positive and not _positive(new):
            errors.append(
                f"{level_label(row)!r}: relativities must be above 0 (was {new:g}); "
                "change not saved"
            )
            continue
        if new < 0:
            errors.append(
                f"{level_label(row)!r}: a negative relativity is not meaningful "
                f"(was {new:g}); change not saved"
            )
            continue
        cfg.adjustments = [
            a
            for a in cfg.adjustments
            if not (
                a.variable == var
                and not a.cell
                and a.from_ == row.from_
                and a.to_ == row.to_
            )
        ]
        if abs(new - float(fit_val)) > TOL:
            cfg.adjustments.append(Adjustment(var, row.from_, row.to_, new))
        changed = True
    return changed, errors


def cell_grid(rm: RateModel, var: str) -> dict[str, Any]:
    """Matrices of an interaction in the parents' row order:
    ``rows``, ``cols`` (labels), ``keys`` (cell keys), ``current``, ``fitted``
    (first snapshot, or current when absent), ``exposure``."""
    cfg = rm.variables[var]
    a, b = cfg.parents
    rows_a = [level_label(r) for r in rm.variables[a].table]
    rows_b = [level_label(r) for r in rm.variables[b].table]
    ka = {(r.from_, r.to_): i for i, r in enumerate(rm.variables[a].table)}
    kb = {(r.from_, r.to_): i for i, r in enumerate(rm.variables[b].table)}
    n_a, n_b = len(rows_a), len(rows_b)
    current = [[1.0] * n_b for _ in range(n_a)]
    fitted = [[1.0] * n_b for _ in range(n_a)]
    exposure = [[0.0] * n_b for _ in range(n_a)]
    keys: list[list[tuple | None]] = [[None] * n_b for _ in range(n_a)]
    base = rm.snapshots[0].relativities.get(var) if rm.snapshots else None
    fitted_by_key = (
        {row.key: float(row.relativity) for row in base}
        if base is not None and len(base) == len(cfg.table)
        else {}
    )
    for row in cfg.table:
        i = ka.get((row.from_a, row.to_a))
        j = kb.get((row.from_b, row.to_b))
        if i is None or j is None:
            continue
        current[i][j] = float(row.relativity)
        fitted[i][j] = fitted_by_key.get(row.key, float(row.relativity))
        exposure[i][j] = float(row.exposure)
        keys[i][j] = row.key
    return {
        "rows": rows_a,
        "cols": rows_b,
        "keys": keys,
        "current": current,
        "fitted": fitted,
        "exposure": exposure,
        "parents": (a, b),
    }


def apply_cell_edits(
    cfg: ModelConfig,
    var: str,
    grid: dict[str, Any],
    edited: list[list[float]],
) -> tuple[bool, list[str]]:
    """Turn an edited interaction matrix into cell adjustments on ``cfg``
    (same replace / remove-when-fitted rule as :func:`apply_row_edits`)."""
    changed = False
    errors: list[str] = []
    for i, row_label in enumerate(grid["rows"]):
        for j, col_label in enumerate(grid["cols"]):
            key = grid["keys"][i][j]
            if key is None:
                continue
            try:
                new = float(edited[i][j])
            except (TypeError, ValueError, IndexError):
                errors.append(
                    f"{row_label} | {col_label}: not a number; change not saved"
                )
                continue
            if new != new:
                errors.append(f"{row_label} | {col_label}: empty; change not saved")
                continue
            if abs(new - grid["current"][i][j]) <= TOL:
                continue
            if not _positive(new):
                errors.append(
                    f"{row_label} | {col_label}: an adjustment must be above 0 "
                    f"(was {new:g}); change not saved"
                )
                continue
            fa, ta, fb, tb = key
            cfg.adjustments = [
                a
                for a in cfg.adjustments
                if not (
                    a.variable == var
                    and a.cell
                    and (a.from_, a.to_, a.from_b, a.to_b) == key
                )
            ]
            if abs(new - grid["fitted"][i][j]) > TOL:
                cfg.adjustments.append(
                    Adjustment(var, fa, ta, new, from_b=fb, to_b=tb, cell=True)
                )
            changed = True
    return changed, errors


def pair_matrices(table: pl.DataFrame) -> dict[str, Any]:
    """Pivot an ``ae_by_pair`` long table into matrices in row order:
    ``rows``, ``cols``, ``ae``, ``actual``, ``expected``, ``exposure``."""
    a_levels = (
        table.select("label_a", "order_a").unique().sort("order_a")["label_a"].to_list()
    )
    b_levels = (
        table.select("label_b", "order_b").unique().sort("order_b")["label_b"].to_list()
    )
    ia = {lab: i for i, lab in enumerate(a_levels)}
    ib = {lab: j for j, lab in enumerate(b_levels)}
    n_a, n_b = len(a_levels), len(b_levels)
    ae = [[None] * n_b for _ in range(n_a)]
    actual = [[0.0] * n_b for _ in range(n_a)]
    expected = [[0.0] * n_b for _ in range(n_a)]
    exposure = [[0.0] * n_b for _ in range(n_a)]
    for r in table.iter_rows(named=True):
        i, j = ia[r["label_a"]], ib[r["label_b"]]
        v = r["ae"]
        ae[i][j] = float(v) if v is not None and v == v and r["expected"] > 0 else None
        actual[i][j] = float(r["actual"])
        expected[i][j] = float(r["expected"])
        exposure[i][j] = float(r["exposure"])
    return {
        "rows": a_levels,
        "cols": b_levels,
        "ae": ae,
        "actual": actual,
        "expected": expected,
        "exposure": exposure,
    }


def linear_tables(run, var: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``(fitted, working)`` tables of a piecewise-linear variable with the
    columns the curve chart and the editor need."""
    fitted = run.tables[var]
    working = rate_model_tables(run.rate_model)[var]
    return fitted, working
