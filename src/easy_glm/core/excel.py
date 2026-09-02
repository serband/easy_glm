"""Excel export of rate tables: one worksheet per variable.

Used by :meth:`easy_glm.EasyGLM.to_excel` (fitted model: summary, coefficient
table and per-variable relativities) and :meth:`easy_glm.engine.RateModel.to_excel`
(any rate model, including one edited in the browser).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

from easy_glm.engine.models import CellRow, level_label
from easy_glm.engine.rate_model import RateModel

_INVALID_SHEET_CHARS = re.compile(r"[\[\]:*?/\\]")
_MAX_SHEET_LEN = 31


def sheet_name(key: str, used: set[str]) -> str:
    """Excel-safe, unique (case-insensitive) worksheet name for ``key``.

    Strips forbidden characters, truncates to 31 characters and appends
    `` (2)``, `` (3)``... on collision. ``used`` is updated in place.
    """
    name = _INVALID_SHEET_CHARS.sub("_", str(key)).strip("'").strip() or "sheet"
    name = name[:_MAX_SHEET_LEN]
    candidate, i = name, 2
    while candidate.lower() in used:
        suffix = f" ({i})"
        candidate = name[: _MAX_SHEET_LEN - len(suffix)] + suffix
        i += 1
    used.add(candidate.lower())
    return candidate


def suffixed_sheet_name(key: str, suffix: str, used: set[str]) -> str:
    """Like :func:`sheet_name` but guarantees the sheet name *ends with*
    ``suffix`` (e.g. ``" (matrix)"``) even when ``key`` has to be truncated and
    de-duplicated: the stem is shortened and numbered, the suffix survives."""
    base = _INVALID_SHEET_CHARS.sub("_", key).strip("'").strip() or "sheet"
    i = 1
    while True:
        extra = "" if i == 1 else f" ({i})"
        stem = base[: _MAX_SHEET_LEN - len(suffix) - len(extra)].rstrip()
        candidate = f"{stem}{extra}{suffix}"
        if candidate.lower() not in used:
            used.add(candidate.lower())
            return candidate
        i += 1


def rate_model_tables(rm: RateModel) -> dict[str, pl.DataFrame]:
    """Per-variable ``from`` / ``to`` / ``label`` / [``fitted``] / ``relativity``
    frames of a :class:`RateModel`. ``relativity`` is the *current* value (manual
    adjustments included); ``fitted`` is the first snapshot's value when present."""
    out: dict[str, pl.DataFrame] = {}
    for var, cfg in rm.variables.items():
        if cfg.type == "interaction":
            out[var] = _interaction_frame(rm, var, cfg)
            continue
        numeric = cfg.type == "numeric"
        dtype = pl.Float64 if numeric else pl.Utf8
        cast = float if numeric else str
        froms = [None if r.from_ is None else cast(r.from_) for r in cfg.table]
        tos = [None if r.to_ is None else cast(r.to_) for r in cfg.table]

        columns: dict[str, Any] = {
            "from": pl.Series(froms, dtype=dtype),
            "to": pl.Series(tos, dtype=dtype),
            "label": [level_label(r) for r in cfg.table],
        }
        # The first snapshot holds the fitted (pre-adjustment) relativities.
        base = rm.snapshots[0].relativities.get(var) if rm.snapshots else None
        if base is not None and len(base) == len(cfg.table):
            columns["fitted"] = pl.Series(
                [float(r.relativity) for r in base], dtype=pl.Float64
            )
        columns["relativity"] = pl.Series(
            [float(r.relativity) for r in cfg.table], dtype=pl.Float64
        )
        out[var] = pl.DataFrame(columns)
    return out


def _interaction_frame(rm: RateModel, var: str, cfg) -> pl.DataFrame:
    """Long table of an interaction: parent edges, labels, exposure,
    [fitted], relativity — one row per cell."""
    a, b = cfg.parents
    dt_a = pl.Float64 if rm.variables[a].type == "numeric" else pl.Utf8
    dt_b = pl.Float64 if rm.variables[b].type == "numeric" else pl.Utf8
    rows: list[CellRow] = cfg.table
    columns: dict[str, Any] = {
        "from_a": pl.Series([r.from_a for r in rows], dtype=dt_a),
        "to_a": pl.Series([r.to_a for r in rows], dtype=dt_a),
        "from_b": pl.Series([r.from_b for r in rows], dtype=dt_b),
        "to_b": pl.Series([r.to_b for r in rows], dtype=dt_b),
        "label": [level_label(r) for r in rows],
        "exposure": pl.Series([float(r.exposure) for r in rows], dtype=pl.Float64),
    }
    base = rm.snapshots[0].relativities.get(var) if rm.snapshots else None
    if base is not None and len(base) == len(rows):
        columns["fitted"] = pl.Series(
            [float(r.relativity) for r in base], dtype=pl.Float64
        )
    columns["relativity"] = pl.Series(
        [float(r.relativity) for r in rows], dtype=pl.Float64
    )
    return pl.DataFrame(columns)


def interaction_matrices(
    rm: RateModel, var: str
) -> tuple[list[str], list[str], list[list[float]], list[list[float]]]:
    """``(row_labels, col_labels, relativity_matrix, exposure_matrix)`` of an
    interaction, in the parents' table order."""
    cfg = rm.variables[var]
    a, b = cfg.parents
    rows_a = [level_label(r) for r in rm.variables[a].table]
    rows_b = [level_label(r) for r in rm.variables[b].table]
    ka = {(r.from_, r.to_): i for i, r in enumerate(rm.variables[a].table)}
    kb = {(r.from_, r.to_): i for i, r in enumerate(rm.variables[b].table)}
    rel = [[1.0] * len(rows_b) for _ in rows_a]
    exp = [[0.0] * len(rows_b) for _ in rows_a]
    for r in cfg.table:
        i, j = ka[(r.from_a, r.to_a)], kb[(r.from_b, r.to_b)]
        rel[i][j] = float(r.relativity)
        exp[i][j] = float(r.exposure)
    return rows_a, rows_b, rel, exp


def _write_matrix_sheet(
    wb, name: str, a: str, b: str, rows_a, rows_b, rel, exp, bold
) -> None:
    """Relativity matrix and exposure matrix side by side (two blocks)."""
    ws = wb.add_worksheet(name)
    ws.write(0, 0, f"{a} (rows) × {b} (columns) — relativity", bold)
    ws.write_row(1, 1, rows_b, bold)
    for i, lab in enumerate(rows_a):
        ws.write(2 + i, 0, lab, bold)
        ws.write_row(2 + i, 1, rel[i])
    gap = len(rows_b) + 3
    ws.write(0, gap, f"{a} (rows) × {b} (columns) — training exposure", bold)
    ws.write_row(1, gap + 1, rows_b, bold)
    for i, lab in enumerate(rows_a):
        ws.write(2 + i, gap, lab, bold)
        ws.write_row(2 + i, gap + 1, exp[i])
    ws.set_column(0, 0, 22)
    ws.set_column(gap, gap, 22)
    ws.freeze_panes(2, 1)


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple | set):
        return ", ".join(str(v) for v in value)
    if isinstance(value, Mapping):
        return json.dumps(value, default=str)
    return str(value)


def write_rate_tables_xlsx(
    tables: Mapping[str, pl.DataFrame],
    path: str | Path,
    *,
    summary: Mapping[str, Any] | None = None,
    coef_table: pl.DataFrame | None = None,
    index_sheet: bool = True,
    matrices: Mapping[str, tuple] | None = None,
) -> Path:
    """Write ``tables`` to an ``.xlsx`` workbook, one worksheet per table.

    Optional leading sheets: ``Summary`` (key/value pairs from ``summary``),
    ``Index`` (sheet name -> variable -> row count, useful when names were
    truncated) and ``Coefficients``. ``matrices`` maps an interaction name to
    ``(a, b, row_labels, col_labels, relativity_matrix, exposure_matrix)`` and
    adds one ``"<name> (matrix)"`` sheet per interaction with the relativity
    and exposure grids side by side. Returns the path written.
    """
    import xlsxwriter

    path = Path(path)
    used: set[str] = set()
    index_rows: list[tuple[str, str, int]] = []

    with xlsxwriter.Workbook(str(path)) as wb:
        bold = wb.add_format({"bold": True})

        if summary is not None:
            ws = wb.add_worksheet("Summary")
            used.add("summary")
            for r, (k, v) in enumerate(summary.items()):
                ws.write(r, 0, str(k), bold)
                ws.write(r, 1, _cell(v))
            ws.set_column(0, 0, 24)
            ws.set_column(1, 1, 80)

        index_ws = None
        if index_sheet:
            index_ws = wb.add_worksheet("Index")
            used.add("index")

        if coef_table is not None:
            used.add("coefficients")
            coef_table.write_excel(workbook=wb, worksheet="Coefficients", autofit=True)

        for key, frame in tables.items():
            name = sheet_name(str(key), used)
            frame.write_excel(workbook=wb, worksheet=name, autofit=True)
            index_rows.append((name, str(key), frame.height))
            if matrices and key in matrices:
                a, b, rows_a, rows_b, rel, exp = matrices[key]
                mname = suffixed_sheet_name(str(key), " (matrix)", used)
                _write_matrix_sheet(wb, mname, a, b, rows_a, rows_b, rel, exp, bold)
                index_rows.append((mname, f"{key} (matrix)", len(rows_a)))

        if index_ws is not None:
            index_ws.write_row(0, 0, ["sheet", "variable", "rows"], bold)
            for r, row in enumerate(index_rows, start=1):
                index_ws.write_row(r, 0, row)
            index_ws.set_column(0, 1, 36)

    return path
