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

from easy_glm.engine.models import level_label
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


def rate_model_tables(rm: RateModel) -> dict[str, pl.DataFrame]:
    """Per-variable ``from`` / ``to`` / ``label`` / ``relativity`` frames of a
    :class:`RateModel` (its *current* version)."""
    out: dict[str, pl.DataFrame] = {}
    for var, cfg in rm.variables.items():
        numeric = cfg.type == "numeric"
        dtype = pl.Float64 if numeric else pl.Utf8
        cast = float if numeric else str
        froms = [None if r.from_ is None else cast(r.from_) for r in cfg.table]
        tos = [None if r.to_ is None else cast(r.to_) for r in cfg.table]

        out[var] = pl.DataFrame(
            {
                "from": pl.Series(froms, dtype=dtype),
                "to": pl.Series(tos, dtype=dtype),
                "label": [level_label(r) for r in cfg.table],
                "relativity": pl.Series(
                    [float(r.relativity) for r in cfg.table], dtype=pl.Float64
                ),
            }
        )
    return out


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
) -> Path:
    """Write ``tables`` to an ``.xlsx`` workbook, one worksheet per table.

    Optional leading sheets: ``Summary`` (key/value pairs from ``summary``),
    ``Index`` (sheet name -> variable -> row count, useful when names were
    truncated) and ``Coefficients``. Returns the path written.
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

        if index_ws is not None:
            index_ws.write_row(0, 0, ["sheet", "variable", "rows"], bold)
            for r, row in enumerate(index_rows, start=1):
                index_ws.write_row(r, 0, row)
            index_ws.set_column(0, 1, 36)

    return path
