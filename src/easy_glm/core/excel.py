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

from easy_glm.engine.models import (
    CellRow,
    FromToRow,
    PairCellRow,
    PairTableConfig,
    VariableConfig,
    level_label,
)
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
    / ``exposure`` frames of a :class:`RateModel`. ``relativity`` is the
    *current* value (manual adjustments included); ``fitted`` is the first
    snapshot's value when present; ``exposure`` is the training exposure that
    fell in the row (0.0 for a hand-built table). Piecewise-linear variables add
    ``slope``, ``relativity_to`` (value at the band end) and ``is_base`` (the
    band starting at ``x_base``); their ``relativity`` is the value at the band
    start."""
    out: dict[str, pl.DataFrame] = {}
    for var, cfg in rm.variables.items():
        if cfg.type == "interaction":
            out[var] = _interaction_frame(rm, var, cfg)
            continue
        base = rm.snapshots[0].relativities.get(var) if rm.snapshots else None
        out[var] = variable_frame(cfg, fitted=base)
    return out


def pair_table_frames(rm: RateModel) -> dict[str, pl.DataFrame]:
    """Long deployed pair tables keyed by stable stage ID, separate from mains."""
    out: dict[str, pl.DataFrame] = {}
    for table in rm.pair_tables:
        axis_a, axis_b = table.axes
        cells = {(c.axis_a_row, c.axis_b_row): c for c in table.cells}
        rows: list[dict[str, Any]] = []
        for ia, a in enumerate(axis_a.table):
            for ib, b in enumerate(axis_b.table):
                cell = cells.get((ia, ib))
                rows.append(
                    {
                        "stage_id": table.stage_id,
                        "parent_a": table.parents[0],
                        "parent_b": table.parents[1],
                        "axis_a_row": ia,
                        "axis_b_row": ib,
                        "from_a": a.from_,
                        "to_a": a.to_,
                        "from_b": b.from_,
                        "to_b": b.to_,
                        "label_a": level_label(a, axis_a.other_label),
                        "label_b": level_label(b, axis_b.other_label),
                        "relativity": cell.relativity if cell else 1.0,
                        "row_count": cell.row_count if cell else 0,
                        "fitting_weight": cell.fitting_weight if cell else 0.0,
                        "weight_share": cell.weight_share if cell else 0.0,
                        "fallback_reason": (
                            cell.fallback_reason if cell else "unrepresented"
                        ),
                        "exposure_total": cell.exposure_total if cell else None,
                    }
                )
        out[table.stage_id] = pl.DataFrame(rows)
    return out


def pair_tables_from_xlsx(path: str | Path) -> list[PairTableConfig]:
    """Rebuild the ordered pair tables from an exported workbook alone."""
    manifest = pl.read_excel(path, sheet_name="Pair stages")
    tables: list[PairTableConfig] = []
    for stage in manifest.sort("order").iter_rows(named=True):
        axes: list[VariableConfig] = []
        for side in ("a", "b"):
            frame = pl.read_excel(path, sheet_name=stage[f"axis_{side}_sheet"])
            axis_type = stage[f"axis_{side}_type"]
            rows = [
                FromToRow(
                    row["from"],
                    row["to"],
                    float(row["relativity"]),
                    float(row["exposure"] or 0.0),
                )
                for row in frame.iter_rows(named=True)
            ]
            axes.append(
                VariableConfig(
                    type=axis_type,
                    table=rows,
                    other_label=stage[f"axis_{side}_other_label"] or None,
                )
            )
        cells_frame = pl.read_excel(path, sheet_name=stage["cells_sheet"])
        cells = [
            PairCellRow(
                axis_a_row=int(row["axis_a_row"]),
                axis_b_row=int(row["axis_b_row"]),
                relativity=float(row["relativity"]),
                row_count=int(row["row_count"] or 0),
                fitting_weight=float(row["fitting_weight"] or 0),
                weight_share=float(row["weight_share"] or 0),
                fallback_reason=row["fallback_reason"],
                exposure_total=(
                    None
                    if row["exposure_total"] is None
                    else float(row["exposure_total"])
                ),
            )
            for row in cells_frame.iter_rows(named=True)
        ]
        tables.append(
            PairTableConfig(
                stage_id=stage["stage_id"],
                parents=(stage["parent_a"], stage["parent_b"]),
                axes=(axes[0], axes[1]),
                cells=cells,
                provenance=json.loads(stage["provenance_json"] or "{}"),
            )
        )
    return tables


def variable_frame(cfg, *, fitted: list[Any] | None = None) -> pl.DataFrame:
    """The frame :func:`rate_model_tables` builds for one main effect.

    ``fitted`` is the same variable's rows in another version (the first
    snapshot: the pre-adjustment values) and becomes the ``fitted`` column.
    Taking a :class:`~easy_glm.engine.models.VariableConfig` rather than a whole
    model is what lets a page draw a *preview* of an edit (the relativity
    tooling) with exactly the columns the charts and the editor already use.
    """
    if cfg.type == "interaction":
        raise ValueError("variable_frame is for main effects; interactions differ")
    numeric = cfg.type in ("numeric", "linear")
    dtype = pl.Float64 if numeric else pl.Utf8
    cast = float if numeric else str
    froms = [None if r.from_ is None else cast(r.from_) for r in cfg.table]
    tos = [None if r.to_ is None else cast(r.to_) for r in cfg.table]

    columns: dict[str, Any] = {
        "from": pl.Series(froms, dtype=dtype),
        "to": pl.Series(tos, dtype=dtype),
        "label": [level_label(r, cfg.other_label) for r in cfg.table],
    }
    # ``fitted`` is the pre-adjustment value of each row (the first snapshot).
    if fitted is not None and len(fitted) == len(cfg.table):
        columns["fitted"] = pl.Series(
            [float(r.relativity) for r in fitted], dtype=pl.Float64
        )
    columns["relativity"] = pl.Series(
        [float(r.relativity) for r in cfg.table], dtype=pl.Float64
    )
    columns["exposure"] = pl.Series(
        [float(r.exposure) for r in cfg.table], dtype=pl.Float64
    )
    if cfg.type == "linear":
        columns["slope"] = pl.Series(
            [float(r.slope) for r in cfg.table], dtype=pl.Float64
        )
        columns["relativity_to"] = pl.Series(
            [float(r.relativity_to) for r in cfg.table], dtype=pl.Float64
        )
        # the row whose lower edge is x_base: the band starting there, or the
        # open "≥ hi" row when x_base is the upper clamp (a one-band term whose
        # exposure sits at the top of the range); from_rate_tables recovers
        # x_base from it
        columns["is_base"] = [
            cfg.x_base is not None
            and r.from_ is not None
            and float(r.from_) == float(cfg.x_base)
            for r in cfg.table
        ]
    return pl.DataFrame(columns)


def _interaction_frame(rm: RateModel, var: str, cfg) -> pl.DataFrame:
    """Long table of an interaction: parent edges, labels, exposure,
    [fitted], relativity — one row per cell."""
    a, b = cfg.parents
    dt_a = pl.Float64 if rm.variables[a].type in ("numeric", "linear") else pl.Utf8
    dt_b = pl.Float64 if rm.variables[b].type in ("numeric", "linear") else pl.Utf8
    rows: list[CellRow] = cfg.table
    others = (rm.variables[a].other_label, rm.variables[b].other_label)
    columns: dict[str, Any] = {
        "from_a": pl.Series([r.from_a for r in rows], dtype=dt_a),
        "to_a": pl.Series([r.to_a for r in rows], dtype=dt_a),
        "from_b": pl.Series([r.from_b for r in rows], dtype=dt_b),
        "to_b": pl.Series([r.to_b for r in rows], dtype=dt_b),
        "label": [level_label(r, others) for r in rows],
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
    if cfg.parents is None:
        raise ValueError(f"Interaction {var!r} has no parents recorded")
    a, b = cfg.parents
    rows_a = [
        level_label(r, rm.variables[a].other_label) for r in rm.variables[a].table
    ]
    rows_b = [
        level_label(r, rm.variables[b].other_label) for r in rm.variables[b].table
    ]
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
    pair_tables: list[PairTableConfig] | None = None,
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

        if pair_tables:
            manifest = wb.add_worksheet("Pair stages")
            used.add("pair stages")
            manifest.write_row(
                0,
                0,
                [
                    "order",
                    "stage_id",
                    "parent_a",
                    "parent_b",
                    "axis_a_sheet",
                    "axis_b_sheet",
                    "cells_sheet",
                    "axis_a_type",
                    "axis_b_type",
                    "axis_a_other_label",
                    "axis_b_other_label",
                    "provenance_json",
                ],
                bold,
            )
            pair_frames = pair_table_frames(RateModel(1.0, {}, pair_tables=pair_tables))
            for order, table in enumerate(pair_tables, start=1):
                a_sheet = sheet_name(f"Pair {order} A", used)
                b_sheet = sheet_name(f"Pair {order} B", used)
                cells_sheet = sheet_name(f"Pair {order} cells", used)
                variable_frame(table.axes[0]).write_excel(
                    workbook=wb, worksheet=a_sheet, autofit=True
                )
                variable_frame(table.axes[1]).write_excel(
                    workbook=wb, worksheet=b_sheet, autofit=True
                )
                pair_frames[table.stage_id].write_excel(
                    workbook=wb, worksheet=cells_sheet, autofit=True
                )
                manifest.write_row(
                    order,
                    0,
                    [
                        order,
                        table.stage_id,
                        *table.parents,
                        a_sheet,
                        b_sheet,
                        cells_sheet,
                        table.axes[0].type,
                        table.axes[1].type,
                        table.axes[0].other_label or "",
                        table.axes[1].other_label or "",
                        json.dumps(table.provenance, default=str),
                    ],
                )
                index_rows.extend(
                    [
                        (a_sheet, f"{table.stage_id} axis A", len(table.axes[0].table)),
                        (b_sheet, f"{table.stage_id} axis B", len(table.axes[1].table)),
                        (
                            cells_sheet,
                            f"{table.stage_id} cells",
                            len(table.axes[0].table) * len(table.axes[1].table),
                        ),
                    ]
                )

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
