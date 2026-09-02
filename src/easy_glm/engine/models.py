from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

#: separator in interaction variable names, e.g. ``"DrivAge×VehPower"`` — the one
#: definition; core and workflow import it from here
INTERACTION_SEP = "×"
#: label of the null (numeric) / Other (categorical) rate-table row, used by the
#: tables, the Excel export and every diagnostic so they are joinable by label
NULL_LABEL = "Other / Unknown"


@dataclass
class ModelMetadata:
    model_type: str | None = None
    target: str | None = None
    weight_col: str | None = None
    exposure_col: str | None = None
    train_test_col: str | None = None
    predictor_variables: list[str] = field(default_factory=list)
    #: offset column applied at scoring (linear-predictor scale when
    #: ``offset_is_log``); ``None`` = no offset
    offset_col: str | None = None
    offset_is_log: bool = True
    #: link function of the fitted GLM ("log" for multiplicative tables)
    link: str = "log"
    #: True when the GLM target was ``target / weight`` (e.g. claim counts with an
    #: exposure weight); ``None`` for files written before this was recorded
    divide_target_by_weight: bool | None = None


@dataclass
class FromToRow:
    from_: float | str | None
    to_: float | str | None
    relativity: float


@dataclass
class CellRow:
    """One cell of a two-way interaction table: the rate-table row of parent A
    (``from_a``/``to_a``), the row of parent B (``from_b``/``to_b``), the
    multiplicative adjustment and the training exposure of the cell."""

    from_a: float | str | None
    to_a: float | str | None
    from_b: float | str | None
    to_b: float | str | None
    relativity: float
    exposure: float = 0.0

    @property
    def key(self) -> tuple[Any, Any, Any, Any]:
        return (self.from_a, self.to_a, self.from_b, self.to_b)


TableType = Literal["numeric", "categorical", "interaction"]


@dataclass
class VariableConfig:
    type: TableType
    table: list[Any]  # list[FromToRow] for mains, list[CellRow] for interactions
    breakpoints: np.ndarray | None = None
    relativities: np.ndarray | None = None
    cat_map: dict[str, float] | None = None
    fallback: float = 1.0
    #: numeric only: relativity applied to null values, taken from an optional
    #: ``FromToRow(None, None, ...)`` row. ``None`` means nulls are an error.
    null_relativity: float | None = None
    #: interaction only: the two parent variables (both must be in the model)
    parents: tuple[str, str] | None = None
    #: interaction only (precomputed): relativity matrix over the parents' rows
    cell_matrix: np.ndarray | None = None
    #: categorical only (precomputed): level -> row index, in table order
    level_index: dict[str, int] | None = None


@dataclass
class Change:
    variable: str
    from_: Any
    to_: Any
    old_relativity: float
    new_relativity: float
    #: interaction cells: the second parent's row (``from_``/``to_`` are the first's)
    from_b: Any = None
    to_b: Any = None
    is_cell: bool = False


@dataclass
class Snapshot:
    version: int
    description: str
    timestamp: str
    parent_version: int | None
    relativities: dict[str, list[Any]]
    changes: list[Change] = field(default_factory=list)
    metrics: dict | None = None
    column_mapping: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionState:
    column_mapping: dict[str, str] = field(default_factory=dict)
    actual_formula: str = "sum_weighted"


def _edge_label(lo: Any, hi: Any) -> str:
    if lo is None and hi is None:
        return NULL_LABEL
    if lo is None:
        return f"< {hi}"
    if hi is None:
        return f"≥ {lo}"
    if lo == hi:
        return str(lo)
    return f"[{lo}, {hi})"


def level_label(row: FromToRow | CellRow) -> str:
    """Human-readable label for a ``FromToRow`` bin or a ``CellRow`` cell.

    ``None``-delimited ends mean open interval::

        < 18    (from_=None,  to_=18)
        ≥ 38    (from_=38,    to_=None)
        Other   (from_=None,  to_=None)
        North   (from_="North", to_="North")
        [18, 23) (from_=18, to_=23, unequal)
        [18, 23) | North   (a CellRow)
    """
    if isinstance(row, CellRow):
        return (
            f"{_edge_label(row.from_a, row.to_a)} | {_edge_label(row.from_b, row.to_b)}"
        )
    return _edge_label(row.from_, row.to_)


def level_labels(rows: list[FromToRow | CellRow]) -> list[str]:
    """Convenience: ``[level_label(r) for r in rows]``."""
    return [level_label(r) for r in rows]
