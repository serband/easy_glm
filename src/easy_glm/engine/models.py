from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass
class ModelMetadata:
    model_type: str | None = None
    target: str | None = None
    weight_col: str | None = None
    exposure_col: str | None = None
    train_test_col: str | None = None
    predictor_variables: list[str] = field(default_factory=list)


@dataclass
class FromToRow:
    from_: float | str | None
    to_: float | str | None
    relativity: float


@dataclass
class VariableConfig:
    type: Literal["numeric", "categorical"]
    table: list[FromToRow]
    breakpoints: np.ndarray | None = None
    relativities: np.ndarray | None = None
    cat_map: dict[str, float] | None = None
    fallback: float = 1.0
    #: numeric only: relativity applied to null values, taken from an optional
    #: ``FromToRow(None, None, ...)`` row. ``None`` means nulls are an error.
    null_relativity: float | None = None


@dataclass
class Change:
    variable: str
    from_: Any
    to_: Any
    old_relativity: float
    new_relativity: float


@dataclass
class Snapshot:
    version: int
    description: str
    timestamp: str
    parent_version: int | None
    relativities: dict[str, list[FromToRow]]
    changes: list[Change] = field(default_factory=list)
    metrics: dict | None = None
    column_mapping: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionState:
    column_mapping: dict[str, str] = field(default_factory=dict)
    actual_formula: str = "sum_weighted"


def level_label(row: FromToRow) -> str:
    """Human-readable label for a ``FromToRow`` bin.

    ``None``-delimited ends mean open interval::

        < 18    (from_=None,  to_=18)
        ≥ 38    (from_=38,    to_=None)
        Other   (from_=None,  to_=None)
        North   (from_="North", to_="North")
        [18, 23) (from_=18, to_=23, unequal)
    """
    if row.from_ is None and row.to_ is None:
        return "Other / Unknown"
    if row.from_ is None:
        return f"< {row.to_}"
    if row.to_ is None:
        return f"≥ {row.from_}"
    if row.from_ == row.to_:
        return str(row.from_)
    return f"[{row.from_}, {row.to_})"


def level_labels(rows: list[FromToRow]) -> list[str]:
    """Convenience: ``[level_label(r) for r in rows]``."""
    return [level_label(r) for r in rows]
