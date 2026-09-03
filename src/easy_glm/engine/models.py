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
#: default name of a categorical encoder's catch-all bucket; when a real level
#: is called that, the encoder uses another name (``"Other (lumped)"``) and the
#: rate tables print *that* name instead of :data:`NULL_LABEL`
DEFAULT_OTHER_LABEL = "Other"


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
    #: True when the offset is the log of the **premium charged today** (the
    #: rate-change setup): the base rate is then the overall rate change and
    #: every relativity is a multiplier on the current premium, not a rate
    #: (the actuary's answer to Q6). Only changes labels, never a number.
    offset_is_premium: bool = False
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
class BandRow:
    """One row of a piecewise-linear (``"linear"``) table.

    ``relativity`` is the value at the band **start** (``from_``); inside the
    band the relativity is ``relativity * exp(slope * (x - from_))``. The two
    flat end rows ``(None, lo)`` / ``(hi, None)`` and the null row ``(None, None)``
    have ``slope == 0`` and a constant relativity. Curves are continuous at the
    interior edges: a band's end value equals the next band's ``relativity``.
    """

    from_: float | None
    to_: float | None
    relativity: float
    slope: float = 0.0

    def relativity_at(self, x: float) -> float:
        """Relativity at ``x`` inside this band (``x`` on the raw scale)."""
        if self.from_ is None or self.to_ is None or self.slope == 0.0:
            return self.relativity
        return float(self.relativity * np.exp(self.slope * (x - self.from_)))

    @property
    def relativity_to(self) -> float:
        """Relativity at the band end (equals the next band's start value)."""
        if self.from_ is None or self.to_ is None:
            return self.relativity
        return self.relativity_at(self.to_)


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


TableType = Literal["numeric", "categorical", "linear", "interaction"]


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
    #: categorical only: the encoder's name for the catch-all row when it is not
    #: the default one (``"Other (lumped)"`` when a real level is called
    #: "Other"); ``None`` prints the usual "Other / Unknown"
    other_label: str | None = None
    #: linear only (precomputed): per non-null row, the slope of log relativity
    #: and the x the row's relativity refers to
    slopes: np.ndarray | None = None
    starts: np.ndarray | None = None
    #: linear only: the x at which relativity is 1.00 (the base risk); None when
    #: the base row was the null row
    x_base: float | None = None


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


#: What one number in a rate table means, by model shape. The label goes on the
#: Excel ``Summary`` sheet, the Rate tables page and the Export page so a reader
#: never has to infer it from the family and the offset.
RELATIVITY_LABEL = "relativity"
ODDS_RELATIVITY_LABEL = "odds relativity"
PREMIUM_MULTIPLIER_LABEL = "multiplier on current premium"


def relativity_label(metadata: ModelMetadata) -> str:
    """Short name for one number of this model's rate tables."""
    if metadata.link == "logit":
        return ODDS_RELATIVITY_LABEL
    if metadata.offset_is_premium:
        return PREMIUM_MULTIPLIER_LABEL
    return RELATIVITY_LABEL


def relativity_note(metadata: ModelMetadata) -> str:
    """One sentence saying how to read this model's tables."""
    if metadata.link == "logit":
        return (
            "Odds relativities: the tables multiply the odds, not the "
            "probability. The base rate is the odds for the base risk and the "
            "scorer turns odds into a probability, so predictions are between "
            "0 and 1 and are never multiplied by exposure."
        )
    if metadata.offset_is_premium:
        premium = metadata.offset_col or "the current premium"
        return (
            "Multipliers on the current premium: this model was fitted with "
            f"{premium} (the log of the premium charged today) as an offset, so "
            "the base rate is the **overall** rate change and each relativity is "
            "the **differential** change for that band. 1.00 means that band's "
            "premium changes by the overall change and no more."
        )
    return (
        "Relativities: the base rate times one relativity per rating factor "
        "gives the predicted rate for a risk."
    )


def lumped_label(other_label: str | None) -> str | None:
    """The name to print on the catch-all row of a categorical table: the
    encoder's own ``other_label`` when it is not the default one (a real level
    is called "Other", so the bucket is "Other (lumped)"), else ``None``, which
    means the usual :data:`NULL_LABEL`."""
    if other_label is None or other_label == DEFAULT_OTHER_LABEL:
        return None
    return other_label


def _edge_label(lo: Any, hi: Any, null_label: str = NULL_LABEL) -> str:
    if lo is None and hi is None:
        return null_label
    if lo is None:
        return f"< {hi}"
    if hi is None:
        return f"≥ {lo}"
    if lo == hi:
        return str(lo)
    return f"[{lo}, {hi})"


def level_label(
    row: FromToRow | BandRow | CellRow,
    other_label: str | tuple[str | None, str | None] | None = None,
) -> str:
    """Human-readable label for a ``FromToRow`` bin, a ``BandRow`` band or a
    ``CellRow`` cell.

    ``None``-delimited ends mean open interval::

        < 18    (from_=None,  to_=18)
        ≥ 38    (from_=38,    to_=None)
        Other / Unknown  (from_=None, to_=None)
        North   (from_="North", to_="North")
        [18, 23) (from_=18, to_=23, unequal)
        [18, 23) | North   (a CellRow)

    ``other_label`` renames that catch-all row to the categorical encoder's own
    lumped-bucket name (see :func:`lumped_label`); for a ``CellRow`` pass a
    ``(label_a, label_b)`` pair, one per parent.
    """
    if isinstance(row, CellRow):
        la, lb = (
            other_label
            if isinstance(other_label, tuple)
            else (other_label, other_label)
        )
        return (
            f"{_edge_label(row.from_a, row.to_a, la or NULL_LABEL)} | "
            f"{_edge_label(row.from_b, row.to_b, lb or NULL_LABEL)}"
        )
    label = other_label if isinstance(other_label, str) else None
    return _edge_label(row.from_, row.to_, label or NULL_LABEL)


def level_labels(
    rows: list[FromToRow | BandRow | CellRow],
    other_label: str | tuple[str | None, str | None] | None = None,
) -> list[str]:
    """Convenience: ``[level_label(r, other_label) for r in rows]``."""
    return [level_label(r, other_label) for r in rows]
