from __future__ import annotations

import copy
import json
import warnings
from collections.abc import Sequence
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ._scoring import (
    score_categorical,
    score_interaction,
    score_linear,
    score_numeric,
)
from .models import (
    INTERACTION_SEP,
    BandRow,
    CellRow,
    Change,
    FromToRow,
    ModelMetadata,
    Snapshot,
    VariableConfig,
    relativity_label,
    relativity_note,
)

_UNSET = object()

#: ``.easyglm`` file format version written by this release. Readers accept
#: older versions (migrating them) and refuse newer ones.
FORMAT_VERSION = 2

#: How each ``VariableConfig.type`` is scored. Unknown types are an error, never
#: silently treated as another type.
_SCORERS: dict[str, Any] = {
    "numeric": lambda col, cfg: score_numeric(col.to_numpy(), cfg),
    "categorical": score_categorical,
    "linear": lambda col, cfg: score_linear(col.cast(pl.Float64).to_numpy(), cfg),
}
#: table types whose rows are numeric bands (``from``/``to`` are floats)
_NUMERIC_TYPES = frozenset({"numeric", "linear"})
#: relative tolerance for the two rows of a linear table that share the node at
#: ``lo`` (the ``(None, lo)`` row and the first sloped band): they are one number
_NODE_RTOL = 1e-6
#: relative tolerance (on the band-end value) beyond which a ``slope`` column
#: that disagrees with the node values is reported; slopes are derived from the
#: nodes, so a rounded table still loads (a slope shown to 6 dp on a 5,000-wide
#: band is off by up to 0.25 % at the band end)
_SLOPE_CHECK_RTOL = 1e-2
#: table types this release can read and score ("interaction" needs two columns
#: and is dispatched separately in :meth:`RateModel.predict`)
KNOWN_TYPES = frozenset(_SCORERS) | {"interaction"}


def _row_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, BandRow):
        return {
            "from": row.from_,
            "to": row.to_,
            "relativity": row.relativity,
            "slope": row.slope,
            "exposure": row.exposure,
        }
    if isinstance(row, CellRow):
        return {
            "from_a": row.from_a,
            "to_a": row.to_a,
            "from_b": row.from_b,
            "to_b": row.to_b,
            "relativity": row.relativity,
            "exposure": row.exposure,
        }
    return {
        "from": row.from_,
        "to": row.to_,
        "relativity": row.relativity,
        "exposure": row.exposure,
    }


def _row_from_dict(r: dict[str, Any]) -> Any:
    if "slope" in r:
        return BandRow(
            r["from"],
            r["to"],
            r["relativity"],
            float(r.get("slope") or 0.0),
            float(r.get("exposure", 0.0) or 0.0),
        )
    if "from_a" in r:
        return CellRow(
            r["from_a"],
            r["to_a"],
            r["from_b"],
            r["to_b"],
            r["relativity"],
            float(r.get("exposure", 0.0) or 0.0),
        )
    return FromToRow(
        from_=r["from"],
        to_=r["to"],
        relativity=r["relativity"],
        exposure=float(r.get("exposure", 0.0) or 0.0),
    )


def _rows_from_list(rows: list[dict[str, Any]]) -> list[Any]:
    return [_row_from_dict(r) for r in rows]


def _split_interaction_name(name: str, variables: dict[str, Any]) -> tuple[str, str]:
    """``"A×B"`` -> ``("A", "B")`` where both are known main effects. Tries every
    split point so a main-effect name that itself contains the separator still
    resolves when the pair is unambiguous."""
    candidates = []
    idx = name.find(INTERACTION_SEP)
    while idx != -1:
        a, b = name[:idx], name[idx + len(INTERACTION_SEP) :]
        if a in variables and b in variables and a != b:
            candidates.append((a, b))
        idx = name.find(INTERACTION_SEP, idx + 1)
    if len(candidates) != 1:
        raise ValueError(
            f"Interaction table {name!r} must be named 'A{INTERACTION_SEP}B' where A "
            f"and B are main effects with their own tables (known: {sorted(variables)})"
            + ("; the name is ambiguous" if len(candidates) > 1 else "")
        )
    return candidates[0]


def _row_key(row: Any) -> tuple:
    return row.key if isinstance(row, CellRow) else (row.from_, row.to_)


def _is_null_row(row: Any) -> bool:
    return row.from_ is None and row.to_ is None


def _validate_linear_rows(name: str, rows: list[Any]) -> list[int]:
    """Check the shape of a linear table's rows and return their indices in
    curve order: the ``(None, lo)`` row, the sloped bands by ``from``, the
    ``(hi, None)`` row, then the null row. Raises for anything that is not a
    tiling of the line with two open flat ends and at most one null row."""
    for r in rows:
        if not isinstance(r, BandRow):
            raise ValueError(
                f"Linear table for {name!r}: every row needs a 'slope' (row "
                f"{(r.from_, r.to_)!r} has none); is this a step table saved as linear?"
            )
    null_idx = [i for i, r in enumerate(rows) if _is_null_row(r)]
    band_idx = [i for i, r in enumerate(rows) if not _is_null_row(r)]
    if len(null_idx) > 1:
        raise ValueError(
            f"Linear table for {name!r} has {len(null_idx)} rows with both "
            "'from' and 'to' empty; only one null row is allowed"
        )
    if len(band_idx) < 3:
        raise ValueError(
            f"Linear table for {name!r} needs an open lower band, at least one "
            "sloped band and an open upper band"
        )
    band_idx.sort(key=lambda i: (rows[i].from_ is not None, rows[i].from_ or 0.0))
    bands = [rows[i] for i in band_idx]
    if bands[0].from_ is not None or bands[-1].to_ is not None:
        raise ValueError(
            f"Linear table for {name!r} must start with an open lower band and "
            "end with an open upper band (the flat parts outside the clamp range)"
        )
    for a, b in zip(bands[:-1], bands[1:], strict=True):
        if a.to_ is None or b.from_ is None or a.to_ != b.from_:
            raise ValueError(
                f"Linear table for {name!r}: band ending at {a.to_!r} is followed "
                f"by a band starting at {b.from_!r}; bands must tile the line"
            )
    for b in bands[1:-1]:
        if not b.to_ > b.from_:
            raise ValueError(
                f"Linear table for {name!r}: band {b.from_!r}–{b.to_!r} has zero "
                "width; every sloped band needs from < to"
            )
    if bands[0].slope != 0.0 or bands[-1].slope != 0.0:
        raise ValueError(
            f"Linear table for {name!r}: the open end bands must have slope 0 "
            "(the curve is flat outside the clamp range)"
        )
    return band_idx + null_idx


def derive_slopes(bands: list[BandRow], which: Sequence[int] | None = None) -> None:
    """Set each sloped band's ``slope`` from its own start value and the next
    band's, in place. ``bands`` is a linear table's non-null rows in curve order
    (``(None, lo)`` first, ``(hi, None)`` last), as :func:`_validate_linear_rows`
    returns them; ``which`` limits the work to those band positions.

    The node values are the truth and the slopes follow from them, so a curve
    built this way is continuous by construction — at the clamp points too. The
    two open end bands keep slope 0 (the curve is flat outside the clamp range).
    """
    for i in range(1, len(bands) - 1) if which is None else which:
        b, nxt = bands[i], bands[i + 1]
        b.slope = (np.log(nxt.relativity) - np.log(b.relativity)) / (b.to_ - b.from_)


def _coerce_edge(value: Any, parent: VariableConfig) -> Any:
    """Bring an interaction table edge to the parent's key type (float for
    numeric bands, string for levels; None stays None)."""
    if value is None:
        return None
    if parent.type in _NUMERIC_TYPES:
        return float(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


_METADATA_FIELDS = {f.name for f in fields(ModelMetadata)}


def _warn_if_levels_unmatched(
    name: str, col: pl.Series, config: VariableConfig
) -> None:
    """Warn when most rows of a categorical fall to the Other row: usually the
    column arrived with a different type (4.0 vs "4") or renamed levels."""
    if col.len() == 0:
        return
    matched = col.cast(pl.Utf8).is_in(list(config.cat_map)).fill_null(False)
    share = 1.0 - matched.sum() / col.len()
    if share > 0.5:
        sample = col.drop_nulls().cast(pl.Utf8).unique().head(3).to_list()
        warnings.warn(
            f"Categorical variable {name!r}: {share:.0%} of rows matched none of its "
            f"{len(config.cat_map)} trained levels and were scored as Other. Values seen: "
            f"{sample}; trained levels e.g. {list(config.cat_map)[:3]}. Check the column's "
            "type (integer vs float vs text) and level names.",
            stacklevel=3,
        )


def _metadata_from_dict(raw: dict[str, Any] | None) -> ModelMetadata:
    """Build metadata tolerating missing (older files) and unknown (newer files) keys."""
    raw = raw or {}
    unknown = sorted(k for k in raw if k not in _METADATA_FIELDS)
    if unknown:
        warnings.warn(
            f"Ignoring unknown model metadata keys {unknown} (written by a newer "
            "easy_glm?); they will not be written back",
            stacklevel=3,
        )
    known = {k: v for k, v in raw.items() if k in _METADATA_FIELDS}
    if "predictor_variables" in known and known["predictor_variables"] is None:
        known["predictor_variables"] = []
    return ModelMetadata(**known)


class RateModel:
    def __init__(
        self,
        base_rate: float,
        variables: dict[str, VariableConfig],
        metadata: ModelMetadata | None = None,
        snapshots: list[Snapshot] | None = None,
        current_version: int = 0,
        column_mapping: dict[str, str] | None = None,
    ):
        """``column_mapping`` maps *dataset* column names to the model's variable
        names, e.g. ``{"driver_age": "DrivAge"}``; ``predict`` renames before scoring.
        """
        self.base_rate = base_rate
        self.variables = variables
        self.metadata = metadata or ModelMetadata()
        self.snapshots = snapshots or []
        self.current_version = current_version
        self._pending_changes: list[Change] = []
        self.column_mapping = column_mapping or {}

    # -- what one number in these tables means ---------------------------
    @property
    def relativity_label(self) -> str:
        """Short name for one number of these tables: ``"relativity"``,
        ``"odds relativity"`` (logit link) or ``"multiplier on current premium"``
        (the offset is the log of today's premium)."""
        return relativity_label(self.metadata)

    @property
    def relativity_note(self) -> str:
        """One sentence saying how to read these tables (see
        :attr:`relativity_label`)."""
        return relativity_note(self.metadata)

    @classmethod
    def from_rate_tables(
        cls,
        tables: dict[str, pl.DataFrame],
        base_rate: float,
        *,
        model_type: str | None = None,
        target: str | None = None,
        weight_col: str | None = None,
        exposure_col: str | None = None,
        train_test_col: str | None = None,
        offset_col: str | None = None,
        offset_is_log: bool = True,
        offset_is_premium: bool = False,
        link: str = "log",
        divide_target_by_weight: bool | None = None,
        predictor_variables: list[str] | None = None,
    ) -> RateModel:
        """Build a RateModel from per-variable relativity tables.

        ``tables`` maps a variable name to a frame with columns ``from``, ``to``
        and ``relativity`` (``label`` and other columns are ignored) — the shape
        produced by :func:`easy_glm.rate_tables`, :func:`easy_glm.core.excel.
        rate_model_tables` and the Excel export. Numeric variables have numeric
        ``from``/``to`` (null = open end); categorical variables have string
        ``from == to`` per level (a numeric-typed table whose rows all have
        ``from == to`` is treated as an integer-coded categorical). In both cases
        a row with ``from`` and ``to`` both null is the null / Other row. When it
        is absent, categoricals get an Other row at 1.0 (with a warning) and
        numerics get no null row, so nulls raise at scoring time. Numeric bands may
        be listed in any order; they must tile the whole line.

        An **interaction** table is keyed ``"A×B"`` (the parents' names joined by
        ``×``; both parents must have their own tables) and has columns
        ``from_a``, ``to_a``, ``from_b``, ``to_b``, ``relativity`` and optionally
        ``exposure``; every row must name a cell of the parents' rows, and cells
        that are not listed adjust by 1.0.

        A **piecewise-linear** table is recognised by its ``slope`` column (a
        table without one is a step table; one that still has ``relativity_to``
        but lost ``slope`` is refused rather than read as steps). ``relativity``
        is the value at the band start and the relativity inside the band is
        ``relativity * exp(slope * (x - from))``. The first and last bands are
        open and flat, and the ``(None, lo)`` row shows the same value as the
        first sloped band (they are one node). The slopes are **derived from
        the consecutive start values** (the last one towards the ``(hi, None)``
        value), so a table rounded to a few decimals reads back as a continuous
        curve; a supplied ``slope`` that disagrees with those values by more
        than 1 % (on the band-end value) is reported with a warning. Zero-width
        bands are refused. ``x_base`` is recovered from an ``is_base`` column or,
        failing that, from the unique row with relativity 1.0.
        """
        pred_vars = predictor_variables or list(tables)
        for var in pred_vars:
            if var not in tables:
                raise KeyError(f"No table for variable {var!r}")
        is_cell = {v: "from_a" in tables[v].columns for v in pred_vars}
        variables: dict[str, VariableConfig] = {}
        for var in pred_vars:  # mains first: interactions need their parents
            if is_cell[var]:
                continue
            cols = set(tables[var].columns)
            if "slope" in cols:
                variables[var] = cls._config_from_linear_table(var, tables[var])
            elif "relativity_to" in cols:
                raise ValueError(
                    f"Table for {var!r} has a 'relativity_to' column but no 'slope' "
                    "column: a piecewise-linear table needs its slopes (or drop "
                    "'relativity_to' to read it as a step table on purpose)"
                )
            else:
                variables[var] = cls._config_from_table(var, tables[var])
        for var in pred_vars:
            if is_cell[var]:
                variables[var] = cls._config_from_cell_table(
                    var, tables[var], variables
                )
        cls._precompute_variables(variables)
        metadata = ModelMetadata(
            model_type=model_type,
            target=target,
            weight_col=weight_col,
            exposure_col=exposure_col,
            train_test_col=train_test_col,
            predictor_variables=list(variables),
            offset_col=offset_col,
            offset_is_log=offset_is_log,
            offset_is_premium=offset_is_premium,
            link=link,
            divide_target_by_weight=divide_target_by_weight,
        )
        rm = cls(base_rate=base_rate, variables=variables, metadata=metadata)
        rm.create_snapshot("Base model")
        return rm

    @classmethod
    def from_glm_model(cls, fit: Any, **kwargs: Any) -> RateModel:
        """Compile a fitted :class:`easy_glm.GLMFit` into a RateModel that
        reproduces it exactly (thin wrapper around :func:`easy_glm.to_rate_model`;
        keyword arguments are forwarded)."""
        from easy_glm.core.tables import to_rate_model

        return to_rate_model(fit, **kwargs)

    @staticmethod
    def _config_from_table(name: str, table: pl.DataFrame) -> VariableConfig:
        missing = {"from", "to", "relativity"} - set(table.columns)
        if missing:
            raise ValueError(f"Table for {name!r} lacks columns {sorted(missing)}")
        if "exposure" in table.columns:
            table = table.with_columns(pl.col("exposure").fill_null(0.0))
        else:
            table = table.with_columns(pl.lit(0.0).alias("exposure"))
        triples = list(table.select("from", "to", "relativity", "exposure").iter_rows())
        if any(rel is None for _, _, rel, _ in triples):
            raise ValueError(f"Table for {name!r} has a null relativity")
        null_rows = [t for t in triples if t[0] is None and t[1] is None]
        if len(null_rows) > 1:
            raise ValueError(
                f"Table for {name!r} has {len(null_rows)} rows with both 'from' and "
                "'to' empty; only one null / Other row is allowed"
            )
        body = [t for t in triples if not (t[0] is None and t[1] is None)]
        numeric_dtype = table["from"].dtype.is_numeric()
        # Integer-coded categoricals (VehPower 4..12 typed as numbers) have
        # from == to on every row; numeric bands never do.
        coded_categorical = (
            numeric_dtype
            and bool(body)
            and all(
                lo is not None and hi is not None and lo == hi for lo, hi, _, _ in body
            )
        )
        numeric = numeric_dtype and not coded_categorical

        def _level(v: Any) -> str:
            if isinstance(v, float) and v.is_integer():
                return str(int(v))
            return str(v)

        if numeric:
            bands = [
                FromToRow(
                    None if lo is None else float(lo),
                    None if hi is None else float(hi),
                    float(rel),
                    float(expo),
                )
                for lo, hi, rel, expo in body
            ]
            if not bands:
                raise ValueError(f"Numeric table for {name!r} has no bands")
            # accept rows in any order: sort by the lower edge, open end first
            bands.sort(key=lambda r: (r.from_ is not None, r.from_ or 0.0))
            if bands[0].from_ is not None:
                raise ValueError(
                    f"Numeric table for {name!r} must start with an open lower band "
                    "(a row whose 'from' is empty covers everything below the first edge)"
                )
            if bands[-1].to_ is not None:
                raise ValueError(
                    f"Numeric table for {name!r} must end with an open upper band "
                    "(a row whose 'to' is empty covers everything from the last edge up)"
                )
            for a, b in zip(bands[:-1], bands[1:], strict=True):
                if a.to_ is None or b.from_ is None or a.to_ != b.from_:
                    raise ValueError(
                        f"Numeric table for {name!r}: band ending at {a.to_!r} is "
                        f"followed by a band starting at {b.from_!r}; bands must "
                        "tile the line with no gaps or overlaps"
                    )
            rows = bands + [
                FromToRow(None, None, float(rel), float(expo))
                for _, _, rel, expo in null_rows
            ]
            return VariableConfig(type="numeric", table=rows)

        rows = [
            FromToRow(
                _level(lo),
                _level(hi if hi is not None else lo),
                float(rel),
                float(expo),
            )
            for lo, hi, rel, expo in body
        ]
        if any(r.from_ != r.to_ for r in rows):
            raise ValueError(
                f"Categorical table for {name!r} must have 'from' == 'to' on every "
                "level row"
            )
        levels = [r.from_ for r in rows]
        dupes = sorted({lv for lv in levels if levels.count(lv) > 1})
        if dupes:
            raise ValueError(
                f"Table for {name!r} lists level(s) {dupes} more than once"
            )
        if null_rows:
            rows.append(
                FromToRow(None, None, float(null_rows[0][2]), float(null_rows[0][3]))
            )
        else:
            warnings.warn(
                f"Table for {name!r} has no Other row (from and to both empty); "
                "unseen levels and nulls will score at 1.0",
                stacklevel=3,
            )
            rows.append(FromToRow(None, None, 1.0))
        return VariableConfig(type="categorical", table=rows)

    @staticmethod
    def _config_from_linear_table(name: str, table: pl.DataFrame) -> VariableConfig:
        missing = {"from", "to", "relativity", "slope"} - set(table.columns)
        if missing:
            raise ValueError(
                f"Linear table for {name!r} lacks columns {sorted(missing)}"
            )
        has_base = "is_base" in table.columns
        has_exposure = "exposure" in table.columns
        cols = (
            ["from", "to", "relativity", "slope"]
            + (["is_base"] if has_base else [])
            + (["exposure"] if has_exposure else [])
        )
        rows: list[BandRow] = []
        given_slopes: list[float | None] = []
        base_flags: list[bool] = []
        for rec in table.select(cols).iter_rows():
            lo, hi, rel, slope = rec[:4]
            if rel is None:
                raise ValueError(f"Linear table for {name!r} has a null relativity")
            if float(rel) <= 0:
                raise ValueError(
                    f"Linear table for {name!r} has a non-positive relativity "
                    f"({rel}); a log-linear band needs relativities > 0"
                )
            rows.append(
                BandRow(
                    None if lo is None else float(lo),
                    None if hi is None else float(hi),
                    float(rel),
                    0.0,
                    float(rec[-1] or 0.0) if has_exposure else 0.0,
                )
            )
            given_slopes.append(None if slope is None else float(slope))
            base_flags.append(bool(rec[4]) if has_base else False)
        order = _validate_linear_rows(name, rows)
        bands = [rows[i] for i in order if not _is_null_row(rows[i])]
        null_rows = [rows[i] for i in order if _is_null_row(rows[i])]
        if not null_rows:
            warnings.warn(
                f"Linear table for {name!r} has no null row (both 'from' and 'to' "
                "empty); missing values will raise at scoring time",
                stacklevel=3,
            )
        # the (None, lo) row and the first sloped band are one node
        v_lo, v_first = bands[0].relativity, bands[1].relativity
        if abs(v_lo - v_first) > _NODE_RTOL * abs(v_first):
            raise ValueError(
                f"Linear table for {name!r} is not continuous at {bands[1].from_!r}: "
                f"the '< {bands[1].from_:g}' row shows {v_lo:.6g} but the first band "
                f"starts at {v_first:.6g}; they are the same point of the curve"
            )
        bands[0].relativity = v_first
        # slopes come from the nodes: continuity is then by construction
        derive_slopes(bands)
        for i, b in zip(order, (rows[i] for i in order), strict=True):
            given = given_slopes[i]
            if given is None or _is_null_row(b) or b.from_ is None or b.to_ is None:
                if given and not _is_null_row(b):
                    raise ValueError(
                        f"Linear table for {name!r}: the open end bands must have "
                        "slope 0 (the curve is flat outside the clamp range)"
                    )
                continue
            end_given = b.relativity * np.exp(given * (b.to_ - b.from_))
            end_used = b.relativity_to
            if abs(end_given - end_used) > _SLOPE_CHECK_RTOL * abs(end_used):
                warnings.warn(
                    f"Linear table for {name!r}, band {b.from_:g}–{b.to_:g}: the "
                    f"slope column ({given:.6g}) would end the band at "
                    f"{end_given:.6g} but the next row starts at {end_used:.6g}; "
                    "the slope derived from the row values is used",
                    stacklevel=3,
                )
        x_base = None
        flagged = [rows[i] for i in order if base_flags[i]]
        if len(flagged) == 1 and not _is_null_row(flagged[0]):
            b = flagged[0]
            x_base = b.from_ if b.from_ is not None else b.to_
        elif not flagged:
            ones = [b for b in bands if abs(b.relativity - 1.0) <= 1e-12]
            if len(ones) == 1:
                b = ones[0]
                x_base = b.from_ if b.from_ is not None else b.to_
            elif len(ones) == 2 and ones[0] is bands[0] and ones[1] is bands[1]:
                x_base = bands[1].from_
        return VariableConfig(type="linear", table=bands + null_rows, x_base=x_base)

    @staticmethod
    def _config_from_cell_table(
        name: str, table: pl.DataFrame, variables: dict[str, VariableConfig]
    ) -> VariableConfig:
        needed = {"from_a", "to_a", "from_b", "to_b", "relativity"}
        missing = needed - set(table.columns)
        if missing:
            raise ValueError(
                f"Interaction table for {name!r} lacks columns {sorted(missing)}"
            )
        a, b = _split_interaction_name(name, variables)
        has_exposure = "exposure" in table.columns
        rows: list[CellRow] = []
        seen: set[tuple] = set()
        for r in table.iter_rows(named=True):
            if r["relativity"] is None:
                raise ValueError(
                    f"Interaction table for {name!r} has a null relativity"
                )
            cell = CellRow(
                _coerce_edge(r["from_a"], variables[a]),
                _coerce_edge(r["to_a"], variables[a]),
                _coerce_edge(r["from_b"], variables[b]),
                _coerce_edge(r["to_b"], variables[b]),
                float(r["relativity"]),
                float(r["exposure"] or 0.0) if has_exposure else 0.0,
            )
            if cell.key in seen:
                raise ValueError(
                    f"Interaction table for {name!r} lists cell {cell.key} twice"
                )
            seen.add(cell.key)
            rows.append(cell)
        cfg = VariableConfig(type="interaction", table=rows, parents=(a, b))
        RateModel._precompute_interaction(name, cfg, variables)  # validates cells
        return cfg

    def predict(
        self,
        data: pl.DataFrame,
        *,
        version: int | None = None,
        column_map: dict[str, str] | None = None,
        exposure_col: str | None = _UNSET,
    ) -> np.ndarray:
        if version is not None and version != self.current_version:
            saved_version = self.current_version
            try:
                self.switch_to(version)
                return self.predict(
                    data,
                    column_map=column_map,
                    exposure_col=exposure_col,
                )
            finally:
                self.switch_to(saved_version)

        mapping = column_map or self.column_mapping
        if mapping:
            rename = {old: new for old, new in mapping.items() if old in data.columns}
            if rename:
                data = data.rename(rename)

        result = np.full(len(data), self.base_rate, dtype=float)

        for name, config in self.variables.items():
            if config.type == "interaction":
                for parent in config.parents or ():
                    if parent not in data.columns:
                        raise ValueError(f"Column '{parent}' not found in data")
                result *= score_interaction(data, config, self.variables)
                continue
            if name not in data.columns:
                raise ValueError(f"Column '{name}' not found in data")

            col = data[name]
            scorer = _SCORERS.get(config.type)
            if scorer is None:
                raise ValueError(
                    f"Variable {name!r} has table type {config.type!r}, which this "
                    f"version of easy_glm cannot score (known: {sorted(KNOWN_TYPES)}). "
                    "The model file probably comes from a newer easy_glm."
                )
            rel = scorer(col, config)
            if (
                config.type == "categorical"
                and config.cat_map
                and len(config.cat_map) > 1
            ):
                _warn_if_levels_unmatched(name, col, config)
            result *= rel

        result = self._apply_offset(result, data)
        result = self._response(result)
        result = self._apply_exposure(result, data, exposure_col)

        return result

    def _response(self, product: np.ndarray) -> np.ndarray:
        """Turn the product of the base rate and the relativities into the
        prediction the model makes.

        With the log link the product *is* the prediction. With the **logit**
        link (a binomial model) the tables are odds relativities, so the product
        is the odds and the prediction is ``odds / (1 + odds)`` — a probability
        strictly between 0 and 1, exactly what the GLM predicts.
        """
        if self.metadata.link != "logit":
            return product
        return product / (1.0 + product)

    def _apply_offset(self, result: np.ndarray, data: pl.DataFrame) -> np.ndarray:
        """Multiply by ``exp(offset)`` (or the raw column) when the fit used an
        offset. It applies on the linear-predictor scale, so for a logit model
        it multiplies the **odds**, before they become a probability."""
        name = self.metadata.offset_col
        if not name:
            return result
        if name not in data.columns:
            warnings.warn(
                f"Offset column '{name}' not found in data — predictions exclude "
                "the offset and will not match the fitted GLM",
                stacklevel=2,
            )
            return result
        offset = data[name].cast(pl.Float64).to_numpy()
        return result * (np.exp(offset) if self.metadata.offset_is_log else offset)

    def _apply_exposure(
        self,
        result: np.ndarray,
        data: pl.DataFrame,
        override_col: str | None | object,
    ) -> np.ndarray:
        exposure_name = (
            override_col if override_col is not _UNSET else self.metadata.exposure_col
        )
        if exposure_name is None:
            return result
        if self.metadata.link == "logit":
            # a probability is not an amount: multiplying it by exposure would
            # produce a number between 0 and the exposure that means nothing
            raise ValueError(
                f"This model predicts a probability (logit link), which cannot be "
                f"multiplied by an exposure column ({exposure_name!r}). Score it "
                "with exposure_col=None and multiply outside if you need counts."
            )
        if exposure_name not in data.columns:
            warnings.warn(
                f"Exposure column '{exposure_name}' not found in data "
                f"— predictions not multiplied by exposure",
                stacklevel=2,
            )
            return result
        return result * data[exposure_name].to_numpy()

    def update_relativity(
        self,
        var: str,
        from_: Any,
        to_: Any,
        new_value: float,
        from_b: Any = _UNSET,
        to_b: Any = _UNSET,
    ) -> None:
        """Set one row's relativity. For an interaction pass the cell as
        ``(from_, to_)`` = parent A's row and ``(from_b, to_b)`` = parent B's row.
        The change is recorded for the next snapshot."""
        if var not in self.variables:
            raise KeyError(f"Variable '{var}' not found")

        config = self.variables[var]
        if config.type == "interaction":
            if from_b is _UNSET or to_b is _UNSET:
                raise ValueError(
                    f"'{var}' is an interaction: pass from_b= and to_b= for the "
                    "second variable's row"
                )
            key = (from_, to_, from_b, to_b)
            for row in config.table:
                if row.key == key:
                    old_value = row.relativity
                    row.relativity = float(new_value)
                    self._pending_changes.append(
                        Change(
                            variable=var,
                            from_=from_,
                            to_=to_,
                            old_relativity=old_value,
                            new_relativity=float(new_value),
                            from_b=from_b,
                            to_b=to_b,
                            is_cell=True,
                        )
                    )
                    config.cell_matrix = None
                    self._precompute_interaction(var, config, self.variables)
                    return
            raise ValueError(f"No cell {key!r} in interaction '{var}'")

        if config.type == "linear":
            self._update_linear(var, config, from_, to_, float(new_value))
            return

        for row in config.table:
            if row.from_ == from_ and row.to_ == to_:
                old_value = row.relativity
                row.relativity = new_value
                self._pending_changes.append(
                    Change(
                        variable=var,
                        from_=from_,
                        to_=to_,
                        old_relativity=old_value,
                        new_relativity=new_value,
                    )
                )
                config.breakpoints = None
                config.relativities = None
                config.cat_map = None
                config.fallback = 1.0
                config.null_relativity = None
                config.level_index = None
                self._precompute_variables({var: config})
                # interactions built on this variable index its rows by position,
                # which an edit does not change; nothing to rebuild there.
                return

        raise ValueError(
            f"No row found with from={from_!r}, to={to_!r} in variable '{var}'"
        )

    def _update_linear(
        self, var: str, config: VariableConfig, from_: Any, to_: Any, new_value: float
    ) -> None:
        """Node-edit rule for piecewise-linear tables.

        Every row's ``relativity`` is a *node* of the curve: the ``(None, lo)``
        row and the first sloped band share the node at ``lo`` (editing either
        sets both), each other sloped band owns the node at its start, and the
        ``(hi, None)`` row is the node at ``hi``. An edit moves that node and
        re-derives the slopes of the bands on either side of it, so the curve
        stays continuous everywhere: one slope changes at the two ends, two in
        the interior. The null row is not on the curve and edits as a step.
        """
        if not new_value > 0:
            raise ValueError(
                f"Linear variable '{var}': relativities must be > 0, got {new_value}"
            )
        rows = config.table
        idx = next(
            (i for i, r in enumerate(rows) if r.from_ == from_ and r.to_ == to_), None
        )
        if idx is None:
            raise ValueError(
                f"No row found with from={from_!r}, to={to_!r} in variable '{var}'"
            )
        row = rows[idx]
        old_value = row.relativity
        row.relativity = new_value
        if not _is_null_row(row):
            order = _validate_linear_rows(var, rows)
            bands = [rows[i] for i in order if not _is_null_row(rows[i])]
            pos = bands.index(row)
            if pos <= 1:  # the node at lo is shared by the first two rows
                bands[0].relativity = bands[1].relativity = new_value
                pos = 1

            def _rederive(i: int) -> None:  # slope of sloped band i from its nodes
                derive_slopes(bands, [i])

            if pos < len(bands) - 1:  # a sloped band: its own slope
                _rederive(pos)
            if pos >= 2:  # the band ending at this node
                _rederive(pos - 1)
        self._pending_changes.append(
            Change(
                variable=var,
                from_=from_,
                to_=to_,
                old_relativity=old_value,
                new_relativity=new_value,
            )
        )
        config.breakpoints = None
        config.relativities = None
        config.slopes = None
        config.starts = None
        config.null_relativity = None
        self._precompute_variables({var: config})

    @property
    def non_constant_variables(self) -> dict[str, VariableConfig]:
        """Variables whose relativities are not all equal.

        Returns only variables with at least two distinct relativity values
        (within 5-decimal tolerance). Variables where all bins have the same
        relativity (e.g. all 1.0) are excluded — they contribute no signal.
        """
        result: dict[str, VariableConfig] = {}
        for name, config in self.variables.items():
            rels = [r.relativity for r in config.table]
            if len({round(r, 5) for r in rels}) > 1:
                result[name] = config
        return result

    def compute_ae_for_variable(
        self,
        data: pl.DataFrame,
        variable: str,
        formula: str = "sum_weighted",
    ) -> dict:
        """Compute actual vs expected metrics for a single variable.

        The data is split by train/test if ``train_test_col`` is present
        in metadata and the column exists in ``data``. Otherwise all data
        is used as a single split.

        Parameters
        ----------
        data : pl.DataFrame
            Dataset containing the target, weight, variable, and optionally
            train/test columns.
        variable : str
            Variable name (must exist in ``self.variables``).
        formula : str
            One of ``"sum_weighted"``, ``"sum_unweighted"``,
            ``"sum_over_weight"``.

        Returns
        -------
        dict
            Keys: ``"variable"``, ``"subsets"`` (dict of ``"train"`` / ``"test"`` / ``"all"``).
            Each subset value is a list of per-bin dicts with keys
            ``"level"``, ``"actual"``, ``"expected"``, ``"exposure"``.
        """
        from easy_glm.ui.metrics import compute_actual_expected

        return compute_actual_expected(self, data, variable, formula=formula)

    def create_snapshot(
        self, description: str, metrics: dict[str, Any] | None = None
    ) -> int:
        """Freeze the current relativities as a new version. ``metrics`` (e.g. the
        train/holdout A/E, Gini and deviance of this version) is stored with it."""
        version = len(self.snapshots) + 1
        parent = self.current_version if self.snapshots else None

        relativities = {
            name: copy.deepcopy(config.table) for name, config in self.variables.items()
        }

        metadata_dict = asdict(self.metadata)

        snapshot = Snapshot(
            version=version,
            description=description,
            timestamp=datetime.now(timezone.utc).isoformat(),
            parent_version=parent,
            relativities=relativities,
            changes=list(self._pending_changes),
            column_mapping=dict(self.column_mapping),
            metadata=metadata_dict,
            metrics=copy.deepcopy(metrics) if metrics is not None else None,
        )
        self.snapshots.append(snapshot)
        self.current_version = version
        self._pending_changes.clear()
        return version

    def set_snapshot_metrics(
        self, metrics: dict[str, Any], version: int | None = None
    ) -> None:
        """Attach ``metrics`` to a snapshot (default: the current version)."""
        version = self.current_version if version is None else version
        if version < 1 or version > len(self.snapshots):
            raise ValueError(f"Invalid version: {version}")
        self.snapshots[version - 1].metrics = copy.deepcopy(metrics)

    def switch_to(self, version: int) -> None:
        if version < 1 or version > len(self.snapshots):
            raise ValueError(f"Invalid version: {version}")
        snapshot = self.snapshots[version - 1]
        for name, table in snapshot.relativities.items():
            self.variables[name].table = copy.deepcopy(table)
            self.variables[name].breakpoints = None
            self.variables[name].relativities = None
            self.variables[name].cat_map = None
            self.variables[name].fallback = 1.0
            self.variables[name].null_relativity = None
            self.variables[name].level_index = None
            self.variables[name].cell_matrix = None
            self.variables[name].slopes = None
            self.variables[name].starts = None
        RateModel._precompute_variables(self.variables)
        self.column_mapping = dict(snapshot.column_mapping)
        if snapshot.metadata:
            self.metadata = _metadata_from_dict(snapshot.metadata)
        self.current_version = version

    def clone(self) -> RateModel:
        """Create an independent deep copy of this RateModel.

        The clone shares no mutable references with the original.
        Mutations to the clone's relativities or snapshots will never
        affect the original, and vice versa.

        Returns
        -------
        RateModel
            A fully independent copy.
        """
        return self._from_dict(self._to_dict())

    def list_snapshots(self) -> list[dict[str, Any]]:
        return [
            {
                "version": s.version,
                "description": s.description,
                "timestamp": s.timestamp,
                "parent_version": s.parent_version,
                "changes_count": len(s.changes),
            }
            for s in self.snapshots
        ]

    def diff(self, v1: int, v2: int, *, tol: float = 1e-12) -> list[Change]:
        """Relativities that differ between snapshot ``v1`` and snapshot ``v2``.

        One :class:`Change` per (variable, row) whose relativity moved by more
        than ``tol`` (``old_relativity`` is NaN for rows absent from ``v1``).
        """
        for v in (v1, v2):
            if v < 1 or v > len(self.snapshots):
                raise ValueError(f"Invalid version: {v}")
        before = self.snapshots[v1 - 1].relativities
        after = self.snapshots[v2 - 1].relativities
        out: list[Change] = []
        for var, rows in after.items():
            old_rows = {_row_key(r): r.relativity for r in before.get(var, [])}
            for r in rows:
                old = old_rows.get(_row_key(r))
                if old is None or abs(old - r.relativity) > tol:
                    is_cell = isinstance(r, CellRow)
                    out.append(
                        Change(
                            variable=var,
                            from_=r.from_a if is_cell else r.from_,
                            to_=r.to_a if is_cell else r.to_,
                            old_relativity=float("nan") if old is None else old,
                            new_relativity=r.relativity,
                            from_b=r.from_b if is_cell else None,
                            to_b=r.to_b if is_cell else None,
                            is_cell=is_cell,
                        )
                    )
        return out

    def to_json(self, path: str | Path) -> None:
        data = self._to_dict()
        path = Path(path)
        path.write_text(json.dumps(data, indent=2, default=str))

    def to_excel(self, path: str | Path) -> Path:
        """Write the current relativities to an ``.xlsx`` workbook: a ``Summary``
        sheet (base rate, metadata, version) plus one sheet per variable with
        ``from`` / ``to`` / ``label`` / ``relativity``."""
        from easy_glm.core.excel import (
            interaction_matrices,
            rate_model_tables,
            write_rate_tables_xlsx,
        )

        summary: dict[str, Any] = {
            "tables": f"current {self.relativity_label} per band "
            "(manual adjustments included)",
            "how to read them": self.relativity_note,
            "base_rate": self.base_rate,
            "model_type": self.metadata.model_type,
            "link": self.metadata.link,
            "target": self.metadata.target,
            "target divided by weight": self.metadata.divide_target_by_weight,
            "weight_col": self.metadata.weight_col,
            "exposure_col": self.metadata.exposure_col,
            "offset_col": self.metadata.offset_col,
            "train_test_col": self.metadata.train_test_col,
            "predictors": list(self.variables),
            **{
                f"x_base ({name})": cfg.x_base
                for name, cfg in self.variables.items()
                if cfg.type == "linear" and cfg.x_base is not None
            },
            "version": self.current_version,
            "snapshots": len(self.snapshots),
        }
        matrices = {
            name: (*cfg.parents, *interaction_matrices(self, name))
            for name, cfg in self.variables.items()
            if cfg.type == "interaction"
        }
        return write_rate_tables_xlsx(
            rate_model_tables(self), path, summary=summary, matrices=matrices or None
        )

    @classmethod
    def from_json(cls, path: str | Path) -> RateModel:
        raw = json.loads(Path(path).read_text())
        return cls._from_dict(raw)

    def _to_dict(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "metadata": asdict(self.metadata),
            "base_rate": self.base_rate,
            "current_version": self.current_version,
            "column_mapping": {str(k): str(v) for k, v in self.column_mapping.items()},
            "variables": {
                name: {
                    "type": config.type,
                    **(
                        {"parents": list(config.parents)}
                        if config.type == "interaction" and config.parents
                        else {}
                    ),
                    **(
                        {"x_base": config.x_base}
                        if config.type == "linear" and config.x_base is not None
                        else {}
                    ),
                    **(
                        {"other_label": config.other_label}
                        if config.type == "categorical" and config.other_label
                        else {}
                    ),
                    "table": [_row_to_dict(row) for row in config.table],
                }
                for name, config in self.variables.items()
            },
            "snapshots": [
                {
                    "version": s.version,
                    "description": s.description,
                    "timestamp": s.timestamp,
                    "parent_version": s.parent_version,
                    "column_mapping": {
                        str(k): str(v) for k, v in s.column_mapping.items()
                    },
                    "metadata": s.metadata,
                    "relativities": {
                        name: [_row_to_dict(row) for row in table]
                        for name, table in s.relativities.items()
                    },
                    "changes": [
                        {
                            "variable": c.variable,
                            "from": c.from_,
                            "to": c.to_,
                            "old_relativity": c.old_relativity,
                            "new_relativity": c.new_relativity,
                            **(
                                {"from_b": c.from_b, "to_b": c.to_b, "is_cell": True}
                                if c.is_cell
                                else {}
                            ),
                        }
                        for c in s.changes
                    ],
                    "metrics": s.metrics,
                }
                for s in self.snapshots
            ],
        }

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> RateModel:
        version = int(raw.get("format_version") or 1)
        if version > FORMAT_VERSION:
            raise ValueError(
                f"This .easyglm file is format version {version}; this easy_glm "
                f"reads up to version {FORMAT_VERSION}. Upgrade easy_glm to open it."
            )
        if version < FORMAT_VERSION:
            raw = cls._migrate(raw, version)
        variables: dict[str, VariableConfig] = {}
        for name, vdata in raw["variables"].items():
            if vdata["type"] not in KNOWN_TYPES:
                raise ValueError(
                    f"Variable {name!r} has table type {vdata['type']!r}, which this "
                    f"version of easy_glm cannot score (known: {sorted(KNOWN_TYPES)}). "
                    "The model file probably comes from a newer easy_glm."
                )
            table = _rows_from_list(vdata["table"])
            if vdata["type"] == "linear":
                table = [table[i] for i in _validate_linear_rows(name, table)]
            parents = vdata.get("parents")
            variables[name] = VariableConfig(
                type=vdata["type"],
                table=table,
                parents=tuple(parents) if parents else None,
                x_base=vdata.get("x_base"),
                other_label=vdata.get("other_label"),
            )

        cls._precompute_variables(variables)

        metadata = _metadata_from_dict(raw.get("metadata"))

        column_mapping = raw.get("column_mapping", {})

        snapshots: list[Snapshot] = []
        for sdata in raw.get("snapshots", []):
            relativities = {
                name: _rows_from_list(rows)
                for name, rows in sdata["relativities"].items()
            }
            changes = [
                Change(
                    variable=c["variable"],
                    from_=c["from"],
                    to_=c["to"],
                    old_relativity=c["old_relativity"],
                    new_relativity=c["new_relativity"],
                    from_b=c.get("from_b"),
                    to_b=c.get("to_b"),
                    is_cell=bool(c.get("is_cell", False)),
                )
                for c in sdata.get("changes", [])
            ]
            snapshots.append(
                Snapshot(
                    version=sdata["version"],
                    description=sdata["description"],
                    timestamp=sdata["timestamp"],
                    parent_version=sdata["parent_version"],
                    relativities=relativities,
                    changes=changes,
                    metrics=sdata.get("metrics"),
                    column_mapping=sdata.get("column_mapping", {}),
                    metadata=sdata.get("metadata", {}),
                )
            )

        return cls(
            base_rate=raw["base_rate"],
            variables=variables,
            metadata=metadata,
            snapshots=snapshots,
            current_version=raw.get("current_version", 0),
            column_mapping=column_mapping,
        )

    @staticmethod
    def _migrate(raw: dict[str, Any], version: int) -> dict[str, Any]:
        """Upgrade an older file dict in memory to :data:`FORMAT_VERSION`."""
        raw = dict(raw)
        if version < 2:
            # v1 (easy_glm <= 0.3): no offset/link/target flags were recorded.
            meta = dict(raw.get("metadata") or {})
            meta.setdefault("offset_col", None)
            meta.setdefault("offset_is_log", True)
            meta.setdefault("link", "log")
            meta.setdefault("divide_target_by_weight", None)
            raw["metadata"] = meta
        raw["format_version"] = FORMAT_VERSION
        return raw

    @staticmethod
    def _precompute_interaction(
        name: str, config: VariableConfig, variables: dict[str, VariableConfig]
    ) -> None:
        """Build the cell relativity matrix over the parents' table rows."""
        if config.parents is None:
            raise ValueError(f"Interaction {name!r} has no parents recorded")
        a, b = config.parents
        for parent in (a, b):
            if parent not in variables:
                raise ValueError(
                    f"Interaction {name!r} needs its parent {parent!r} in the model"
                )
            if variables[parent].type == "interaction":
                raise ValueError(
                    f"Interaction {name!r}: parent {parent!r} is itself one"
                )
        ka = {(r.from_, r.to_): i for i, r in enumerate(variables[a].table)}
        kb = {(r.from_, r.to_): i for i, r in enumerate(variables[b].table)}
        matrix = np.ones((len(ka), len(kb)), dtype=float)
        for row in config.table:
            ia = ka.get((row.from_a, row.to_a))
            ib = kb.get((row.from_b, row.to_b))
            if ia is None or ib is None:
                raise ValueError(
                    f"Interaction {name!r}: cell {row.key} does not match a row of "
                    f"{a!r} × {b!r}"
                )
            matrix[ia, ib] = float(row.relativity)
        config.cell_matrix = matrix

    @staticmethod
    def _precompute_variables(variables: dict[str, VariableConfig]) -> None:
        for name, config in variables.items():
            if config.type == "interaction" and config.cell_matrix is None:
                RateModel._precompute_interaction(name, config, variables)
        for config in variables.values():
            if config.type == "numeric" and config.breakpoints is None:
                # An optional (None, None) row carries the relativity for nulls.
                null_rows = [
                    r for r in config.table if r.from_ is None and r.to_ is None
                ]
                bins = [
                    r for r in config.table if not (r.from_ is None and r.to_ is None)
                ]
                config.breakpoints = np.array(
                    [float(r.from_) for r in bins if r.from_ is not None],
                    dtype=float,
                )
                config.relativities = np.array(
                    [r.relativity for r in bins], dtype=float
                )
                config.null_relativity = null_rows[0].relativity if null_rows else None
            elif config.type == "linear" and config.breakpoints is None:
                null_rows = [
                    r for r in config.table if r.from_ is None and r.to_ is None
                ]
                bands = sorted(
                    (
                        r
                        for r in config.table
                        if not (r.from_ is None and r.to_ is None)
                    ),
                    key=lambda r: (r.from_ is not None, r.from_ or 0.0),
                )
                config.breakpoints = np.array(
                    [float(r.from_) for r in bands if r.from_ is not None], dtype=float
                )
                config.relativities = np.array(
                    [r.relativity for r in bands], dtype=float
                )
                config.slopes = np.array([float(r.slope) for r in bands], dtype=float)
                lo = float(config.breakpoints[0]) if len(config.breakpoints) else 0.0
                config.starts = np.array(
                    [lo if r.from_ is None else float(r.from_) for r in bands],
                    dtype=float,
                )
                config.null_relativity = null_rows[0].relativity if null_rows else None
            elif config.type == "categorical" and config.cat_map is None:
                config.cat_map = {}
                config.level_index = {}
                for i, row in enumerate(config.table):
                    if row.from_ is not None:
                        config.cat_map[str(row.from_)] = row.relativity
                        config.level_index[str(row.from_)] = i
                    else:
                        config.fallback = row.relativity

    def launch_editor(self, data=None, test_data=None, port=8501, **kwargs):
        from easy_glm.ui import launch_editor as _launch

        _launch(self, data=data, test_data=test_data, port=port, **kwargs)


def create_rate_model(
    tables: dict[str, pl.DataFrame],
    base_rate: float,
    *,
    save_to: str | Path | None = None,
    **metadata: Any,
) -> RateModel:
    """:meth:`RateModel.from_rate_tables` plus an optional ``to_json(save_to)``."""
    rm = RateModel.from_rate_tables(tables, base_rate, **metadata)
    if save_to is not None:
        rm.to_json(save_to)
    return rm
