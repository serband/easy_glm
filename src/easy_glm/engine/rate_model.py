from __future__ import annotations

import copy
import json
import warnings
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ._scoring import score_categorical, score_numeric
from .models import Change, FromToRow, ModelMetadata, Snapshot, VariableConfig

_UNSET = object()

#: ``.easyglm`` file format version written by this release. Readers accept
#: older versions (migrating them) and refuse newer ones.
FORMAT_VERSION = 2

#: How each ``VariableConfig.type`` is scored. Unknown types are an error, never
#: silently treated as another type.
_SCORERS: dict[str, Any] = {
    "numeric": lambda col, cfg: score_numeric(col.to_numpy(), cfg),
    "categorical": score_categorical,
}

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
        self.base_rate = base_rate
        self.variables = variables
        self.metadata = metadata or ModelMetadata()
        self.snapshots = snapshots or []
        self.current_version = current_version
        self._pending_changes: list[Change] = []
        self.column_mapping = column_mapping or {}

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
        ``from == to`` per level. In both cases a row with ``from`` and ``to``
        both null is the null / Other row.
        """
        pred_vars = predictor_variables or list(tables)
        variables: dict[str, VariableConfig] = {}
        for var in pred_vars:
            if var not in tables:
                raise KeyError(f"No table for variable {var!r}")
            variables[var] = cls._config_from_table(var, tables[var])
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
        from_dtype = table["from"].dtype
        numeric = from_dtype.is_numeric()
        rows: list[FromToRow] = []
        for lo, hi, rel in table.select("from", "to", "relativity").iter_rows():
            if rel is None:
                raise ValueError(f"Table for {name!r} has a null relativity")
            if lo is None and hi is None:
                rows.append(FromToRow(None, None, float(rel)))
            elif numeric:
                rows.append(
                    FromToRow(
                        None if lo is None else float(lo),
                        None if hi is None else float(hi),
                        float(rel),
                    )
                )
            else:
                rows.append(
                    FromToRow(str(lo), str(hi if hi is not None else lo), float(rel))
                )
        if numeric:
            # bands must tile the line: (None, k0), [k0, k1), ..., [k_last, None)
            bands = [r for r in rows if not (r.from_ is None and r.to_ is None)]
            if not bands:
                raise ValueError(f"Numeric table for {name!r} has no bands")
            if bands[0].from_ is not None or bands[-1].to_ is not None:
                raise ValueError(
                    f"Numeric table for {name!r} must start with an open lower band "
                    "and end with an open upper band"
                )
            for a, b in zip(bands[:-1], bands[1:], strict=True):
                if a.to_ != b.from_:
                    raise ValueError(
                        f"Numeric table for {name!r} has a gap or overlap at {a.to_!r}"
                    )
            return VariableConfig(type="numeric", table=rows)
        if not any(r.from_ is None and r.to_ is None for r in rows):
            rows.append(FromToRow(None, None, 1.0))  # Other / unseen levels
        return VariableConfig(type="categorical", table=rows)

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
            if name not in data.columns:
                raise ValueError(f"Column '{name}' not found in data")

            col = data[name]
            scorer = _SCORERS.get(config.type)
            if scorer is None:
                raise ValueError(
                    f"Variable {name!r} has table type {config.type!r}, which this "
                    f"version of easy_glm cannot score (known: {sorted(_SCORERS)}). "
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
        result = self._apply_exposure(result, data, exposure_col)

        return result

    def _apply_offset(self, result: np.ndarray, data: pl.DataFrame) -> np.ndarray:
        """Multiply by ``exp(offset)`` (or the raw column) when the fit used an offset."""
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
        if exposure_name not in data.columns:
            warnings.warn(
                f"Exposure column '{exposure_name}' not found in data "
                f"— predictions not multiplied by exposure",
                stacklevel=2,
            )
            return result
        return result * data[exposure_name].to_numpy()

    def update_relativity(
        self, var: str, from_: Any, to_: Any, new_value: float
    ) -> None:
        if var not in self.variables:
            raise KeyError(f"Variable '{var}' not found")

        config = self.variables[var]
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
                self._precompute_variables({var: config})
                return

        raise ValueError(
            f"No row found with from={from_!r}, to={to_!r} in variable '{var}'"
        )

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
            old_rows = {(r.from_, r.to_): r.relativity for r in before.get(var, [])}
            for r in rows:
                old = old_rows.get((r.from_, r.to_))
                if old is None or abs(old - r.relativity) > tol:
                    out.append(
                        Change(
                            variable=var,
                            from_=r.from_,
                            to_=r.to_,
                            old_relativity=float("nan") if old is None else old,
                            new_relativity=r.relativity,
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
        from easy_glm.core.excel import rate_model_tables, write_rate_tables_xlsx

        summary: dict[str, Any] = {
            "tables": "current relativities (manual adjustments included)",
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
            "version": self.current_version,
            "snapshots": len(self.snapshots),
        }
        return write_rate_tables_xlsx(rate_model_tables(self), path, summary=summary)

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
                    "table": [
                        {
                            "from": row.from_,
                            "to": row.to_,
                            "relativity": row.relativity,
                        }
                        for row in config.table
                    ],
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
                        name: [
                            {
                                "from": row.from_,
                                "to": row.to_,
                                "relativity": row.relativity,
                            }
                            for row in table
                        ]
                        for name, table in s.relativities.items()
                    },
                    "changes": [
                        {
                            "variable": c.variable,
                            "from": c.from_,
                            "to": c.to_,
                            "old_relativity": c.old_relativity,
                            "new_relativity": c.new_relativity,
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
            table = [
                FromToRow(
                    from_=r["from"],
                    to_=r["to"],
                    relativity=r["relativity"],
                )
                for r in vdata["table"]
            ]
            if vdata["type"] not in _SCORERS:
                raise ValueError(
                    f"Variable {name!r} has table type {vdata['type']!r}, which this "
                    f"version of easy_glm cannot score (known: {sorted(_SCORERS)}). "
                    "The model file probably comes from a newer easy_glm."
                )
            variables[name] = VariableConfig(type=vdata["type"], table=table)

        cls._precompute_variables(variables)

        metadata = _metadata_from_dict(raw.get("metadata"))

        column_mapping = raw.get("column_mapping", {})

        snapshots: list[Snapshot] = []
        for sdata in raw.get("snapshots", []):
            relativities = {
                name: [
                    FromToRow(
                        from_=r["from"],
                        to_=r["to"],
                        relativity=r["relativity"],
                    )
                    for r in rows
                ]
                for name, rows in sdata["relativities"].items()
            }
            changes = [
                Change(
                    variable=c["variable"],
                    from_=c["from"],
                    to_=c["to"],
                    old_relativity=c["old_relativity"],
                    new_relativity=c["new_relativity"],
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
    def _precompute_variables(variables: dict[str, VariableConfig]) -> None:
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
            elif config.type == "categorical" and config.cat_map is None:
                config.cat_map = {}
                for row in config.table:
                    if row.from_ is not None:
                        config.cat_map[str(row.from_)] = row.relativity
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
