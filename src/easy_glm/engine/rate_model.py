from __future__ import annotations

import copy
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from glum import GeneralizedLinearRegressor

from ._scoring import score_categorical, score_numeric
from .models import Change, FromToRow, ModelMetadata, Snapshot, VariableConfig

_UNSET = object()


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
        all_tables: dict[str, pl.DataFrame],
        blueprint: dict[str, Any],
        base_rate: float,
        model_type: str | None = None,
        target: str | None = None,
        weight_col: str | None = None,
        exposure_col: str | None = None,
        train_test_col: str | None = None,
        predictor_variables: list[str] | None = None,
    ) -> RateModel:
        variables: dict[str, VariableConfig] = {}
        pred_vars = predictor_variables or list(all_tables.keys())

        for var_name in pred_vars:
            rate_table = all_tables.get(var_name)
            bp_values = blueprint.get(var_name)
            if rate_table is None or bp_values is None or not bp_values:
                continue

            if isinstance(bp_values[0], int | float):
                variables[var_name] = cls._build_numeric_rows(
                    rate_table, bp_values, var_name
                )
            else:
                variables[var_name] = cls._build_categorical_rows(
                    rate_table, bp_values, var_name
                )

        metadata = ModelMetadata(
            model_type=model_type,
            target=target,
            weight_col=weight_col,
            exposure_col=exposure_col,
            train_test_col=train_test_col,
            predictor_variables=list(variables.keys()),
        )
        rm = cls(base_rate=base_rate, variables=variables, metadata=metadata)
        rm.create_snapshot("Base model")
        return rm

    @classmethod
    def from_glm_model(
        cls,
        model: GeneralizedLinearRegressor,
        dataset: pl.DataFrame,
        blueprint: dict[str, Any],
        base_rate: float,
        *,
        model_type: str | None = None,
        target: str | None = None,
        weight_col: str | None = None,
        exposure_col: str | None = None,
        train_test_col: str | None = None,
        predictor_variables: list[str] | None = None,
        random_seed: int = 42,
    ) -> RateModel:
        from easy_glm.core.all_ratetables import generate_all_ratetables

        if predictor_variables is None:
            predictor_variables = [
                v
                for v in blueprint
                if blueprint.get(v)
                and isinstance(blueprint[v], list)
                and len(blueprint[v]) > 0
            ]

        all_tables = generate_all_ratetables(
            model=model,
            dataset=dataset,
            predictor_variables=predictor_variables,
            blueprint=blueprint,
            random_seed=random_seed,
        )

        return cls.from_rate_tables(
            all_tables=all_tables,
            blueprint=blueprint,
            base_rate=base_rate,
            model_type=model_type,
            target=target,
            weight_col=weight_col,
            exposure_col=exposure_col,
            train_test_col=train_test_col,
            predictor_variables=predictor_variables,
        )

    @staticmethod
    def _build_numeric_rows(
        rate_table: pl.DataFrame, bp_values: list, col_name: str
    ) -> VariableConfig:
        rate_dict = dict(
            zip(
                rate_table[col_name].to_list(),
                rate_table["relativity"].to_list(),
                strict=False,
            )
        )

        first_bp = bp_values[0]
        first_rel = rate_dict.get(first_bp, 1.0)
        rows = [FromToRow(from_=None, to_=first_bp, relativity=first_rel)]

        for i in range(len(bp_values)):
            from_val = bp_values[i]
            to_val = bp_values[i + 1] if i + 1 < len(bp_values) else None
            rel = rate_dict.get(from_val, 1.0)
            rows.append(FromToRow(from_=from_val, to_=to_val, relativity=rel))

        breakpoints = np.array(
            [float(r.from_) for r in rows if r.from_ is not None], dtype=float
        )
        relativities = np.array([r.relativity for r in rows], dtype=float)

        return VariableConfig(
            type="numeric",
            table=rows,
            breakpoints=breakpoints,
            relativities=relativities,
        )

    @staticmethod
    def _build_categorical_rows(
        rate_table: pl.DataFrame, bp_values: list, col_name: str
    ) -> VariableConfig:
        rate_dict = dict(
            zip(
                rate_table[col_name].cast(pl.Utf8).to_list(),
                rate_table["relativity"].to_list(),
                strict=False,
            )
        )

        rows: list[FromToRow] = []
        for level in bp_values:
            rel = rate_dict.get(str(level), 1.0)
            rows.append(FromToRow(from_=level, to_=level, relativity=rel))

        rows.append(FromToRow(from_=None, to_=None, relativity=1.0))

        cat_map: dict[str, float] = {}
        fallback = 1.0
        for row in rows:
            if row.from_ is not None:
                cat_map[str(row.from_)] = row.relativity
            else:
                fallback = row.relativity

        return VariableConfig(
            type="categorical",
            table=rows,
            cat_map=cat_map,
            fallback=fallback,
        )

    def predict(
        self,
        data: pl.DataFrame,
        *,
        version: int | None = None,
        column_map: dict[str, str] | None = None,
        exposure_col: str | None = _UNSET,
    ) -> np.ndarray:
        if version is not None:
            self.switch_to(version)

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
            if config.type == "numeric":
                rel = score_numeric(col.to_numpy(), config)
            else:
                rel = score_categorical(col, config)

            result *= rel

        result = self._apply_exposure(result, data, exposure_col)

        return result

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
                self._precompute_variables(self.variables)
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
            if len(set(round(r, 5) for r in rels)) > 1:
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

    def create_snapshot(self, description: str) -> int:
        version = len(self.snapshots) + 1
        parent = self.current_version if self.snapshots else None

        relativities = {
            name: copy.deepcopy(config.table) for name, config in self.variables.items()
        }

        metadata_dict = {
            "model_type": self.metadata.model_type,
            "target": self.metadata.target,
            "weight_col": self.metadata.weight_col,
            "exposure_col": self.metadata.exposure_col,
            "train_test_col": self.metadata.train_test_col,
            "predictor_variables": list(self.metadata.predictor_variables),
        }

        snapshot = Snapshot(
            version=version,
            description=description,
            timestamp=datetime.now(timezone.utc).isoformat(),
            parent_version=parent,
            relativities=relativities,
            changes=list(self._pending_changes),
            column_mapping=dict(self.column_mapping),
            metadata=metadata_dict,
        )
        self.snapshots.append(snapshot)
        self.current_version = version
        self._pending_changes.clear()
        return version

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
        RateModel._precompute_variables(self.variables)
        self.column_mapping = dict(snapshot.column_mapping)
        if snapshot.metadata:
            self.metadata = ModelMetadata(**snapshot.metadata)
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

    def diff(self, v1: int, v2: int) -> list[Change]:
        return self.snapshots[v2 - 1].changes

    def to_json(self, path: str | Path) -> None:
        data = self._to_dict()
        path = Path(path)
        path.write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def from_json(cls, path: str | Path) -> RateModel:
        raw = json.loads(Path(path).read_text())
        return cls._from_dict(raw)

    def _to_dict(self) -> dict[str, Any]:
        return {
            "metadata": {
                "model_type": self.metadata.model_type,
                "target": self.metadata.target,
                "weight_col": self.metadata.weight_col,
                "exposure_col": self.metadata.exposure_col,
                "train_test_col": self.metadata.train_test_col,
                "predictor_variables": list(self.metadata.predictor_variables),
            },
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
            variables[name] = VariableConfig(type=vdata["type"], table=table)

        cls._precompute_variables(variables)

        meta_raw = raw.get("metadata", {})
        metadata = ModelMetadata(
            model_type=meta_raw.get("model_type"),
            target=meta_raw.get("target"),
            weight_col=meta_raw.get("weight_col"),
            exposure_col=meta_raw.get("exposure_col"),
            train_test_col=meta_raw.get("train_test_col"),
            predictor_variables=meta_raw.get("predictor_variables", []),
        )

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
    def _precompute_variables(variables: dict[str, VariableConfig]) -> None:
        for config in variables.values():
            if config.type == "numeric" and config.breakpoints is None:
                config.breakpoints = np.array(
                    [float(r.from_) for r in config.table if r.from_ is not None],
                    dtype=float,
                )
                config.relativities = np.array(
                    [r.relativity for r in config.table], dtype=float
                )
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
    all_tables: dict[str, pl.DataFrame],
    blueprint: dict[str, Any],
    base_rate: float,
    *,
    model_type: str | None = None,
    target: str | None = None,
    weight_col: str | None = None,
    exposure_col: str | None = None,
    train_test_col: str | None = None,
    save_to: str | Path | None = None,
) -> RateModel:
    """Create a RateModel from pre-computed rate tables and optionally save to disk.

    This is a convenience wrapper around :meth:`RateModel.from_rate_tables`
    that handles the common workflow of converting the output of
    :func:`~easy_glm.generate_all_ratetables` into a portable ``.easyglm``
    JSON file that can be scored on new data.

    Parameters
    ----------
    all_tables : dict[str, pl.DataFrame]
        Rate tables produced by :func:`~easy_glm.generate_all_ratetables`.
    blueprint : dict[str, Any]
        Blueprint produced by :func:`~easy_glm.generate_blueprint`.
    base_rate : float
        The base rate to use as the multiplicative starting point when scoring.
    model_type : str or None
        The GLM family used (e.g. ``"poisson"``, ``"gamma"``, ``"normal"``).
    target : str or None
        The name of the target variable column.
    weight_col : str or None
        The name of the weight column used during fitting.
    exposure_col : str or None
        The name of the exposure column. When scoring, predictions are
        multiplied by this column if present in the data. Pass ``None``
        to ``RateModel.predict(exposure_col=None)`` to skip this step.
    train_test_col : str or None
        The name of the train/test split indicator column.
    save_to : str or Path or None
        If provided, the model is serialized as JSON to this path.

    Returns
    -------
    RateModel
        The constructed rate model (also saved to ``save_to`` if requested).
    """
    rm = RateModel.from_rate_tables(
        all_tables=all_tables,
        blueprint=blueprint,
        base_rate=base_rate,
        model_type=model_type,
        target=target,
        weight_col=weight_col,
        exposure_col=exposure_col,
        train_test_col=train_test_col,
        predictor_variables=list(all_tables.keys()),
    )
    if save_to is not None:
        rm.to_json(save_to)
    return rm
