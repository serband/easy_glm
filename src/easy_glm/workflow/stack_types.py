"""Ordered, frozen multiplicative components and their accepted scorers."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import polars as pl

from easy_glm.engine.rate_model import RateModel

_DESIGN_DEFAULT_KEYS = {
    "n_bins",
    "min_level_share",
    "null_indicator",
    "max_integer_knots",
}
_VARIABLE_DESIGN_KEYS = {
    "kind",
    "knots",
    "clamp",
    "n_bins",
    "null_indicator",
    "min_level_share",
    "max_levels",
    "levels",
    "monotone",
    "penalty_weight",
}


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_component_design(
    design: Any, factors: tuple[str, ...]
) -> dict[str, Any] | None:
    """Validate a self-contained, component-local design snapshot.

    It deliberately has the same JSON shape as ``Project.design``, but the
    variable definitions may only describe factors fitted by this recipe.  The
    project-wide design remains untouched.
    """
    if design is None:
        return None
    if not isinstance(design, dict) or set(design) != {"defaults", "variables"}:
        raise ValueError("Component design needs only defaults and variables")
    defaults = design["defaults"]
    variables = design["variables"]
    if not isinstance(defaults, dict) or set(defaults) != _DESIGN_DEFAULT_KEYS:
        raise ValueError("Component design defaults are incomplete or invalid")
    if not isinstance(variables, dict):
        raise ValueError("Component design variables must be an object")
    if not all(isinstance(name, str) and name for name in variables):
        raise ValueError("Component design variable names must be non-empty strings")
    outside = sorted(set(variables) - set(factors))
    if outside:
        raise ValueError(
            "Component design may only define selected factors: " + ", ".join(outside)
        )
    if (
        isinstance(defaults["n_bins"], bool)
        or not isinstance(defaults["n_bins"], int)
        or not 2 <= defaults["n_bins"] <= 200
    ):
        raise ValueError(
            "Component design default n_bins must be an integer from 2 to 200"
        )
    if (
        not _finite_number(defaults["min_level_share"])
        or not 0 <= float(defaults["min_level_share"]) < 1
    ):
        raise ValueError("Component design default min_level_share must be in [0, 1)")
    if not isinstance(defaults["null_indicator"], bool):
        raise ValueError(
            "Component design default null_indicator must be true or false"
        )
    if (
        isinstance(defaults["max_integer_knots"], bool)
        or not isinstance(defaults["max_integer_knots"], int)
        or defaults["max_integer_knots"] < 1
    ):
        raise ValueError(
            "Component design default max_integer_knots must be a positive integer"
        )
    for name, definition in variables.items():
        if not isinstance(definition, dict) or set(definition) != _VARIABLE_DESIGN_KEYS:
            raise ValueError(f"Component design for {name!r} is incomplete or invalid")
        kind = definition["kind"]
        if kind not in (None, "step", "linear", "continuous", "categorical"):
            raise ValueError(f"Component design kind for {name!r} is invalid")
        knots = definition["knots"]
        if not (
            knots in ("quantile", "integer")
            or (
                isinstance(knots, list)
                and all(_finite_number(value) for value in knots)
                and all(
                    left < right for left, right in zip(knots, knots[1:], strict=False)
                )
            )
        ):
            raise ValueError(
                f"Component design cuts for {name!r} must be increasing finite values"
            )
        clamp = definition["clamp"]
        if clamp is not None and not (
            isinstance(clamp, list)
            and len(clamp) == 2
            and all(_finite_number(value) for value in clamp)
            and clamp[0] < clamp[1]
        ):
            raise ValueError(f"Component design clamp for {name!r} must be [low, high]")
        n_bins = definition["n_bins"]
        if n_bins is not None and (
            isinstance(n_bins, bool)
            or not isinstance(n_bins, int)
            or not 2 <= n_bins <= 200
        ):
            raise ValueError(
                f"Component design n_bins for {name!r} must be from 2 to 200"
            )
        if definition["null_indicator"] is not None and not isinstance(
            definition["null_indicator"], bool
        ):
            raise ValueError(
                f"Component design null_indicator for {name!r} must be true or false"
            )
        share = definition["min_level_share"]
        if share is not None and (
            not _finite_number(share) or not 0 <= float(share) < 1
        ):
            raise ValueError(
                f"Component design min_level_share for {name!r} must be in [0, 1)"
            )
        maximum = definition["max_levels"]
        if maximum is not None and (
            isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1
        ):
            raise ValueError(
                f"Component design max_levels for {name!r} must be positive"
            )
        levels = definition["levels"]
        if levels is not None and (
            not isinstance(levels, list)
            or not all(isinstance(value, str) for value in levels)
            or len(set(levels)) != len(levels)
        ):
            raise ValueError(
                f"Component design levels for {name!r} must be distinct strings"
            )
        monotone = definition["monotone"]
        if monotone not in (None, "increasing", "decreasing"):
            raise ValueError(f"Component design monotone for {name!r} is invalid")
        if monotone and kind == "categorical":
            raise ValueError(
                f"Component design categorical {name!r} cannot be monotone"
            )
        penalty = definition["penalty_weight"]
        if not _finite_number(penalty) or float(penalty) < 0:
            raise ValueError(
                f"Component design penalty_weight for {name!r} must be nonnegative"
            )
    return copy.deepcopy(design)


@dataclass(frozen=True)
class ComponentRecipe:
    component_id: str
    name: str
    kind: str
    factors: tuple[str, ...] = ()
    settings: dict[str, Any] = field(default_factory=dict)
    amendments: tuple[dict[str, Any], ...] = ()
    #: Full effective per-component definitions.  ``None`` preserves legacy
    #: project-wide design behaviour and is deliberately omitted on export.
    design: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value.strip() for value in self.factors):
            raise ValueError("Component factors must be non-empty strings")
        if len(set(self.factors)) != len(self.factors):
            raise ValueError("Component factors must not contain duplicates")
        object.__setattr__(
            self, "design", validate_component_design(self.design, self.factors)
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ComponentRecipe:
        if not isinstance(raw, dict):
            raise ValueError("Component recipe must be an object")
        raw_factors = raw.get("factors", ())
        if not isinstance(raw_factors, list | tuple):
            raise ValueError("Component factors must be a list")
        if not isinstance(raw.get("settings", {}), dict):
            raise ValueError("Component settings must be an object")
        if not isinstance(raw.get("amendments", ()), list | tuple):
            raise ValueError("Component amendments must be a list")
        factors = tuple(raw_factors)
        return cls(
            component_id=str(raw["component_id"]),
            name=str(raw["name"]),
            kind=str(raw["kind"]),
            factors=factors,
            settings=dict(raw.get("settings", {})),
            amendments=tuple(raw.get("amendments", ())),
            design=validate_component_design(raw.get("design"), factors),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.design is None:
            result.pop("design")
        return result


@dataclass
class ComponentArtifact:
    recipe: ComponentRecipe
    prefix_fingerprint: str
    scorer: Any
    evidence: dict[str, Any] = field(default_factory=dict)
    status: str = "current"

    @property
    def name(self) -> str:
        return self.recipe.name

    @property
    def kind(self) -> str:
        return self.recipe.kind

    @property
    def factors(self) -> tuple[str, ...]:
        return self.recipe.factors

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe.to_dict(),
            "prefix_fingerprint": self.prefix_fingerprint,
            "scorer": self.scorer.to_dict(),
            "evidence": self.evidence,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ComponentArtifact:
        data = raw["scorer"]
        kind = data["type"]
        scorer_type: Any = {
            "rate_model": RateModelScorer,
            "catboost": CatBoostScorer,
            "lookup": LookupScorer,
            "constant": ConstantScorer,
        }[kind]
        scorer = scorer_type.from_dict(data)
        return cls(
            ComponentRecipe.from_dict(raw["recipe"]),
            raw["prefix_fingerprint"],
            scorer,
            dict(raw.get("evidence", {})),
            raw.get("status", "current"),
        )


@dataclass
class ConstantScorer:
    log_value: float

    def linear_predictor(self, frame: pl.DataFrame) -> np.ndarray:
        return np.full(frame.height, self.log_value, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "constant", "log_value": self.log_value}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ConstantScorer:
        return cls(float(raw["log_value"]))


@dataclass
class RateModelScorer:
    rate_model: RateModel

    def linear_predictor(self, frame: pl.DataFrame) -> np.ndarray:
        return self.rate_model.linear_predictor(frame, include_offset=False)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "rate_model", "model": self.rate_model.to_dict()}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RateModelScorer:
        return cls(RateModel.from_dict(raw["model"]))


@dataclass
class CatBoostScorer:
    factors: tuple[str, ...]
    model: Any
    categorical: tuple[int, ...] = ()

    def linear_predictor(self, frame: pl.DataFrame) -> np.ndarray:
        from .stack import raw_features

        raw, _ = raw_features(frame, self.factors, self.categorical)
        result = np.asarray(
            self.model.predict(raw, prediction_type="RawFormulaVal"), dtype=float
        )
        if result.shape != (frame.height,) or not np.all(np.isfinite(result)):
            raise ValueError("CatBoost component has a nonfinite correction")
        return result

    def to_dict(self) -> dict[str, Any]:
        encoded = base64.b64encode(self.model._serialize_model()).decode("ascii")
        return {
            "type": "catboost",
            "factors": list(self.factors),
            "categorical": list(self.categorical),
            "cbm": encoded,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CatBoostScorer:
        from catboost import CatBoostRegressor

        model = CatBoostRegressor()
        model.load_model(blob=base64.b64decode(raw["cbm"]))
        return cls(tuple(raw["factors"]), model, tuple(raw.get("categorical", ())))


@dataclass
class LookupScorer:
    key: str
    table: pl.DataFrame
    attributes: tuple[str, ...] = ()
    fallback: float | None = None

    def coverage(self, frame: pl.DataFrame) -> dict[str, Any]:
        """Count missing keys and explicit fallback use on this scoring book."""
        if frame.schema.get(self.key) != self.table.schema.get(self.key):
            raise ValueError(
                f"Lookup key {self.key!r} has a different type from its pinned catalogue"
            )
        matched = frame.select(self.key).join(
            self.table.select(self.key).with_columns(pl.lit(True).alias("__matched")),
            on=self.key,
            how="left",
            validate="m:1",
        )
        unknown = matched.filter(pl.col("__matched").is_null())
        return {
            "rows": frame.height,
            "missing_key_rows": unknown.height,
            "fallback_rows": unknown.height if self.fallback is not None else 0,
            "unknown_examples": [
                str(value) for value in unknown[self.key].unique().head(3).to_list()
            ],
        }

    def enrich(self, frame: pl.DataFrame) -> pl.DataFrame:
        from .catalogue import join_catalogue

        return join_catalogue(
            frame,
            self.table.drop("multiplier"),
            self.key,
            self.attributes,
            require_coverage=self.fallback is None,
        )

    def linear_predictor(self, frame: pl.DataFrame) -> np.ndarray:
        if frame.schema.get(self.key) != self.table.schema.get(self.key):
            raise ValueError(
                f"Lookup key {self.key!r} has a different type from its pinned catalogue"
            )
        joined = (
            frame.with_row_index("__lookup_row_index")
            .join(
                self.table.select(self.key, "multiplier"),
                on=self.key,
                how="left",
                validate="m:1",
                suffix="__lookup",
            )
            .sort("__lookup_row_index")
        )
        column = "multiplier__lookup" if "multiplier" in frame.columns else "multiplier"
        values = joined[column].to_numpy()
        if self.fallback is not None:
            values = np.where(
                joined[column].is_null().to_numpy(), self.fallback, values
            )
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values) & (values > 0)):
            raise ValueError(
                f"Lookup {self.key!r} has nonpositive or missing multipliers"
            )
        return np.log(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "lookup",
            "key": self.key,
            "table": self.table.to_dict(as_series=False),
            "schema": {name: str(dtype) for name, dtype in self.table.schema.items()},
            "attributes": list(self.attributes),
            "fallback": self.fallback,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LookupScorer:
        from .catalogue import restore_schema, validate_catalogue

        result = cls(
            raw["key"],
            restore_schema(pl.DataFrame(raw["table"]), raw.get("schema")),
            tuple(raw.get("attributes", ())),
            raw.get("fallback"),
        )
        validate_catalogue(result.table, result.key, result.attributes)
        if "multiplier" not in result.table.columns:
            raise ValueError("Loaded lookup has no multiplier column")
        values = result.table["multiplier"].cast(pl.Float64).to_numpy()
        if not np.all(np.isfinite(values) & (values > 0)):
            raise ValueError("Loaded lookup has nonpositive or nonfinite multipliers")
        if result.fallback is not None and (
            not math.isfinite(result.fallback) or result.fallback <= 0
        ):
            raise ValueError("Loaded lookup fallback must be positive and finite")
        return result


@dataclass
class StackScorer:
    family: str
    link: str
    target: str
    weight: str | None
    offset_col: str | None
    divide_target_by_weight: bool
    tweedie_power: float | None
    components: list[ComponentArtifact] = field(default_factory=list)
    #: Inspectable pieces of an adapted legacy scorer; never summed again.
    virtual_components: list[ComponentArtifact] = field(default_factory=list)

    def component(self, identifier: str) -> ComponentArtifact:
        matches = [
            item
            for item in [*self.virtual_components, *self.components]
            if identifier in (item.recipe.component_id, item.recipe.name)
        ]
        if self.virtual_components:
            virtual = [item for item in matches if item in self.virtual_components]
            if len(virtual) == 1:
                return virtual[0]
        if len(matches) != 1:
            raise KeyError(f"Component {identifier!r} is missing or ambiguous")
        return matches[0]

    def diagnostic_components(self) -> list[ComponentArtifact]:
        """Return the real fitted stages that can be used as diagnostic cut-offs.

        ``baseline`` is an internal conditional intercept for correction-only
        stacks, and ``virtual_components`` only expose the shape of an adapted
        legacy model.  Neither is a model a user can select as a diagnostic
        result.  A legacy component itself *is* a physical scorer and remains
        selectable when native stages follow it.
        """
        return [item for item in self.components if item.recipe.kind != "baseline"]

    def prefix_through(self, component_id: str | None = None) -> StackScorer:
        """Return the accepted scorer through one physical fitted component.

        The returned scorer retains the original response convention and
        external offset.  It uses the saved artifacts themselves, including
        amendments and catalogue enrichments, rather than fitting anything
        again or substituting a currently selected parent.
        """
        if component_id is None:
            return self.clone()
        indices = [
            index
            for index, item in enumerate(self.components)
            if item.recipe.component_id == component_id
            and item.recipe.kind != "baseline"
        ]
        if len(indices) != 1:
            raise KeyError(f"Diagnostic stage {component_id!r} is unavailable")
        end = indices[0] + 1
        result = self.clone()
        result.components = result.components[:end]
        # Virtual components are inspection aids for a legacy scorer.  They
        # have already been incorporated in the physical legacy artifact.
        result.virtual_components = []
        return result

    def component_multiplier(self, identifier: str, frame: pl.DataFrame) -> np.ndarray:
        target = self.component(identifier)
        if target in self.virtual_components:
            return np.exp(target.scorer.linear_predictor(frame))
        working = frame
        for artifact in self.components:
            if artifact is target:
                return np.exp(artifact.scorer.linear_predictor(working))
            if isinstance(artifact.scorer, LookupScorer):
                working = artifact.scorer.enrich(working)
        raise KeyError(identifier)

    def linear_predictor(self, frame: pl.DataFrame) -> np.ndarray:
        result = np.zeros(frame.height, dtype=np.float64)
        if self.offset_col:
            if self.offset_col not in frame:
                raise ValueError(f"Offset column {self.offset_col!r} not found")
            result += frame[self.offset_col].cast(pl.Float64).to_numpy()
        working = frame
        for artifact in self.components:
            result += artifact.scorer.linear_predictor(working)
            if isinstance(artifact.scorer, LookupScorer):
                working = artifact.scorer.enrich(working)
        if not np.all(np.isfinite(result)):
            raise ValueError("Stack linear predictor is not finite")
        return result

    def predict_unit(self, frame: pl.DataFrame) -> np.ndarray:
        with np.errstate(over="ignore", under="ignore"):
            value = np.exp(self.linear_predictor(frame))
        if not np.all(np.isfinite(value) & (value > 0)):
            raise ValueError("Stack prediction is not finite and strictly positive")
        return value

    def predict(
        self, frame: pl.DataFrame, exposure_col: str | None = None
    ) -> np.ndarray:
        result = self.predict_unit(frame)
        if exposure_col is not None:
            exposure = frame[exposure_col].cast(pl.Float64).to_numpy()
            if not np.all(np.isfinite(exposure) & (exposure >= 0)):
                raise ValueError("Scoring exposure must be nonnegative and finite")
            result = result * exposure
        return result

    def enrich(self, frame: pl.DataFrame) -> pl.DataFrame:
        result = frame
        for artifact in self.components:
            if isinstance(artifact.scorer, LookupScorer):
                result = artifact.scorer.enrich(result)
        return result

    def clone(self) -> StackScorer:
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "family": self.family,
            "link": self.link,
            "target": self.target,
            "weight": self.weight,
            "offset_col": self.offset_col,
            "divide_target_by_weight": self.divide_target_by_weight,
            "tweedie_power": self.tweedie_power,
            "components": [item.to_dict() for item in self.components],
            "virtual_components": [item.to_dict() for item in self.virtual_components],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> StackScorer:
        if raw.get("format_version") != 1:
            raise ValueError("Unsupported stack format version")
        result = cls(
            raw["family"],
            raw["link"],
            raw["target"],
            raw.get("weight"),
            raw.get("offset_col"),
            bool(raw["divide_target_by_weight"]),
            raw.get("tweedie_power"),
            [ComponentArtifact.from_dict(item) for item in raw["components"]],
            [
                ComponentArtifact.from_dict(item)
                for item in raw.get("virtual_components", [])
            ],
        )
        for item in result.components:
            if isinstance(item.scorer, ConstantScorer) and not math.isfinite(
                item.scorer.log_value
            ):
                raise ValueError(
                    f"Component {item.recipe.component_id!r} has a nonfinite starting level"
                )
        return result

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()
