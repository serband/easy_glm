"""The Project spec: a declarative, JSON-serialisable description of a whole
modelling workflow (data → variables → split → design → models → adjustments).

The GUI edits a :class:`Project`; the engine (:mod:`easy_glm.workflow.run`)
executes it; :mod:`easy_glm.workflow.export` renders it as a Python script.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_VERSION = 1

ROLES = ("target", "weight", "exposure", "offset", "split", "id", "predictor", "ignore")
SINGLE_ROLES = ("target", "weight", "exposure", "offset", "split")
SOURCE_TYPES = ("parquet", "csv", "sas7bdat", "xlsx", "ipc")
FAMILIES = ("poisson", "gamma", "tweedie", "gaussian", "binomial", "inverse_gaussian")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@dataclass
class DataSource:
    type: str = "parquet"
    path: str = ""
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Recode:
    """Map categorical levels: ``mapping`` old -> new; ``default`` for levels not
    in the mapping: ``None`` keeps the original value, otherwise the literal."""

    mapping: dict[str, str] = field(default_factory=dict)
    default: str | None = None


@dataclass
class Derived:
    """A new column from a polars expression string, e.g.
    ``"pl.when(pl.col('Lic') == 'Q').then(pl.col('Exp')).otherwise(0)"``."""

    name: str
    expr: str


@dataclass
class Split:
    mode: str = "column"  # "column" | "random"
    column: str = "traintest"
    train_value: Any = 1
    fraction: float = 0.7
    seed: int = 42


@dataclass
class DataConfig:
    source: DataSource = field(default_factory=DataSource)
    sample_rows: int | None = None
    sample_seed: int = 42
    renames: dict[str, str] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)
    types: dict[str, str] = field(default_factory=dict)  # "categorical" | "numeric"
    recodes: dict[str, Recode] = field(default_factory=dict)
    derived: list[Derived] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    split: Split = field(default_factory=Split)


# --------------------------------------------------------------------------
# Design
# --------------------------------------------------------------------------
@dataclass
class VariableDesign:
    """Per-variable overrides of the design defaults. ``None`` = inherit."""

    kind: str | None = None  # "step" | "categorical" | None (infer from dtype)
    knots: str | list[float] = "quantile"  # "quantile" | "integer" | explicit list
    n_bins: int | None = None
    null_indicator: bool | None = None
    min_level_share: float | None = None
    max_levels: int | None = None
    levels: list[str] | None = None  # explicit levels, first = reference
    monotone: str | None = None  # "increasing" | "decreasing"


@dataclass
class DesignDefaults:
    n_bins: int = 20
    min_level_share: float = 0.0025
    null_indicator: bool = True
    max_integer_knots: int = 150


@dataclass
class DesignConfig:
    defaults: DesignDefaults = field(default_factory=DesignDefaults)
    variables: dict[str, VariableDesign] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
@dataclass
class Penalty:
    alpha: float | None = None  # explicit; None -> choose by cv
    cv: int | None = 5
    n_alphas: int = 20
    l1_ratio: float = 1.0
    min_alpha_ratio: float | None = None


@dataclass
class Adjustment:
    """A manual relativity override applied after the fit (from the editor)."""

    variable: str
    from_: Any
    to_: Any
    relativity: float


@dataclass
class ModelConfig:
    family: str = "poisson"
    link: str | None = None
    target: str | None = None
    weight: str | None = None
    offset: str | None = None
    divide_target_by_weight: bool = False
    predictors: list[str] = field(default_factory=list)
    penalty: Penalty = field(default_factory=Penalty)
    monotone: dict[str, str] = field(default_factory=dict)
    base: str = "modal"
    base_rate_override: float | None = None
    adjustments: list[Adjustment] = field(default_factory=list)
    notes: str = ""


# --------------------------------------------------------------------------
# Project
# --------------------------------------------------------------------------
@dataclass
class Project:
    name: str = "untitled"
    version: int = PROJECT_VERSION
    data: DataConfig = field(default_factory=DataConfig)
    design: DesignConfig = field(default_factory=DesignConfig)
    models: dict[str, ModelConfig] = field(default_factory=dict)
    champion: str | None = None
    exploration: dict[str, Any] = field(
        default_factory=lambda: {"leakage": {"ignored": [], "acknowledged": []}}
    )

    # -- role helpers ---------------------------------------------------
    def columns_with_role(self, role: str) -> list[str]:
        return [c for c, r in self.data.roles.items() if r == role]

    def column_with_role(self, role: str) -> str | None:
        cols = self.columns_with_role(role)
        return cols[0] if cols else None

    @property
    def target(self) -> str | None:
        return self.column_with_role("target")

    @property
    def weight(self) -> str | None:
        return self.column_with_role("weight")

    @property
    def exposure(self) -> str | None:
        return self.column_with_role("exposure")

    @property
    def offset(self) -> str | None:
        return self.column_with_role("offset")

    @property
    def split_column(self) -> str:
        return self.data.split.column

    @property
    def predictors(self) -> list[str]:
        return self.columns_with_role("predictor")

    def set_role(self, column: str, role: str) -> None:
        if role not in ROLES:
            raise ValueError(f"Unknown role {role!r}; use one of {ROLES}")
        if role in SINGLE_ROLES:
            for c in self.columns_with_role(role):
                if c != column:
                    self.data.roles[c] = "ignore"
        self.data.roles[column] = role
        if role == "split":
            self.data.split.column = column
            self.data.split.mode = "column"

    def new_model(self, name: str, **overrides: Any) -> ModelConfig:
        """Create a model config pre-filled from the roles."""
        cfg = ModelConfig(
            target=self.target,
            weight=self.weight,
            offset=self.offset,
            predictors=list(self.predictors),
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        self.models[name] = cfg
        if self.champion is None:
            self.champion = name
        return cfg

    # -- validation -----------------------------------------------------
    def validate(self, model: str | None = None) -> list[str]:
        """Return a list of problems (empty = valid)."""
        problems: list[str] = []
        for role in SINGLE_ROLES:
            if len(self.columns_with_role(role)) > 1:
                problems.append(f"More than one column has role {role!r}")
        for c, r in self.data.roles.items():
            if r not in ROLES:
                problems.append(f"Column {c!r} has unknown role {r!r}")
        if self.data.split.mode not in ("column", "random"):
            problems.append("split.mode must be 'column' or 'random'")
        if self.data.split.mode == "random" and not 0 < self.data.split.fraction < 1:
            problems.append("split.fraction must be in (0, 1)")
        for var, vd in self.design.variables.items():
            if vd.kind not in (None, "step", "categorical"):
                problems.append(f"design[{var!r}].kind must be 'step' or 'categorical'")
            if vd.monotone not in (None, "increasing", "decreasing"):
                problems.append(f"design[{var!r}].monotone invalid")
            if isinstance(vd.knots, str) and vd.knots not in ("quantile", "integer"):
                problems.append(
                    f"design[{var!r}].knots must be 'quantile', 'integer' or a list"
                )
        names = [model] if model else list(self.models)
        for name in names:
            cfg = self.models.get(name)
            if cfg is None:
                problems.append(f"No model named {name!r}")
                continue
            if cfg.family not in FAMILIES:
                problems.append(f"{name}: unknown family {cfg.family!r}")
            if not cfg.target:
                problems.append(f"{name}: no target column")
            if cfg.divide_target_by_weight and not cfg.weight:
                problems.append(
                    f"{name}: divide_target_by_weight needs a weight column"
                )
            if not cfg.predictors:
                problems.append(f"{name}: no predictors")
            bad = [p for p in cfg.predictors if self.data.roles.get(p) != "predictor"]
            if bad:
                problems.append(f"{name}: not predictor-role columns: {bad}")
            if cfg.penalty.alpha is None and cfg.penalty.cv is None:
                problems.append(f"{name}: penalty needs alpha or cv")
            for v, d in cfg.monotone.items():
                if d not in ("increasing", "decreasing"):
                    problems.append(f"{name}: monotone[{v!r}] invalid")
        if self.champion is not None and self.champion not in self.models:
            problems.append(f"champion {self.champion!r} is not a model")
        return problems

    # -- (de)serialisation ---------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for m in d["models"].values():
            m["adjustments"] = [
                {
                    "variable": a["variable"],
                    "from": a["from_"],
                    "to": a["to_"],
                    "relativity": a["relativity"],
                }
                for a in m["adjustments"]
            ]
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Project:
        raw = dict(raw)
        version = raw.get("version", PROJECT_VERSION)
        if version > PROJECT_VERSION:
            raise ValueError(
                f"Project version {version} is newer than supported ({PROJECT_VERSION})"
            )
        d = raw.get("data", {})
        data = DataConfig(
            source=DataSource(**d.get("source", {})),
            sample_rows=d.get("sample_rows"),
            sample_seed=d.get("sample_seed", 42),
            renames=dict(d.get("renames", {})),
            roles=dict(d.get("roles", {})),
            types=dict(d.get("types", {})),
            recodes={k: Recode(**v) for k, v in d.get("recodes", {}).items()},
            derived=[Derived(**x) for x in d.get("derived", [])],
            filters=list(d.get("filters", [])),
            split=Split(**d.get("split", {})),
        )
        g = raw.get("design", {})
        design = DesignConfig(
            defaults=DesignDefaults(**g.get("defaults", {})),
            variables={
                k: VariableDesign(**v) for k, v in g.get("variables", {}).items()
            },
        )
        models: dict[str, ModelConfig] = {}
        for name, m in raw.get("models", {}).items():
            m = dict(m)
            penalty = Penalty(**m.pop("penalty", {}))
            adjustments = [
                Adjustment(
                    a["variable"],
                    a.get("from", a.get("from_")),
                    a.get("to", a.get("to_")),
                    a["relativity"],
                )
                for a in m.pop("adjustments", [])
            ]
            models[name] = ModelConfig(penalty=penalty, adjustments=adjustments, **m)
        exploration = raw.get("exploration") or {
            "leakage": {"ignored": [], "acknowledged": []}
        }
        return cls(
            name=raw.get("name", "untitled"),
            version=PROJECT_VERSION,
            data=data,
            design=design,
            models=models,
            champion=raw.get("champion"),
            exploration=exploration,
        )

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> Project:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def copy(self) -> Project:
        return Project.from_dict(self.to_dict())
