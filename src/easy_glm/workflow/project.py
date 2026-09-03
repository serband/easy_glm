"""The Project spec: a declarative, JSON-serialisable description of a whole
modelling workflow (data → variables → split → design → models → adjustments).

The GUI edits a :class:`Project`; the engine (:mod:`easy_glm.workflow.run`)
executes it; :mod:`easy_glm.workflow.export` renders it as a Python script.
"""

from __future__ import annotations

import errno
import json
import os
import re
import warnings
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any
from uuid import uuid4

from easy_glm.engine.models import INTERACTION_SEP


def _col_pattern(column: str) -> re.Pattern[str]:
    """Matches ``pl.col('column')`` / ``pl.col("column")`` — that reference and
    nothing else (never a name that merely contains ``column``, never a string
    literal that happens to spell it)."""
    return re.compile(
        r"""pl\.col\(\s*(?P<q>['"])""" + re.escape(column) + r"""(?P=q)\s*\)"""
    )


def rename_in_expression(expr: str, old: str, new: str) -> str:
    """``expr`` with every ``pl.col('old')`` rewritten to ``pl.col('new')``.

    Only the column reference is touched: text, other columns and any name that
    merely contains ``old`` are left exactly as they are.
    """
    quote = '"' if "'" in new else "'"
    return _col_pattern(old).sub(f"pl.col({quote}{new}{quote})", expr)


#: Characters a model name may not contain (names become file and sheet names).
_MODEL_NAME_BAD = set('/\\:*?"<>|') | {chr(c) for c in range(32)}
MODEL_NAME_MAX = 60
#: Windows keeps these names for devices: a file called ``CON.xlsx`` (or
#: ``NUL``, ``COM1`` ...) cannot be created there, whatever the extension, so a
#: model that would be exported under such a name is refused here rather than
#: on the actuary's PC.
_MODEL_NAME_RESERVED = (
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def validate_model_name(name: str, existing: Iterable[str] = ()) -> str | None:
    """Why ``name`` cannot be a model name, or ``None`` if it can.

    Model names appear in file names (persisted fits, downloads) and worksheet
    names, so path separators, control characters and Windows-reserved
    characters are refused; names are compared after stripping whitespace.
    """
    stripped = (name or "").strip()
    if not stripped:
        return "Model name cannot be empty"
    if stripped in (".", ".."):
        return "Model name cannot be '.' or '..'"
    bad = sorted({c for c in stripped if c in _MODEL_NAME_BAD})
    if bad:
        shown = (
            ", ".join(repr(c) for c in bad if c.isprintable()) or "control characters"
        )
        return f"Model name cannot contain {shown}"
    if stripped.split(".")[0].upper() in _MODEL_NAME_RESERVED:
        return (
            f"Model name {stripped!r} is reserved by Windows (CON, NUL, PRN, AUX, "
            "COM1-9, LPT1-9); its downloads could not be saved there"
        )
    if len(stripped) > MODEL_NAME_MAX:
        return f"Model name is longer than {MODEL_NAME_MAX} characters"
    if stripped in set(existing):
        return f"A model named {stripped!r} already exists"
    return None


def safe_filename(name: str, fallback: str = "model") -> str:
    """A file-name-safe version of a model / project name (never a path).

    Model names are already refused if they cannot be file names
    (:func:`validate_model_name`); project names are free text, so anything a
    file system would object to becomes ``_`` and the result is capped at 80
    characters. Used by every download button and by the command line."""
    cleaned = re.sub(r"[^\w.\- ]+", "_", str(name)).strip(" ._")
    return cleaned[:80] if re.search(r"\w", cleaned) else fallback


PROJECT_VERSION = 2  # 1 = easy_glm 0.3 files (loaded and migrated)

ROLES = (
    "target",
    "weight",
    "exposure",
    "offset",
    "current_premium",
    "split",
    "id",
    "predictor",
    "ignore",
)
SINGLE_ROLES = ("target", "weight", "exposure", "offset", "current_premium", "split")

#: Prefix of the column :mod:`easy_glm.workflow.prep` derives from the column
#: with role ``current_premium``: ``log(premium)``, the offset of a rate-change
#: model. Deriving it (rather than letting ``offset`` mean "take the log of
#: this") keeps one rule everywhere — an offset column is *always* already on
#: the linear-predictor scale — and puts the derivation in the exported script
#: as a line of polars anyone can read.
PREMIUM_OFFSET_PREFIX = "log_"


def premium_offset_column(premium: str) -> str:
    """Name of the derived ``log(current premium)`` column for ``premium``."""
    return f"{PREMIUM_OFFSET_PREFIX}{premium}"


SOURCE_TYPES = ("parquet", "csv", "sas7bdat", "xlsx", "ipc")
FAMILIES = ("poisson", "gamma", "tweedie", "gaussian", "binomial", "inverse_gaussian")


def _warn_unknown(raw: dict[str, Any], known: set[str], where: str) -> None:
    unknown = sorted(k for k in raw if k not in known)
    if unknown:
        warnings.warn(
            f"Ignoring unknown project keys {unknown} in {where} (written by a newer "
            "easy_glm?); they will not be written back when the project is saved",
            stacklevel=4,
        )


def _build(cls, raw: dict[str, Any] | None, where: str):
    """Construct a dataclass from a dict, warning about (and dropping) unknown
    keys instead of crashing, so files written by a newer minor version open."""
    raw = dict(raw or {})
    known = {f.name for f in fields(cls)}
    _warn_unknown(raw, known, where)
    return cls(**{k: v for k, v in raw.items() if k in known})


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

    #: ``"step"`` (bands with their own relativity), ``"linear"`` (a continuous
    #: curve whose slope may change at each knot), ``"continuous"`` (one slope
    #: on the raw clamped value — a linear term with no interior knots),
    #: ``"categorical"`` (each value a level) or ``None`` = infer (numeric →
    #: step, everything else → categorical).
    kind: str | None = None
    knots: str | list[float] = "quantile"  # "quantile" | "integer" | explicit list
    #: linear / continuous only: (lo, hi) the value is clipped to;
    #: None = training min/max rounded outward
    clamp: list[float] | None = None
    n_bins: int | None = None
    null_indicator: bool | None = None
    min_level_share: float | None = None
    max_levels: int | None = None
    levels: list[str] | None = None  # explicit levels, first = reference
    monotone: str | None = None  # "increasing" | "decreasing"
    #: how hard the lasso shrinks this variable relative to the rest of the
    #: design: 1.0 = as everything else, 2.0 = twice as hard, **0 = not
    #: penalised at all** (every band or level is kept). See
    #: :func:`easy_glm.core.fit.penalty_weights`.
    penalty_weight: float = 1.0


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
    """A manual relativity override applied after the fit (from the editor).

    For an interaction cell ``variable`` is ``"A×B"``, ``from_``/``to_`` is
    parent A's row and ``from_b``/``to_b`` parent B's row (``cell=True``)."""

    variable: str
    from_: Any
    to_: Any
    relativity: float
    from_b: Any = None
    to_b: Any = None
    cell: bool = False


@dataclass
class TableSnapshot:
    """The rate tables of one model as they stood at a moment, by name.

    A snapshot is a **named copy of the model's manual adjustments**, not of the
    tables themselves: the fit plus a list of adjustments *is* the tables (that
    is what ``rebuild_rate_model`` recompiles, without refitting), so storing the
    adjustments keeps the project the single source of truth and lets a snapshot
    survive a reload, a refit and a rebuilt rate model. Restoring one is putting
    its adjustments back; comparing two is comparing the tables each one gives.
    """

    name: str
    created_at: str = ""
    adjustments: list[Adjustment] = field(default_factory=list)
    #: the base-rate override in force when the snapshot was taken
    base_rate_override: float | None = None


@dataclass
class Interaction:
    """A two-way interaction ``a × b`` on top of the mains ``a`` and ``b``.

    ``alpha`` is the penalty strength of the **second stage** — the fit of the
    interaction cells on top of the frozen mains. The default ``None`` means
    "the same alpha as the mains", which is the honest starting point: a cell
    column costs the same per unit of log adjustment as a main effect that half
    the exposure shares. Set it only to penalise cells differently from the
    mains as a whole; to make one interaction shrink harder than another, use
    ``penalty_weight`` instead, because the second stage is a single fit with a
    single alpha. When several interactions of one model set ``alpha``, the
    **largest** is used (the most cautious of the requests).
    """

    a: str
    b: str
    min_cell_exposure: float = 0.005
    penalty_weight: float = 1.0
    alpha: float | None = None

    @property
    def name(self) -> str:
        return f"{self.a}{INTERACTION_SEP}{self.b}"


@dataclass
class ModelConfig:
    family: str = "poisson"
    #: only for ``family="tweedie"``: the power of the compound Poisson-Gamma,
    #: strictly between 1 (Poisson) and 2 (Gamma). 1.5 is the usual starting
    #: point for a pure-premium model.
    tweedie_power: float = 1.5
    link: str | None = None
    target: str | None = None
    weight: str | None = None
    offset: str | None = None
    divide_target_by_weight: bool = False
    predictors: list[str] = field(default_factory=list)
    penalty: Penalty = field(default_factory=Penalty)
    monotone: dict[str, str] = field(default_factory=dict)
    interactions: list[Interaction] = field(default_factory=list)
    base: str = "modal"
    base_rate_override: float | None = None
    adjustments: list[Adjustment] = field(default_factory=list)
    #: named copies of the adjustments (see :class:`TableSnapshot`); like the
    #: adjustments themselves they are applied after the fit, so they never
    #: invalidate one
    snapshots: list[TableSnapshot] = field(default_factory=list)
    notes: str = ""

    def drop_adjustments_for(self, variable: str) -> int:
        """Remove every adjustment on ``variable`` — from the working set **and
        from every snapshot** — and return how many went.

        One method because a snapshot is a copy of the adjustments: a caller
        that cleaned only the working set (what removing an interaction used to
        do) left a snapshot that could no longer be restored.
        """

        def keep(adjustments: list[Adjustment]) -> tuple[list[Adjustment], int]:
            kept = [a for a in adjustments if a.variable != variable]
            return kept, len(adjustments) - len(kept)

        self.adjustments, dropped = keep(self.adjustments)
        for snap in self.snapshots:
            snap.adjustments, gone = keep(snap.adjustments)
            dropped += gone
        return dropped


def _adjustment_to_dict(a: dict[str, Any]) -> dict[str, Any]:
    """One adjustment (already an ``asdict`` mapping) in the project's JSON
    shape: ``from`` / ``to`` rather than the dataclass's ``from_`` / ``to_``,
    and the cell keys only for an interaction cell."""
    return {
        "variable": a["variable"],
        "from": a["from_"],
        "to": a["to_"],
        "relativity": a["relativity"],
        **(
            {"from_b": a["from_b"], "to_b": a["to_b"], "cell": True}
            if a["cell"]
            else {}
        ),
    }


def _adjustment_from_dict(a: dict[str, Any], where: str) -> Adjustment:
    _warn_unknown(
        a,
        {
            "variable",
            "from",
            "to",
            "from_",
            "to_",
            "relativity",
            "from_b",
            "to_b",
            "cell",
        },
        where,
    )
    return Adjustment(
        a["variable"],
        a.get("from", a.get("from_")),
        a.get("to", a.get("to_")),
        a["relativity"],
        from_b=a.get("from_b"),
        to_b=a.get("to_b"),
        cell=bool(a.get("cell", False)),
    )


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
    def current_premium(self) -> str | None:
        """Column holding the premium charged today (role ``current_premium``).

        Its log is derived by :func:`easy_glm.workflow.prep.apply_variables` as
        :func:`premium_offset_column` and used as the model offset, which is the
        standard rate-review setup: the model then fits the *change* from
        today's premium, so the base rate is the overall rate change and every
        relativity is a multiplier on the current premium.
        """
        return self.column_with_role("current_premium")

    @property
    def offset_column(self) -> str | None:
        """The offset a new model should use: an explicit ``offset`` role if
        there is one, else ``log(current premium)`` when a current-premium
        column is set."""
        if self.offset:
            return self.offset
        premium = self.current_premium
        return premium_offset_column(premium) if premium else None

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
        """Create a model config pre-filled from the roles.

        Raises ``ValueError`` for a name that cannot be used in file and sheet
        names (see :func:`validate_model_name`)."""
        problem = validate_model_name(name, self.models)
        if problem:
            raise ValueError(problem)
        name = name.strip()
        cfg = ModelConfig(
            target=self.target,
            weight=self.weight,
            offset=self.offset_column,
            predictors=list(self.predictors),
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        self.models[name] = cfg
        if self.champion is None:
            self.champion = name
        return cfg

    # -- consistent edits ------------------------------------------------
    def expressions_using(self, column: str) -> list[str]:
        """Row filters and derived-column formulas that reference ``column``
        as ``pl.col('column')`` — what a rename has to follow into."""
        pattern = _col_pattern(column)
        return [
            expr
            for expr in [*self.data.filters, *(d.expr for d in self.data.derived)]
            if pattern.search(expr)
        ]

    def rename_column(self, old: str, new: str) -> list[str]:
        """Rename a (post-rename) column everywhere it is referenced: roles,
        types, recodes, design, split, exploration lists, row filters and
        derived-column formulas (``pl.col('old')`` only, nothing else in the
        expression) and every model (target / weight / offset, predictors,
        monotone, interactions, adjustments). Returns the models that were
        updated."""
        if old == new:
            return []
        self.data.filters = [
            rename_in_expression(f, old, new) for f in self.data.filters
        ]
        for derived in self.data.derived:
            derived.expr = rename_in_expression(derived.expr, old, new)
        stores: tuple[dict[str, Any], ...] = (
            self.data.roles,
            self.data.types,
            self.data.recodes,
        )
        for store in stores:
            if old in store:
                store[new] = store.pop(old)
        if old in self.design.variables:
            self.design.variables[new] = self.design.variables.pop(old)
        if self.data.split.column == old:
            self.data.split.column = new
        leak = self.exploration.get("leakage", {})
        for key in ("ignored", "acknowledged"):
            lst = leak.get(key, [])
            leak[key] = [new if v == old else v for v in lst]
        # a renamed current-premium column renames the log column derived from
        # it, so every model offsetting on that derivation has to follow
        premium_offset = (
            (premium_offset_column(old), premium_offset_column(new))
            if self.data.roles.get(new) == "current_premium"
            else None
        )
        touched: list[str] = []
        for name, cfg in self.models.items():
            hit = False
            for attr in ("target", "weight", "offset"):
                if getattr(cfg, attr) == old:
                    setattr(cfg, attr, new)
                    hit = True
            if premium_offset is not None and cfg.offset == premium_offset[0]:
                cfg.offset = premium_offset[1]
                hit = True
            if old in cfg.predictors:
                cfg.predictors = [new if v == old else v for v in cfg.predictors]
                hit = True
            if old in cfg.monotone:
                cfg.monotone[new] = cfg.monotone.pop(old)
                hit = True
            for it in cfg.interactions:
                if it.a == old:
                    it.a = new
                    hit = True
                if it.b == old:
                    it.b = new
                    hit = True
            snapshot_adjustments = [a for s in cfg.snapshots for a in s.adjustments]
            for adj in [*cfg.adjustments, *snapshot_adjustments]:
                parts = adj.variable.split(INTERACTION_SEP)
                if old in parts:
                    adj.variable = INTERACTION_SEP.join(
                        new if x == old else x for x in parts
                    )
                    hit = True
            if hit:
                touched.append(name)
        return touched

    def apply_role_change(self, column: str, role: str) -> list[str]:
        """Set a role and keep every model consistent; returns plain-language
        notices (a predictor that left a model, an interaction dropped ...)."""
        notices: list[str] = []
        was_premium = self.data.roles.get(column) == "current_premium"
        self.set_role(column, role)
        if was_premium and role != "current_premium":
            # the derived log(premium) column is gone with the role; a model
            # still offsetting on it would fail at the next fit
            gone = premium_offset_column(column)
            for name, cfg in self.models.items():
                if cfg.offset == gone:
                    cfg.offset = None
                    notices.append(
                        f"Model {name} no longer offsets on {gone!r}: {column} is "
                        f"not the current premium any more"
                    )
        if role == "predictor":
            return notices
        for name, cfg in self.models.items():
            if column in cfg.predictors:
                cfg.predictors = [v for v in cfg.predictors if v != column]
                notices.append(
                    f"{column} was removed from model {name}: its role is now {role}"
                )
            dropped = [it for it in cfg.interactions if column in (it.a, it.b)]
            if dropped:
                cfg.interactions = [it for it in cfg.interactions if it not in dropped]
                notices.append(
                    f"Interaction(s) {', '.join(it.name for it in dropped)} removed from "
                    f"model {name}: {column} is no longer a predictor"
                )
            cfg.monotone.pop(column, None)
        return notices

    def missing_columns(
        self, columns: Iterable[str], model: str | None = None
    ) -> list[str]:
        """Problems for column references that are not in ``columns`` (the
        prepared data): roles, split column and every model's references."""
        have = set(columns)
        problems: list[str] = []
        for role in ("target", "weight", "exposure", "offset", "current_premium"):
            col = self.column_with_role(role)
            if col is not None and col not in have:
                problems.append(f"{role} column {col!r} is not in the data")
        if self.data.split.mode == "column" and self.data.split.column not in have:
            problems.append(
                f"split column {self.data.split.column!r} is not in the data"
            )
        names = [model] if model else list(self.models)
        for name in names:
            cfg = self.models.get(name)
            if cfg is None:
                continue
            for attr in ("target", "weight", "offset"):
                col = getattr(cfg, attr)
                if col is not None and col not in have:
                    problems.append(f"{name}: {attr} column {col!r} is not in the data")
            gone = [v for v in cfg.predictors if v not in have]
            if gone:
                problems.append(f"{name}: predictor(s) not in the data: {gone}")
            for it in cfg.interactions:
                for parent in (it.a, it.b):
                    if parent not in have:
                        problems.append(
                            f"{name}: interaction parent {parent!r} is not in the data"
                        )
        return problems

    # -- validation -----------------------------------------------------
    def validate(
        self, model: str | None = None, columns: Iterable[str] | None = None
    ) -> list[str]:
        """Return a list of problems (empty = valid). With ``columns`` (the
        prepared data's columns) every column reference is checked too."""
        problems: list[str] = []
        if columns is not None:
            problems.extend(self.missing_columns(columns, model))
        if self.data.split.mode == "random" and not str(self.data.split.column).strip():
            problems.append("split column name cannot be empty")
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
            if vd.kind not in (None, "step", "linear", "continuous", "categorical"):
                problems.append(
                    f"design[{var!r}].kind must be 'step', 'linear', 'continuous' "
                    "or 'categorical'"
                )
            if vd.monotone not in (None, "increasing", "decreasing"):
                problems.append(f"design[{var!r}].monotone invalid")
            if vd.monotone and vd.kind == "categorical":
                problems.append(
                    f"design[{var!r}]: monotone constraints apply to numeric "
                    "designs only"
                )
            if vd.clamp is not None and (
                len(vd.clamp) != 2 or not float(vd.clamp[0]) < float(vd.clamp[1])
            ):
                problems.append(f"design[{var!r}].clamp must be [lo, hi] with lo < hi)")
            if not (
                isinstance(vd.penalty_weight, int | float)
                and float(vd.penalty_weight) >= 0
                and float(vd.penalty_weight) < float("inf")
            ):
                problems.append(
                    f"design[{var!r}].penalty_weight must be a number >= 0 "
                    "(0 = unpenalised)"
                )
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
            if cfg.family == "tweedie" and not 1.0 < float(cfg.tweedie_power) < 2.0:
                problems.append(
                    f"{name}: tweedie_power must be strictly between 1 and 2 "
                    f"(1 = Poisson, 2 = Gamma), got {cfg.tweedie_power!r}"
                )
            if not cfg.target:
                problems.append(f"{name}: no target column")
            if cfg.divide_target_by_weight and not cfg.weight:
                problems.append(
                    f"{name}: divide_target_by_weight needs a weight column"
                )
            if cfg.target and cfg.weight and cfg.target == cfg.weight:
                problems.append(f"{name}: target and weight are the same column")
            if cfg.target and cfg.offset and cfg.target == cfg.offset:
                problems.append(f"{name}: target and offset are the same column")
            if cfg.target and cfg.target in cfg.predictors:
                problems.append(f"{name}: the target cannot also be a predictor")
            if cfg.penalty.alpha is not None and cfg.penalty.alpha <= 0:
                problems.append(
                    f"{name}: alpha must be > 0 (alpha = 0 is an unpenalised fit "
                    "that the solver cannot handle; use a small value such as 1e-4)"
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
                vd_kind = self.design.variables.get(v)
                if vd_kind is not None and vd_kind.kind == "categorical":
                    problems.append(
                        f"{name}: monotone[{v!r}] — {v!r} is categorical; monotone "
                        "constraints apply to numeric designs only"
                    )
            seen_pairs: set[frozenset[str]] = set()
            for it in cfg.interactions:
                if it.a == it.b:
                    problems.append(f"{name}: interaction {it.a!r} × itself")
                for parent in (it.a, it.b):
                    if parent not in cfg.predictors:
                        problems.append(
                            f"{name}: interaction parent {parent!r} is not one of the "
                            "model's predictors"
                        )
                pair = frozenset((it.a, it.b))
                if pair in seen_pairs:
                    problems.append(f"{name}: interaction {it.name} listed twice")
                seen_pairs.add(pair)
                if not 0 <= it.min_cell_exposure < 1:
                    problems.append(
                        f"{name}: {it.name} min_cell_exposure not in [0, 1)"
                    )
                if it.penalty_weight <= 0:
                    problems.append(f"{name}: {it.name} penalty_weight must be > 0")
                if it.alpha is not None and it.alpha <= 0:
                    problems.append(
                        f"{name}: {it.name} alpha must be > 0 (leave it unset to "
                        "use the mains' alpha)"
                    )
        if self.champion is not None and self.champion not in self.models:
            problems.append(f"champion {self.champion!r} is not a model")
        return problems

    # -- (de)serialisation ---------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for m in d["models"].values():
            m["adjustments"] = [_adjustment_to_dict(a) for a in m["adjustments"]]
            for snap in m.get("snapshots", []):
                snap["adjustments"] = [
                    _adjustment_to_dict(a) for a in snap["adjustments"]
                ]
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Project:
        raw = dict(raw)
        version = int(raw.get("version", 1))
        if version > PROJECT_VERSION:
            raise ValueError(
                f"Project version {version} is newer than supported "
                f"({PROJECT_VERSION}); upgrade easy_glm to open this project"
            )
        if version < PROJECT_VERSION:
            raw = cls._migrate(raw, version)
        _warn_unknown(raw, {f.name for f in fields(cls)}, "project")
        d = raw.get("data", {})
        _warn_unknown(d, {f.name for f in fields(DataConfig)}, "data")
        data = DataConfig(
            source=_build(DataSource, d.get("source", {}), "data.source"),
            sample_rows=d.get("sample_rows"),
            sample_seed=d.get("sample_seed", 42),
            renames=dict(d.get("renames", {})),
            roles=dict(d.get("roles", {})),
            types=dict(d.get("types", {})),
            recodes={
                k: _build(Recode, v, f"data.recodes[{k!r}]")
                for k, v in d.get("recodes", {}).items()
            },
            derived=[_build(Derived, x, "data.derived") for x in d.get("derived", [])],
            filters=list(d.get("filters", [])),
            split=_build(Split, d.get("split", {}), "data.split"),
        )
        g = raw.get("design", {})
        _warn_unknown(g, {f.name for f in fields(DesignConfig)}, "design")
        design = DesignConfig(
            defaults=_build(DesignDefaults, g.get("defaults", {}), "design.defaults"),
            variables={
                k: _build(VariableDesign, v, f"design.variables[{k!r}]")
                for k, v in g.get("variables", {}).items()
            },
        )
        models: dict[str, ModelConfig] = {}
        for name, m in raw.get("models", {}).items():
            m = dict(m)
            penalty = _build(Penalty, m.pop("penalty", {}), f"models[{name!r}].penalty")
            adjustments = [
                _adjustment_from_dict(a, f"models[{name!r}].adjustments")
                for a in m.pop("adjustments", [])
            ]
            snapshots = []
            for snap in m.pop("snapshots", []):
                _warn_unknown(
                    snap,
                    {f.name for f in fields(TableSnapshot)},
                    f"models[{name!r}].snapshots",
                )
                snapshots.append(
                    TableSnapshot(
                        name=str(snap.get("name", "snapshot")),
                        created_at=str(snap.get("created_at", "")),
                        adjustments=[
                            _adjustment_from_dict(
                                a, f"models[{name!r}].snapshots.adjustments"
                            )
                            for a in snap.get("adjustments", [])
                        ],
                        base_rate_override=snap.get("base_rate_override"),
                    )
                )
            interactions = [
                _build(Interaction, it, f"models[{name!r}].interactions")
                for it in m.pop("interactions", [])
            ]
            models[name] = _build(
                ModelConfig,
                {
                    **m,
                    "penalty": penalty,
                    "adjustments": adjustments,
                    "snapshots": snapshots,
                    "interactions": interactions,
                },
                f"models[{name!r}]",
            )
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

    @staticmethod
    def _migrate(raw: dict[str, Any], version: int) -> dict[str, Any]:
        """Upgrade an older project dict in memory to :data:`PROJECT_VERSION`.

        v1 (easy_glm 0.3) → v2: no structural change; v2 readers additionally
        tolerate unknown keys (written by newer minor versions).
        """
        raw = dict(raw)
        raw["version"] = PROJECT_VERSION
        return raw

    def to_json(self, path: str | Path) -> Path:
        """Write the project as JSON. The bytes go to a unique temporary file
        next to ``path`` and are then renamed over it, so a reader (another
        browser tab, the conflict check, a backup) never sees a half-written
        file, and two writers never interleave."""
        path = Path(path)
        if path.exists() and not os.access(path, os.W_OK):
            # the rename below would sail past a read-only file (the *folder*
            # is what os.replace needs); a file the user protected must still
            # refuse the write, and say so the way an ordinary write would
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex[:8]}.tmp")
        try:
            tmp.write_text(json.dumps(self.to_dict(), indent=2, default=str))
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> Project:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def copy(self) -> Project:
        return Project.from_dict(self.to_dict())
