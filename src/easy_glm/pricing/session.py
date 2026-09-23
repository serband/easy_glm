"""A compact, stateful setup object for interactive pricing work."""

from __future__ import annotations

import copy
import hashlib
import json
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from easy_glm.core.design import CategoricalEncoder, StepEncoder, frequent_levels
from easy_glm.workflow.prep import prepare, train_holdout
from easy_glm.workflow.project import (
    ModelConfig,
    Penalty,
    Project,
    Split,
    VariableDesign,
    validate_model_name,
)
from easy_glm.workflow.run import build_design, run_model

from .validation import require_column, validate_factors, validate_setup

_GENERATED_SPLIT = "__easy_glm_split__"


def _data_fingerprint(frame: pl.DataFrame, identifier: str | None) -> str:
    """Stable data identity; ID-backed books may be supplied in another order."""
    columns = [
        column for column in frame.columns if not column.startswith(_GENERATED_SPLIT)
    ]
    values = frame.select(columns).hash_rows(seed=0).to_numpy()
    if identifier:
        values = np.sort(values)
    schema = [(column, str(frame.schema[column])) for column in columns]
    digest = hashlib.sha256(json.dumps(schema).encode("utf-8"))
    digest.update(np.asarray(values, dtype=np.uint64).tobytes())
    return digest.hexdigest()


def _add_stable_id_split(
    frame: pl.DataFrame,
    *,
    identifier: str,
    column: str,
    fraction: float,
    seed: int,
) -> pl.DataFrame:
    """Assign whole policy IDs reproducibly, independently of input row order."""
    if frame[identifier].null_count():
        raise ValueError(
            f"ID column {identifier!r} contains missing values; a stable random split "
            "requires every row to have an ID"
        )
    keys = frame[identifier].cast(pl.String).alias("__id_key__")
    keyed = frame.with_columns(keys)
    unique = keyed["__id_key__"].unique().sort().to_list()
    if len(unique) < 2:
        raise ValueError(
            "A random train/holdout split requires at least two unique IDs"
        )
    count = min(len(unique) - 1, max(1, int(round(fraction * len(unique)))))
    order = np.random.default_rng(seed).permutation(len(unique))
    training = {unique[index] for index in order[:count]}
    return keyed.with_columns(
        pl.col("__id_key__").is_in(training).cast(pl.Int64).alias(column)
    ).drop("__id_key__")


def _require_nonempty_split(frame: pl.DataFrame, project: Project) -> None:
    train, holdout = train_holdout(frame, project.data.split)
    if train.is_empty() or holdout.is_empty():
        raise ValueError(
            "The split must contain at least one training row and one holdout row; "
            "adjust train_fraction or provide an explicit split column"
        )


class PricingSession:
    """Data, roles, split and mutable *future-fit* settings for a pricing study.

    Fitted :class:`~easy_glm.pricing.PricingModel` objects detach these settings,
    so changing bands here cannot alter an existing checkpoint.
    """

    def __init__(
        self,
        data: pl.DataFrame | Any,
        *,
        family: str = "poisson",
        claims: str | None = None,
        exposure: str | None = None,
        target: str | None = None,
        weight: str | None = None,
        offset: str | None = None,
        id: str | None = None,
        split: str | None = None,
        train_fraction: float = 0.7,
        seed: int = 42,
        ignored: Iterable[str] | None = None,
        divide_target_by_weight: bool | None = None,
        link: str | None = None,
        tweedie_power: float = 1.5,
    ) -> None:
        frame = data.clone() if isinstance(data, pl.DataFrame) else pl.DataFrame(data)
        if claims is not None and target is not None and claims != target:
            raise ValueError("Use either claims or target, not two different columns")
        target = target or claims
        target = require_column(frame, target, "Claims/target")
        if exposure is not None and weight is not None and exposure != weight:
            raise ValueError("Use exposure for frequency, or weight for another model")
        fitting_weight = exposure or weight
        frequency = claims is not None or exposure is not None
        divide = (
            frequency if divide_target_by_weight is None else divide_target_by_weight
        )
        ignored_values = list(ignored or [])
        validate_setup(
            frame,
            family=family,
            target=target,
            weight=fitting_weight,
            offset=offset,
            identifier=id,
            split=split,
            ignored=ignored_values,
            divide_target_by_weight=bool(divide),
        )
        if not 0 < float(train_fraction) < 1:
            raise ValueError("train_fraction must be strictly between 0 and 1")

        project = Project(name="Interactive pricing")
        project.data.roles[target] = "target"
        if exposure:
            project.data.roles[exposure] = "exposure"
        elif weight:
            project.data.roles[weight] = "weight"
        if offset:
            project.data.roles[offset] = "offset"
        if id:
            project.data.roles[id] = "id"
        for column in ignored_values:
            project.data.roles[column] = "ignore"
        if split:
            project.data.roles[split] = "split"
            project.data.split = Split(
                mode="column",
                column=split,
                train_value=1,
                holdout_value=0,
                seed=int(seed),
            )
        else:
            split_name = _GENERATED_SPLIT
            while split_name in frame.columns:
                split_name += "_"
            project.data.roles[split_name] = "split"
            if id:
                frame = _add_stable_id_split(
                    frame,
                    identifier=id,
                    column=split_name,
                    fraction=float(train_fraction),
                    seed=int(seed),
                )
                project.data.split = Split(
                    mode="column",
                    column=split_name,
                    train_value=1,
                    holdout_value=0,
                    fraction=float(train_fraction),
                    seed=int(seed),
                )
            else:
                project.data.split = Split(
                    mode="random",
                    column=split_name,
                    fraction=float(train_fraction),
                    seed=int(seed),
                )
        self._raw_data = frame
        self._project = project
        self._data = prepare(project, frame)
        _require_nonempty_split(self._data, project)
        self._data_fingerprint = _data_fingerprint(
            frame, project.column_with_role("id")
        )
        self._main_cache: dict[str, Any] = {}
        self._pair_cache: dict[str, Any] = {}
        self._family = family
        self._weight = fitting_weight
        self._divide_target_by_weight = bool(divide)
        self._link = link
        self._tweedie_power = float(tweedie_power)

    def _training(self) -> pl.DataFrame:
        return train_holdout(self._data, self._project.data.split)[0]

    def bands(
        self,
        variable: str | None = None,
        *,
        default: int | None = None,
        number: int | None = None,
        cuts: Sequence[float] | None = None,
    ) -> Any:
        """Set band rules and return their training-only support preview."""
        if default is not None:
            if variable is not None or number is not None or cuts is not None:
                raise ValueError(
                    "default is a session setting; do not combine it with a variable"
                )
            if int(default) < 2:
                raise ValueError("default band count must be at least 2")
            self._project.design.defaults.n_bins = int(default)
            return self._display(
                pl.DataFrame({"setting": ["default_n_bands"], "value": [int(default)]}),
                "Default numeric bands",
            )
        if variable is None:
            raise ValueError("Name a variable, or use bands(default=...)")
        require_column(self._data, variable, "Band")
        if number is not None and cuts is not None:
            raise ValueError("Use either number or cuts for one variable")
        if number is not None and int(number) < 2:
            raise ValueError("number must be at least 2")
        design = VariableDesign(
            kind="step",
            knots=[float(value) for value in cuts] if cuts is not None else "quantile",
            n_bins=int(number) if number is not None else None,
        )
        self._project.design.variables[variable] = design
        train = self._training()
        spec = build_design(
            self._project,
            train,
            [variable],
            weight_col=self._weight,
        )
        encoder = spec[variable]
        if not isinstance(encoder, StepEncoder):
            raise TypeError(f"{variable!r} did not produce numeric bands")
        return self._display(
            self._encoder_preview(train, variable, encoder),
            f"Training band support — {variable}",
        )

    def categories(
        self,
        variable: str,
        *,
        levels: Sequence[Any] | None = None,
        max_levels: int | None = None,
    ) -> Any:
        """Pin a categorical design, including for numeric-coded categories."""
        require_column(self._data, variable, "Categorical")
        train = self._training()
        weight = train[self._weight] if self._weight else None
        chosen = (
            [str(value) for value in levels]
            if levels is not None
            else frequent_levels(
                train[variable],
                min_share=self._project.design.defaults.min_level_share,
                max_levels=max_levels,
                weights=weight,
            )
        )
        if not chosen:
            raise ValueError(f"No usable levels found for {variable!r}")
        self._project.design.variables[variable] = VariableDesign(
            kind="categorical", levels=chosen, max_levels=max_levels
        )
        encoder = CategoricalEncoder(variable, chosen)
        return self._display(
            self._encoder_preview(train, variable, encoder),
            f"Training category support — {variable}",
        )

    @staticmethod
    def _display(table: pl.DataFrame, title: str) -> Any:
        from .views import DisplayResult

        return DisplayResult(table=table, title=title)

    def _encoder_preview(
        self, frame: pl.DataFrame, variable: str, encoder: Any
    ) -> pl.DataFrame:
        codes = encoder.row_index(frame[variable])
        support = (
            frame[self._weight].cast(pl.Float64).to_numpy()
            if self._weight
            else np.ones(frame.height, dtype=float)
        )
        rows = []
        for index, (lower, upper) in enumerate(encoder.rows()):
            rows.append(
                {
                    "variable": variable,
                    "row": index,
                    "from": lower,
                    "to": upper,
                    "exposure": float(support[codes == index].sum()),
                    "rows": int(np.sum(codes == index)),
                }
            )
        return pl.DataFrame(rows)

    def fit_glm(
        self,
        name: str,
        *,
        factors: Sequence[str],
        cv: int = 5,
        n_alphas: int = 20,
        alpha: float | None = None,
        l1_ratio: float = 1.0,
    ):
        """Fit and return a named immutable main-effects checkpoint."""
        project = Project.from_dict(self._project.to_dict())
        factors = validate_factors(project, self._data, factors)
        problem = validate_model_name(name, project.models)
        if problem:
            raise ValueError(problem)
        for factor in factors:
            project.data.roles[factor] = "predictor"
        config = ModelConfig(
            family=self._family,
            tweedie_power=self._tweedie_power,
            link=self._link,
            target=project.target,
            weight=self._weight,
            offset=project.offset_column,
            divide_target_by_weight=self._divide_target_by_weight,
            predictors=list(factors),
            penalty=Penalty(
                alpha=alpha,
                cv=cv,
                n_alphas=n_alphas,
                l1_ratio=float(l1_ratio),
            ),
        )
        project.models[name] = config
        project.champion = project.champion or name
        return self._fit(
            project,
            name,
            history=[{"action": "fit_glm", "name": name, "factors": list(factors)}],
        )

    def _fit(
        self,
        project: Project,
        name: str,
        *,
        history: list[dict[str, Any]] | None = None,
        evidence: dict[str, Any] | None = None,
        frozen_stages: list[str] | None = None,
        replay_pair_adjustments: bool = False,
        frozen_prefix_artifacts: list[Any] | None = None,
        frozen_pair_rate_model: Any | None = None,
    ):
        """Fit on training rows only and detach the resulting checkpoint."""
        from .model import PricingModel

        checkpoint = Project.from_dict(project.to_dict())
        problems = checkpoint.validate(name, columns=self._data.columns)
        if problems:
            raise ValueError("Pricing model is not valid:\n- " + "\n- ".join(problems))
        training = train_holdout(self._data, checkpoint.data.split)[0]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run = run_model(
                checkpoint,
                training,
                name,
                main_effects_cache=self._main_cache,
                pair_stages_cache=self._pair_cache,
                replay_pair_adjustments=replay_pair_adjustments,
                frozen_pair_prefix=frozen_prefix_artifacts,
                frozen_pair_rate_model=frozen_pair_rate_model,
            )
        fit_warnings = list(
            dict.fromkeys(
                f"{item.category.__name__}: {item.message}" for item in caught
            )
        )
        captured_evidence = copy.deepcopy(evidence or {})
        captured_evidence["data_fingerprint"] = self._data_fingerprint
        if fit_warnings:
            captured_evidence["fit_warnings"] = fit_warnings
        self._project.models[name] = copy.deepcopy(checkpoint.models[name])
        for factor in checkpoint.models[name].predictors:
            self._project.data.roles[factor] = "predictor"
        self._project.champion = self._project.champion or name
        return PricingModel(
            self,
            checkpoint,
            run,
            history=history,
            evidence=captured_evidence,
            frozen_stages=frozen_stages,
        )

    def summary(self) -> pl.DataFrame:
        """Roles and candidacy without reading target outcomes from holdout."""
        protected = {
            "target",
            "weight",
            "exposure",
            "offset",
            "current_premium",
            "split",
            "time",
            "id",
            "ignore",
        }
        return pl.DataFrame(
            [
                {
                    "column": column,
                    "role": self._project.data.roles.get(column, "unassigned"),
                    "factor_candidate": self._project.data.roles.get(column)
                    not in protected,
                    "pair_candidate": self._project.data.roles.get(column)
                    in (None, "predictor"),
                }
                for column in self._data.columns
            ]
        )

    def settings(self) -> dict[str, Any]:
        """A detached, JSON-serialisable copy of all current settings."""
        return copy.deepcopy(self._project.to_dict())

    def save_settings(self, path: str | Path) -> Path:
        """Save roles, split, designs and model recipes as portable JSON."""
        destination = Path(path)
        payload = {
            "format": "easy-glm-pricing-settings",
            "project": self._project.to_dict(),
            "session": {
                "family": self._family,
                "weight": self._weight,
                "divide_target_by_weight": self._divide_target_by_weight,
                "link": self._link,
                "tweedie_power": self._tweedie_power,
                "data_fingerprint": self._data_fingerprint,
            },
        }
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return destination

    @classmethod
    def from_settings(
        cls, data: pl.DataFrame | Any, path: str | Path
    ) -> PricingSession:
        """Restore a session specification against caller-supplied data."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("format") == "easy-glm-pricing-settings":
            supplied = data if isinstance(data, pl.DataFrame) else pl.DataFrame(data)
            identifier = Project.from_dict(raw["project"]).column_with_role("id")
            expected_fingerprint = raw.get("session", {}).get("data_fingerprint")
            if (
                expected_fingerprint
                and _data_fingerprint(supplied, identifier) != expected_fingerprint
            ):
                raise ValueError(
                    "The supplied data does not match the policies used to save these settings"
                )
            self = cls._from_saved(data, Project.from_dict(raw["project"]))
            setup = raw.get("session", {})
            self._family = setup.get("family", self._family)
            self._weight = setup.get("weight", self._weight)
            self._divide_target_by_weight = bool(
                setup.get("divide_target_by_weight", self._divide_target_by_weight)
            )
            self._link = setup.get("link", self._link)
            self._tweedie_power = float(setup.get("tweedie_power", self._tweedie_power))
            return self
        # Also accept an ordinary workbench Project JSON.
        return cls._from_saved(data, Project.from_dict(raw))

    @classmethod
    def _from_saved(cls, data: pl.DataFrame | Any, project: Project) -> PricingSession:
        """Internal reconstruction hook used by the fitted-model loader."""
        self = cls.__new__(cls)
        frame = data.clone() if isinstance(data, pl.DataFrame) else pl.DataFrame(data)
        self._raw_data = frame
        self._project = Project.from_dict(project.to_dict())
        identifier = self._project.column_with_role("id")
        split = self._project.data.split
        if (
            split.mode == "column"
            and split.column.startswith(_GENERATED_SPLIT)
            and split.column not in frame.columns
            and identifier
        ):
            frame = _add_stable_id_split(
                frame,
                identifier=identifier,
                column=split.column,
                fraction=split.fraction,
                seed=split.seed,
            )
        self._data = prepare(self._project, frame)
        _require_nonempty_split(self._data, self._project)
        self._data_fingerprint = _data_fingerprint(frame, identifier)
        self._main_cache = {}
        self._pair_cache = {}
        exemplar = next(iter(project.models.values()), None)
        self._family = exemplar.family if exemplar else "poisson"
        self._weight = (
            exemplar.weight if exemplar else (project.exposure or project.weight)
        )
        self._divide_target_by_weight = (
            exemplar.divide_target_by_weight
            if exemplar
            else project.exposure is not None
        )
        self._link = exemplar.link if exemplar else None
        self._tweedie_power = exemplar.tweedie_power if exemplar else 1.5
        return self
