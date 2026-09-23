"""Immutable named model checkpoints for the interactive pricing facade."""

from __future__ import annotations

import copy
import re
from collections.abc import Sequence
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from easy_glm.workflow.prep import train_holdout
from easy_glm.workflow.project import (
    PairSearchConfig,
    PairStageConfig,
    Project,
    validate_model_name,
)
from easy_glm.workflow.run import ModelRun, exposure_for

from .validation import (
    compare_models,
    metric_row,
    validate_factors,
    validate_holdout_models,
)


class PricingModel:
    """A named, detached fitted checkpoint.

    Methods create new checkpoints or read this one.  Session setting changes
    never rewrite the Project, configuration or scorer stored here.
    """

    def __init__(
        self,
        session: Any,
        project: Project,
        run: ModelRun,
        *,
        history: list[dict[str, Any]] | None = None,
        evidence: dict[str, Any] | None = None,
        frozen_stages: list[str] | None = None,
    ) -> None:
        if run.name not in project.models:
            raise ValueError(f"Project has no configuration for model {run.name!r}")
        self._session = session
        self._project = Project.from_dict(project.to_dict())
        # A shallow run copy retains the fitted estimator/spec, which are treated
        # as immutable; every mutable public artefact is detached explicitly.
        self._run = copy.copy(run)
        self._run.config = self._project.models[run.name]
        self._run.rate_model = run.rate_model.clone()
        self._run.tables = {name: table.clone() for name, table in run.tables.items()}
        self._run.metrics = copy.deepcopy(run.metrics)
        self._run.project_snapshot = self._project.to_dict()
        self._run.pair_stages = copy.deepcopy(run.pair_stages)
        self._history = copy.deepcopy(history or [])
        self._evidence = copy.deepcopy(evidence or {})
        self._frozen_stages = list(frozen_stages or [])

    @property
    def name(self) -> str:
        return self._run.name

    @property
    def history(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._history)

    @property
    def evidence(self) -> dict[str, Any]:
        return copy.deepcopy(self._evidence)

    def _frame(self, subset: str = "train") -> pl.DataFrame:
        """Prepared rows from the session's one fixed split."""
        data = getattr(self._session, "_data", None)
        if data is None:
            raise ValueError(
                "This saved model has no analysis data attached. Load it with "
                "PricingModel.load(path, data=...) to analyse, edit, or refit it."
            )
        train, holdout = train_holdout(data, self._project.data.split)
        if subset == "train":
            return train
        if subset == "holdout":
            return holdout
        if subset == "all":
            return data
        raise ValueError("subset must be 'train', 'holdout' or 'all'")

    def refit(
        self,
        name: str,
        *,
        add: Sequence[str] | None = None,
        remove: Sequence[str] | None = None,
        factors: Sequence[str] | None = None,
        rebuild_main: bool = False,
    ) -> PricingModel:
        """Explicitly refit all main effects using current session band settings."""
        if factors is not None and (add or remove):
            raise ValueError("Use factors=..., or add/remove, not both")
        project = Project.from_dict(self._project.to_dict())
        if (
            self._run.config.pair_stages or self._run.config.adjustments
        ) and not rebuild_main:
            raise ValueError(
                "This checkpoint has pair stages or manual rate edits. Main refitting "
                "would remove them; pass rebuild_main=True to make that choice explicit."
            )
        # This is the one operation where later session band choices are meant
        # to enter a model. Appending a pair uses the snapshot unchanged.
        data = self._frame("all")
        project.design = copy.deepcopy(self._session._project.design)
        current = list(self._run.config.predictors)
        selected = list(factors) if factors is not None else current
        for factor in add or []:
            if factor not in selected:
                selected.append(factor)
        removed = set(remove or [])
        selected = [factor for factor in selected if factor not in removed]
        selected = validate_factors(project, data, selected)
        problem = validate_model_name(name, project.models)
        if problem:
            raise ValueError(problem)
        config = copy.deepcopy(self._run.config)
        config.predictors = selected
        config.interactions = []
        config.pair_method = "legacy_glm"
        config.pair_stages = []
        config.adjustments = []
        config.snapshots = []
        config.base_rate_override = None
        for factor in selected:
            project.data.roles[factor] = "predictor"
        project.models[name] = config
        history = [
            *self._history,
            {
                "action": "refit",
                "from": self.name,
                "name": name,
                "factors": list(selected),
            },
        ]
        return self._session._fit(project, name, history=history)

    def fit_interaction(
        self,
        a: str,
        b: str,
        *,
        name: str,
        trials: int = 8,
        prefix_trials: int = 4,
        min_weight_share: float = 0.001,
        seed: int | None = None,
        time_limit_minutes: float | None = None,
    ) -> PricingModel:
        """Append one CatBoost pair on this checkpoint's frozen table scorer.

        ``time_limit_minutes`` covers the shared main fit and all interaction
        validation/tuning work; when omitted, the checkpoint's limit is inherited.
        """
        if a == b:
            raise ValueError("An interaction needs two different variables")
        if trials < 1 or prefix_trials < 1:
            raise ValueError("trials and prefix_trials must be positive")
        if not 0 <= float(min_weight_share) < 1:
            raise ValueError("min_weight_share must be at least 0 and below 1")
        project = Project.from_dict(self._project.to_dict())
        problem = validate_model_name(name, project.models)
        if problem:
            raise ValueError(problem)
        parent_config = copy.deepcopy(self._run.config)
        if time_limit_minutes is not None:
            parent_config.pair_time_limit_minutes = time_limit_minutes
        # Validate against the copied parent config before registering the new name.
        data = self._frame("all")
        project.models[name] = parent_config
        validate_factors(
            project,
            data,
            [a, b],
            pair_only=True,
            model_name=name,
        )
        base = re.sub(r"[^A-Za-z0-9_]+", "_", f"{a}_{b}").strip("_").lower()
        used = {stage.stage_id for stage in parent_config.pair_stages}
        stage_id = base or "pair"
        suffix = 2
        while stage_id in used:
            stage_id = f"{base}_{suffix}"
            suffix += 1
        parent_config.pair_method = "sequential_catboost"
        parent_config.pair_stages.append(
            PairStageConfig(
                stage_id=stage_id,
                a=a,
                b=b,
                min_weight_share=float(min_weight_share),
                seed=int(self._project.data.split.seed if seed is None else seed),
                search=PairSearchConfig(
                    trials=int(trials), prefix_trials=int(prefix_trials)
                ),
            )
        )
        frozen = list(self._frozen_stages)
        history = [
            *self._history,
            {
                "action": "fit_interaction",
                "from": self.name,
                "name": name,
                "a": a,
                "b": b,
            },
        ]
        return self._session._fit(
            project,
            name,
            history=history,
            frozen_stages=frozen,
            frozen_prefix_artifacts=list(self._run.pair_stages),
            frozen_pair_rate_model=self._run.rate_model.clone(),
        )

    def ae(self, a: str, b: str | None = None, *, subset: str = "train"):
        from .views import ae

        return ae(self, a, b, subset=subset)

    def relativities(self, a: str | None = None, b: str | None = None):
        from .views import relativities

        return relativities(self, a, b)

    def find_missing_factors(
        self,
        candidates: Sequence[str] | None = None,
        *,
        n_bins: int = 10,
        min_expected: float = 3.0,
    ):
        from .search import find_missing_factors

        return find_missing_factors(
            self,
            None if candidates is None else list(candidates),
            n_bins=n_bins,
            min_expected=min_expected,
        )

    def find_interactions(
        self,
        candidates: Sequence[str] | None = None,
        *,
        n_bins: int = 8,
        min_expected: float = 3.0,
        min_cell_share: float = 0.0,
        top: int = 20,
    ):
        from .search import find_interactions

        return find_interactions(
            self,
            None if candidates is None else list(candidates),
            n_bins=n_bins,
            min_expected=min_expected,
            min_cell_share=min_cell_share,
            top=top,
        )

    def edit_rates(self, name: str):
        from .review import RateReview

        return RateReview(self, name=name)

    def compare(
        self,
        other: PricingModel | Sequence[PricingModel],
        *,
        subset: str = "train",
        cv: bool = False,
    ) -> pl.DataFrame:
        return compare_models(self, other, subset=subset, cv=cv)

    def validate_holdout(
        self,
        compare_with: PricingModel | Sequence[PricingModel] | None = None,
    ) -> pl.DataFrame:
        return validate_holdout_models(self, compare_with)

    def predict(
        self,
        data: pl.DataFrame | Any | None = None,
        *,
        expected: bool = False,
    ) -> np.ndarray:
        frame = (
            self._frame("train")
            if data is None
            else (data if isinstance(data, pl.DataFrame) else pl.DataFrame(data))
        )
        prediction = self._run.rate_model.predict(frame, exposure_col=None)
        if expected:
            column = exposure_for(self._project, self._run.config)
            if column:
                prediction = prediction * frame[column].cast(pl.Float64).to_numpy()
        return np.asarray(prediction, dtype=float)

    def summary(self) -> dict[str, Any]:
        """Training-only fit summary; holdout stays closed until requested."""
        stages = []
        warnings = list(self._evidence.get("fit_warnings", []))
        for artifact in self._run.pair_stages:
            unsupported = sum(
                cell.fitting_weight <= 0 or cell.fallback_reason is not None
                for cell in artifact.table.cells
            )
            stages.append(
                {
                    "stage_id": artifact.stage_id,
                    "parents": artifact.parents,
                    "status": artifact.status,
                    "cv_evidence": artifact.table.provenance.get(
                        "cv_evidence",
                        (
                            "historical_pre_edit"
                            if artifact.status == "pricing_adjustment"
                            else "current"
                        ),
                    ),
                    "prefix_validation_loss": artifact.prefix_cv_loss,
                    "table_validation_loss": artifact.table_cv_loss,
                    "teacher_validation_loss": artifact.teacher_cv_loss,
                    "table_approximation_loss": artifact.approximation_loss,
                    "table_minus_teacher_validation_loss": (
                        artifact.observed_table_minus_teacher_cv_loss
                    ),
                    "unsupported_cells": unsupported,
                    "cells": len(artifact.table.cells),
                    "reused": artifact.reused,
                }
            )
            if artifact.status != "up_to_date":
                warnings.append(
                    f"Pair stage {artifact.stage_id}: {artifact.status.replace('_', ' ')}"
                )
        if self._run.dropped_predictors:
            warnings.append(
                "Dropped constant/all-null factors: "
                + ", ".join(self._run.dropped_predictors)
            )
        return {
            "name": self.name,
            "family": self._run.config.family,
            "factors": list(self._run.config.predictors),
            "pair_stages": stages,
            "training": (
                metric_row(self, "train")
                if getattr(self._session, "_data", None) is not None
                else copy.deepcopy(self._run.metrics.get("train", {}))
            ),
            "dropped_factors": list(self._run.dropped_predictors),
            "fit_warnings": warnings,
            "history": self.history,
        }

    def __repr__(self) -> str:
        pairs = len(self._run.config.pair_stages)
        return (
            f"PricingModel(name={self.name!r}, family={self._run.config.family!r}, "
            f"factors={len(self._run.config.predictors)}, pair_stages={pairs})"
        )

    def _repr_html_(self) -> str:
        summary = self.summary()
        training = summary["training"]
        ae = training.get("ae")
        deviance = training.get("mean_deviance", training.get("deviance"))
        metric_html = (
            f"<p>Training A/E {float(ae):.4f} · mean deviance "
            f"{float(deviance):.6g}</p>"
            if ae is not None and deviance is not None
            else "<p>Saved training metrics available in summary().</p>"
        )
        warnings = summary["fit_warnings"]
        warning_html = (
            "<ul>"
            + "".join(f"<li>{escape(message)}</li>" for message in warnings)
            + "</ul>"
            if warnings
            else ""
        )
        return (
            f"<h3>{escape(self.name)}</h3>"
            f"<p>{escape(self._run.config.family.title())} · "
            f"{len(self._run.config.predictors)} main effects · "
            f"{len(self._run.config.pair_stages)} pair stages</p>"
            f"{metric_html}{warning_html}"
        )

    def save(self, path: str | Path) -> Path:
        from .persistence import save_model

        return save_model(self, path)

    @classmethod
    def load(cls, path: str | Path, *, data: pl.DataFrame | Any | None = None):
        from .persistence import load_model

        return load_model(path, data=data)

    def to_excel(self, path: str | Path) -> Path:
        from .excel import export_excel

        return export_excel(self, path)
