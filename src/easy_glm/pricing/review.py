"""Draft and apply deliberate pricing-table changes without mutating a model."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go
import polars as pl

from easy_glm.core.excel import rate_model_tables
from easy_glm.engine.models import (
    PairTableConfig,
    VariableConfig,
    level_label,
)
from easy_glm.engine.rate_model import RateModel
from easy_glm.workflow.diagnostics import model_metrics, totals
from easy_glm.workflow.project import Adjustment, ModelConfig, Project, VariableDesign
from easy_glm.workflow.run import (
    ModelRun,
    apply_adjustments,
    null_model_predict,
    run_model,
    snapshot_metrics,
)

if TYPE_CHECKING:
    from .model import PricingModel


@dataclass
class ReviewPreview:
    """Before/after training totals, exact table changes, and a small chart."""

    summary: pl.DataFrame
    changes: pl.DataFrame
    figure: go.Figure
    note: str = ""

    @property
    def table(self) -> pl.DataFrame:
        """Alias for the before/after summary used by notebook display helpers."""
        return self.summary

    def _repr_html_(self) -> str:
        note = f"<p>{escape(self.note)}</p>" if self.note else ""
        return (
            self.figure.to_html(full_html=False, include_plotlyjs="cdn")
            + note
            + self.summary.to_pandas().to_html(index=False)
            + "<h4>Table changes</h4>"
            + self.changes.to_pandas().to_html(index=False)
        )

    def show(self) -> ReviewPreview:
        """Display in IPython when available and return this preview."""
        try:
            from IPython.display import display

            display(self)
        except ImportError:
            self.figure.show()
            print(self.summary)
            print(self.changes)
        return self


def _adjustment_key(adjustment: Adjustment) -> tuple[Any, ...]:
    return (
        adjustment.stage_id,
        adjustment.variable,
        adjustment.from_,
        adjustment.to_,
        adjustment.from_b,
        adjustment.to_b,
    )


def _finite_positive(value: float, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{label} must be a positive finite number")
    return float(value)


def _row_label(row: Any, config: VariableConfig) -> str:
    return level_label(row, config.other_label)


def _resolve_axis_row(
    config: VariableConfig,
    *,
    lower: Any = None,
    upper: Any = None,
    level: str | None = None,
    label: str,
) -> tuple[int, Any]:
    if level is not None:
        matches = [
            (index, row)
            for index, row in enumerate(config.table)
            if _row_label(row, config) == str(level) or row.from_ == level
        ]
    else:
        if lower is None and upper is None:
            raise ValueError(
                f"Address {label} with level=... or with at least one of lower=/upper="
            )
        matches = [
            (index, row)
            for index, row in enumerate(config.table)
            if row.from_ == lower and row.to_ == upper
        ]
    if len(matches) != 1:
        choices = ", ".join(_row_label(row, config) for row in config.table)
        raise ValueError(
            f"{label} does not identify one row; available rows: {choices}"
        )
    return matches[0]


def _pair_for(rate_model: RateModel, a: str, b: str) -> PairTableConfig:
    matches = [
        table for table in rate_model.pair_tables if set(table.parents) == {a, b}
    ]
    if len(matches) != 1:
        raise ValueError(f"No single deployed pair table matches {a!r} × {b!r}")
    return matches[0]


def _metric_summary(
    frame: pl.DataFrame, cfg: ModelConfig, before: RateModel, after: RateModel
) -> pl.DataFrame:
    rows = []
    before_expected_total: float | None = None
    for label, scorer in (("before", before), ("proposed", after)):
        prediction = scorer.predict(frame, exposure_col=None)
        actual, expected, exposure = totals(frame, cfg, prediction)
        expected_total = float(expected.sum())
        if before_expected_total is None:
            before_expected_total = expected_total
        rows.append(
            {
                "version": label,
                "actual": float(actual.sum()),
                "expected": expected_total,
                "ae": float(actual.sum() / expected_total),
                "exposure": float(exposure.sum()),
                "expected_change": expected_total - before_expected_total,
                "expected_change_pct": (
                    expected_total / before_expected_total - 1
                    if before_expected_total > 0
                    else None
                ),
            }
        )
    return pl.DataFrame(rows)


def _preview_chart(summary: pl.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_bar(
        x=summary["version"],
        y=summary["expected"],
        name="expected claims",
        marker_color=["#72808e", "#e07b39"],
    )
    figure.add_scatter(
        x=summary["version"],
        y=summary["actual"],
        name="actual claims",
        mode="lines+markers",
        line={"color": "#1f5f99", "width": 2.5},
    )
    figure.update_layout(
        title="Training totals before and after the proposed rates",
        template="plotly_white",
        height=380,
        yaxis={"title": "claims", "rangemode": "tozero"},
    )
    return figure


class RateReview:
    """A detached draft of manual rate changes for one immutable model."""

    def __init__(self, model: PricingModel, *, name: str = "Reviewed rates") -> None:
        self._model = model
        self.name = str(name).strip() or "Reviewed rates"
        self._project = Project.from_dict(model._project.to_dict())
        self._config = self._project.models[model.name]
        self._edited_keys: set[tuple[Any, ...]] = set()
        self._base_rate_changed = False
        train = model._frame("train")
        before_prediction = model._run.rate_model.predict(train, exposure_col=None)
        _, before_expected, _ = totals(train, model._run.config, before_prediction)
        self._original_expected_total = float(before_expected.sum())

    def _proposed_rate_model(self) -> RateModel:
        scorer = self._model._run.rate_model.clone()
        apply_adjustments(scorer, self._config)
        if self._config.base_rate_override is not None:
            scorer.base_rate = float(self._config.base_rate_override)
        return scorer

    def _upsert(self, adjustment: Adjustment) -> None:
        key = _adjustment_key(adjustment)
        self._config.adjustments = [
            item for item in self._config.adjustments if _adjustment_key(item) != key
        ]
        self._config.adjustments.append(adjustment)
        self._edited_keys.add(key)

    def set_relativity(
        self,
        variable: str,
        *,
        lower: Any = None,
        upper: Any = None,
        level: str | None = None,
        value: float,
    ) -> RateReview:
        """Set one main-table row by its human band edges or displayed level."""
        scorer = self._proposed_rate_model()
        try:
            config = scorer.variables[variable]
        except KeyError as exc:
            raise ValueError(f"{variable!r} is not a fitted main factor") from exc
        if config.type == "interaction":
            raise ValueError("Use set_pair_relativity for a two-way table")
        _, row = _resolve_axis_row(
            config, lower=lower, upper=upper, level=level, label=variable
        )
        self._upsert(
            Adjustment(
                variable=variable,
                from_=row.from_,
                to_=row.to_,
                relativity=_finite_positive(value, "Relativity"),
            )
        )
        return self

    def set_pair_relativity(
        self,
        a: str,
        b: str,
        *,
        lower_a: Any = None,
        upper_a: Any = None,
        level_a: str | None = None,
        lower_b: Any = None,
        upper_b: Any = None,
        level_b: str | None = None,
        value: float,
    ) -> RateReview:
        """Set one deployed pair cell using bands/levels in the requested order."""
        scorer = self._proposed_rate_model()
        table = _pair_for(scorer, a, b)
        if table.parents == (a, b):
            requested = (
                (lower_a, upper_a, level_a),
                (lower_b, upper_b, level_b),
            )
        else:
            requested = (
                (lower_b, upper_b, level_b),
                (lower_a, upper_a, level_a),
            )
        (axis_a_row, row_a), (axis_b_row, row_b) = (
            _resolve_axis_row(
                axis,
                lower=address[0],
                upper=address[1],
                level=address[2],
                label=parent,
            )
            for axis, address, parent in zip(
                table.axes, requested, table.parents, strict=True
            )
        )
        self._upsert(
            Adjustment(
                variable=f"{table.parents[0]}×{table.parents[1]}",
                from_=row_a.from_,
                to_=row_a.to_,
                from_b=row_b.from_,
                to_b=row_b.to_,
                relativity=_finite_positive(value, "Pair relativity"),
                cell=True,
                stage_id=table.stage_id,
                axis_a_row=axis_a_row,
                axis_b_row=axis_b_row,
            )
        )
        return self

    def rebalance(self, expected_total: float | None = None) -> RateReview:
        """Draft a separate base-rate change to a chosen training expected total.

        Omitting ``expected_total`` restores the original model's training total.
        Nothing is changed until :meth:`apply` is called.
        """
        target = (
            self._original_expected_total
            if expected_total is None
            else _finite_positive(expected_total, "Expected total")
        )
        scorer = self._proposed_rate_model()
        if scorer.metadata.link != "log":
            raise ValueError(
                "Rebalancing needs a multiplicative log-link model; with "
                f"the {scorer.metadata.link!r} link, scaling the base rate does "
                "not scale the expected total proportionally"
            )
        train = self._model._frame("train")
        prediction = scorer.predict(train, exposure_col=None)
        _, expected, _ = totals(train, self._config, prediction)
        current = float(expected.sum())
        if current <= 0 or not math.isfinite(current):
            raise ValueError("The proposed model has no positive finite expected total")
        self._config.base_rate_override = float(scorer.base_rate * target / current)
        self._base_rate_changed = True
        return self

    def _changes_table(self, proposed: RateModel) -> pl.DataFrame:
        baseline = self._model._run.rate_model
        rows: list[dict[str, Any]] = []
        for adjustment in self._config.adjustments:
            key = _adjustment_key(adjustment)
            if key not in self._edited_keys:
                continue
            if adjustment.stage_id is None:
                config = baseline.variables[adjustment.variable]
                matches = [
                    row
                    for row in config.table
                    if (row.from_, row.to_) == (adjustment.from_, adjustment.to_)
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"The original {adjustment.variable!r} row is no longer unique"
                    )
                old_row = matches[0]
                location = _row_label(old_row, config)
                old_value = float(old_row.relativity)
            else:
                table = baseline.get_pair_table(adjustment.stage_id)
                if (
                    table.cell_matrix is None
                    or adjustment.axis_a_row is None
                    or adjustment.axis_b_row is None
                ):
                    raise ValueError(
                        f"Pair adjustment {adjustment.stage_id!r} has no stable cell"
                    )
                axis_a_row = adjustment.axis_a_row
                axis_b_row = adjustment.axis_b_row
                old_value = float(table.cell_matrix[axis_a_row, axis_b_row])
                location = (
                    f"{_row_label(table.axes[0].table[axis_a_row], table.axes[0])}"
                    f" × {_row_label(table.axes[1].table[axis_b_row], table.axes[1])}"
                )
            rows.append(
                {
                    "table": adjustment.stage_id or adjustment.variable,
                    "row": location,
                    "before": old_value,
                    "proposed": float(adjustment.relativity),
                }
            )
        if self._base_rate_changed:
            rows.append(
                {
                    "table": "base rate",
                    "row": "all policies",
                    "before": float(baseline.base_rate),
                    "proposed": float(proposed.base_rate),
                }
            )
        return pl.DataFrame(
            rows,
            schema={
                "table": pl.Utf8,
                "row": pl.Utf8,
                "before": pl.Float64,
                "proposed": pl.Float64,
            },
        )

    def preview(self) -> ReviewPreview:
        """Show the draft's training totals and exact table changes without fitting."""
        train = self._model._frame("train")
        proposed = self._proposed_rate_model()
        summary = _metric_summary(
            train, self._config, self._model._run.rate_model, proposed
        )
        changes = self._changes_table(proposed)
        note = (
            "No automatic rebalancing or refitting has occurred. "
            "A/E and expected totals use training rows."
        )
        return ReviewPreview(summary, changes, _preview_chart(summary), note)

    def _new_adjustments(self) -> list[Adjustment]:
        return [
            adjustment
            for adjustment in self._config.adjustments
            if _adjustment_key(adjustment) in self._edited_keys
        ]

    def _affected_stage_index(self) -> int | None:
        stages = self._config.pair_stages
        if not stages:
            return None
        if self._base_rate_changed or any(
            adjustment.stage_id is None for adjustment in self._new_adjustments()
        ):
            return 0
        positions = {stage.stage_id: index for index, stage in enumerate(stages)}
        edited = [
            positions[adjustment.stage_id]
            for adjustment in self._new_adjustments()
            if adjustment.stage_id in positions
        ]
        return min(edited) if edited else None

    def _validate_fold_replay(self, frozen_count: int) -> None:
        if frozen_count >= len(self._config.pair_stages):
            return
        project = self._project
        scorer = self._proposed_rate_model()
        for adjustment in self._config.adjustments:
            if adjustment.stage_id is None:
                config = scorer.variables.get(adjustment.variable)
                if config is None:
                    continue
                design = project.design.variables.get(
                    adjustment.variable, VariableDesign()
                )
                if config.type in {"numeric", "linear"} and not isinstance(
                    design.knots, list | tuple
                ):
                    raise ValueError(
                        f"Cannot refit later interactions after editing {adjustment.variable!r}: "
                        "its bands are data-derived. Specify fixed cuts first"
                    )
                if config.type == "categorical" and not design.levels:
                    raise ValueError(
                        f"Cannot refit later interactions after editing {adjustment.variable!r}: "
                        "its levels are data-derived. Specify fixed levels first"
                    )
                continue
            stage_position = next(
                (
                    index
                    for index, stage in enumerate(self._config.pair_stages)
                    if stage.stage_id == adjustment.stage_id
                ),
                None,
            )
            if stage_position is None or stage_position >= frozen_count:
                continue
            table = scorer.get_pair_table(adjustment.stage_id)
            for parent, axis in zip(table.parents, table.axes, strict=True):
                design = project.design.variables.get(parent, VariableDesign())
                if axis.type == "numeric" and not isinstance(
                    design.knots, list | tuple
                ):
                    raise ValueError(
                        f"Cannot replay pair edit {adjustment.stage_id!r} across folds: "
                        f"{parent!r} has data-derived cuts. Specify fixed cuts first"
                    )
                if axis.type == "categorical" and not design.levels:
                    raise ValueError(
                        f"Cannot replay pair edit {adjustment.stage_id!r} across folds: "
                        f"{parent!r} has data-derived levels. Specify fixed levels first"
                    )

    def _renamed_project(self) -> Project:
        project = Project.from_dict(self._project.to_dict())
        config = copy.deepcopy(project.models[self._model.name])
        config.adjustments = copy.deepcopy(self._config.adjustments)
        config.base_rate_override = self._config.base_rate_override
        project.models[self.name] = config
        project.champion = self.name
        return project

    @staticmethod
    def _mark_historical(run: ModelRun, stage_ids: list[str]) -> None:
        selected = set(stage_ids)
        for artifact in run.pair_stages:
            if artifact.stage_id not in selected:
                continue
            artifact.table = copy.deepcopy(
                run.rate_model.get_pair_table(artifact.stage_id)
            )
            artifact.status = "pricing_adjustment"
            artifact.table.provenance = {
                **artifact.table.provenance,
                "pricing_adjustment": True,
                "cv_evidence": "historical_pre_edit",
            }
        for table in run.rate_model.pair_tables:
            if table.stage_id in selected:
                table.provenance = {
                    **table.provenance,
                    "pricing_adjustment": True,
                    "cv_evidence": "historical_pre_edit",
                }

    def _training_metrics(self, project: Project, run: ModelRun) -> None:
        train = self._model._frame("train")
        prediction = run.rate_model.predict(train, exposure_col=None)
        if run.fit is None:
            from .validation import metric_row

            candidate = copy.copy(self._model)
            candidate._run = run
            train_metrics = metric_row(candidate, "train")
            run.metrics = {"train": train_metrics}
            if run.rate_model.snapshots:
                current = run.rate_model.snapshots[
                    run.rate_model.current_version - 1
                ].metrics
                combined = copy.deepcopy(run.metrics)
                if current and "model" in current:
                    combined["model"] = copy.deepcopy(current["model"])
                run.rate_model.set_snapshot_metrics(combined)
            return
        null_prediction = null_model_predict(project, run.config, train, train)
        run.metrics = model_metrics(
            run.fit,
            {"train": prediction},
            {"train": train},
            run.config,
            {"train": null_prediction},
        )
        run.rate_model.set_snapshot_metrics(
            snapshot_metrics(
                run.fit,
                run.metrics,
                cv_seed=project.data.split.seed,
            )
        )

    def _result(
        self,
        project: Project,
        run: ModelRun,
        *,
        refit: bool,
        frozen_stages: list[str],
    ) -> PricingModel:
        preview = self.preview()
        train = self._model._frame("train")
        applied = _metric_summary(
            train,
            run.config,
            self._model._run.rate_model,
            run.rate_model,
        )
        proposed_before_refit = preview.summary.row(1, named=True)
        history = copy.deepcopy(self._model._history)
        history.append(
            {
                "action": "rate_review",
                "name": self.name,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "refit_later_interactions": refit,
                "changes": preview.changes.to_dicts(),
                "before": applied.row(0, named=True),
                "after": applied.row(1, named=True),
                "proposed_before_refit": proposed_before_refit if refit else None,
            }
        )
        evidence = copy.deepcopy(self._model._evidence)
        prior_validation = evidence.pop("validation", None)
        if prior_validation is not None:
            historical = evidence.get("historical_validation", [])
            if not isinstance(historical, list):
                historical = [historical]
            historical.append(
                {
                    "source_model": self._model.name,
                    "status": "superseded_by_rate_review",
                    "validation": prior_validation,
                }
            )
            evidence["historical_validation"] = historical
        evidence["rate_review"] = {
            "refit_later_interactions": refit,
            "frozen_stages": list(frozen_stages),
            "cv_evidence": dict.fromkeys(frozen_stages, "historical_pre_edit"),
        }
        return type(self._model)(
            self._model._session,
            project,
            run,
            history=history,
            evidence=evidence,
            frozen_stages=frozen_stages,
        )

    def apply(self, *, refit_later_interactions: bool) -> PricingModel:
        """Create a new model, explicitly keeping or refitting affected later pairs."""
        if not self._edited_keys and not self._base_rate_changed:
            raise ValueError("No rate changes have been proposed")
        project = self._renamed_project()
        stages = self._config.pair_stages
        affected = self._affected_stage_index()
        baseline_changed = self._base_rate_changed or any(
            adjustment.stage_id is None for adjustment in self._new_adjustments()
        )
        refit_start = (
            0
            if baseline_changed and stages
            else (affected + 1 if affected is not None else len(stages))
        )
        if (
            not refit_later_interactions
            or affected is None
            or refit_start >= len(stages)
        ):
            run = copy.deepcopy(self._model._run)
            run.name = self.name
            run.config = project.models[self.name]
            run.rate_model = self._proposed_rate_model()
            run.tables = rate_model_tables(run.rate_model)
            run.project_snapshot = project.to_dict()
            frozen = (
                [stage.stage_id for stage in stages[affected:]]
                if affected is not None
                else []
            )
            self._mark_historical(run, frozen)
            self._training_metrics(project, run)
            return self._result(
                project,
                run,
                refit=refit_later_interactions,
                frozen_stages=frozen,
            )

        frozen_count = refit_start
        self._validate_fold_replay(frozen_count)
        frozen_artifacts = copy.deepcopy(self._model._run.pair_stages[:frozen_count])
        train = self._model._frame("train")
        run = run_model(
            project,
            train,
            self.name,
            main_effects_cache=self._model._session._main_cache,
            pair_stages_cache=self._model._session._pair_cache,
            replay_pair_adjustments=True,
            frozen_pair_prefix=frozen_artifacts,
            frozen_pair_rate_model=self._proposed_rate_model(),
        )
        frozen = [
            stage.stage_id
            for stage in stages
            if any(
                adjustment.stage_id == stage.stage_id
                for adjustment in self._config.adjustments
            )
        ]
        self._mark_historical(run, frozen)
        return self._result(
            project,
            run,
            refit=True,
            frozen_stages=frozen,
        )
