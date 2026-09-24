"""Inspection and export of one accepted pricing component."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from easy_glm.core.excel import rate_model_tables
from easy_glm.workflow.stack_types import LookupScorer, RateModelScorer

from .views import DisplayResult


def legacy_stage_names(model: Any) -> tuple[str | None, dict[str, str]]:
    """Recover public checkpoint names for a legacy main and accepted pairs."""
    history = getattr(model, "_history", [])
    main_name = next(
        (
            entry.get("name")
            for entry in reversed(history)
            if entry.get("action") in {"fit_glm", "refit"}
        ),
        None,
    )
    pair_entries = [
        entry for entry in history if entry.get("action") == "fit_interaction"
    ]
    stage_ids = [stage.stage_id for stage in model._run.config.pair_stages]
    pair_names = {
        stage_id: str(entry["name"])
        for stage_id, entry in zip(stage_ids, pair_entries, strict=False)
        if entry.get("name")
    }
    return (str(main_name) if main_name else None), pair_names


class ComponentView:
    """A named component of a frozen pricing model."""

    def __init__(self, model: Any, identifier: str) -> None:
        self._model = model
        stack = getattr(model._run, "stack", None)
        if stack is None:
            from easy_glm.workflow.stack import stack_from_legacy

            stack = stack_from_legacy(model._project, model._run)
        self._stack = stack
        self._legacy_part: str | None = None
        self._legacy_stage_id: str | None = None
        try:
            self._artifact = stack.component(identifier)
        except KeyError:
            main_name, pair_names = legacy_stage_names(model)
            legacy = next(
                (item for item in stack.components if item.kind == "legacy"), None
            )
            if legacy is None:
                raise
            if identifier == main_name:
                self._artifact = legacy
                self._legacy_part = "main"
            else:
                matches = [
                    stage_id
                    for stage_id, name in pair_names.items()
                    if identifier in (stage_id, name)
                ]
                if len(matches) != 1:
                    raise
                self._artifact = legacy
                self._legacy_part = "pair"
                self._legacy_stage_id = matches[0]
        self._identifier = identifier

    @property
    def unit_label(self) -> str:
        """Describe whether the file starts a rate or corrects an earlier one."""
        components = [
            item for item in self._stack.components if item.kind != "baseline"
        ]
        first = bool(components and components[0] is self._artifact)
        if (
            self._stack.virtual_components
            and components
            and components[0].kind == "legacy"
        ):
            first = bool(self._stack.virtual_components[0] is self._artifact)
        if (
            self.kind in {"catalogue", "glm"}
            and first
            and self._stack.offset_col is None
        ):
            return "unit prediction"
        return "incremental multiplier"

    @property
    def name(self) -> str:
        return self._identifier if self._legacy_part else self._artifact.name

    @property
    def kind(self) -> str:
        return (
            ("glm" if self._legacy_part == "main" else "interaction")
            if self._legacy_part
            else self._artifact.kind
        )

    def coverage(self, data: pl.DataFrame | Any | None = None) -> dict[str, Any]:
        """Count fallback use and missing keys against this pinned catalogue."""
        scorer = self._artifact.scorer
        if not isinstance(scorer, LookupScorer):
            raise ValueError(f"{self.name!r} is not a catalogue lookup")
        frame = (
            self._model._frame("train")
            if data is None
            else data if isinstance(data, pl.DataFrame) else pl.DataFrame(data)
        )
        return {
            **scorer.coverage(frame),
            "component": self.name,
            "value_type": self.unit_label,
            "unseen_attribute_levels": self._artifact.evidence.get(
                "unseen_attribute_levels", {}
            ),
        }

    def show(self) -> DisplayResult:
        """Show deployed values and available training support for this component."""
        scorer = self._artifact.scorer
        if isinstance(scorer, LookupScorer):
            table = scorer.table.clone()
        elif isinstance(scorer, RateModelScorer):
            if self._legacy_part == "pair" or self.kind == "interaction":
                from easy_glm.core.excel import pair_table_frames

                pairs = pair_table_frames(scorer.rate_model)
                table = (
                    pairs[self._legacy_stage_id]
                    if self._legacy_stage_id is not None
                    else next(iter(pairs.values()))
                )
            else:
                frames = [
                    value.with_columns(pl.lit(name).alias("factor"))
                    for name, value in rate_model_tables(scorer.rate_model).items()
                ]
                table = (
                    pl.concat(frames, how="diagonal_relaxed")
                    if frames
                    else pl.DataFrame()
                )
        else:
            table = pl.DataFrame(
                {
                    "name": [self.name],
                    "type": [self.kind],
                    "factors": [", ".join(self._artifact.factors)],
                    "status": [self._artifact.status],
                }
            )
        unseen = self._artifact.evidence.get("unseen_attribute_levels", {})
        unseen_note = f" Unseen catalogue attribute levels: {unseen}." if unseen else ""
        return DisplayResult(
            table,
            title=f"{self._model.name}: {self.name} accepted component",
            note=(
                f"Values are a {self.unit_label}. The complete model multiplies "
                f"all components in order.{unseen_note}"
            ),
        )

    def export_lookup(self, path: str | Path) -> Path:
        """Write the full accepted key and multiplier table."""
        scorer = self._artifact.scorer
        if not isinstance(scorer, LookupScorer):
            raise ValueError(f"{self.name!r} is not a catalogue lookup")
        destination = Path(path)
        exported = scorer.table.with_columns(
            pl.lit(self.unit_label).alias("value_type")
        )
        if destination.suffix.lower() == ".csv":
            exported.write_csv(destination)
        elif destination.suffix.lower() in {".parquet", ".pq"}:
            exported.write_parquet(destination)
        else:
            raise ValueError("Lookup export path must end in .csv or .parquet")
        return destination
