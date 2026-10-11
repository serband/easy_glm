"""Score accepted EasyGLM artifacts on a new book without fitting anything."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl

from easy_glm.engine.rate_model import RateModel

from .prep import apply_variables, eval_expr
from .stack_types import ConstantScorer, LookupScorer, RateModelScorer, StackScorer

Scope = Literal["full", "through", "model", "catalogue"]


def _frame(data: pl.DataFrame | Any) -> pl.DataFrame:
    return data if isinstance(data, pl.DataFrame) else pl.DataFrame(data)


def _fingerprint(value: Any) -> str:
    """Stable SHA-256 for saved configuration, not a fit or a pickle."""
    if is_dataclass(value):
        value = asdict(cast(Any, value))
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dataset_digest(frame: pl.DataFrame) -> str:
    """A deterministic digest of exactly the unmodified rows supplied to score."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            [(name, str(dtype)) for name, dtype in frame.schema.items()]
        ).encode()
    )
    # Hashing rows avoids a JSON-sized duplicate of a large scoring book while
    # retaining order and values (including string identifiers such as "001").
    digest.update(
        np.asarray(frame.hash_rows(seed=0).to_numpy(), dtype=np.uint64).tobytes()
    )
    return digest.hexdigest()


def _scorer_fingerprint(scorer: RateModel | StackScorer) -> str:
    return (
        scorer.fingerprint()
        if isinstance(scorer, StackScorer)
        else _fingerprint(scorer.to_dict())
    )


def _preparation_fingerprint(project: Any | None) -> str | None:
    if project is None:
        return None
    return _fingerprint(project.data)


def _prepare(
    frame: pl.DataFrame, project: Any | None, required: set[str] | None = None
) -> pl.DataFrame:
    """Apply frozen column transforms, deliberately excluding row selection."""
    if project is None:
        return frame
    data = copy.deepcopy(project.data)
    data.filters = []
    wanted = set(required or ())
    # New-book scoring must not evaluate unrelated derived columns (especially
    # log(current premium)) merely because they exist in a fitting project.
    # Work backwards through derived columns so a selected frozen derivation
    # brings only its own inputs (including earlier derivations), never every
    # project calculation.  ``root_names`` is Polars' parsed expression view,
    # rather than a fragile textual search of a user expression.
    selected_derived: set[str] = set()
    for item in reversed(data.derived):
        if item.name in wanted:
            selected_derived.add(item.name)
            wanted.update(eval_expr(item.expr).meta.root_names())
    data.renames = {
        old: new for old, new in data.renames.items() if old in wanted or new in wanted
    }
    data.recodes = {
        name: value for name, value in data.recodes.items() if name in wanted
    }
    data.types = {name: value for name, value in data.types.items() if name in wanted}
    data.derived = [item for item in data.derived if item.name in selected_derived]
    # ``DataConfig`` has no role helper; dropping this role prevents
    # apply_variables from adding an unrelated premium logarithm.
    data.roles = {
        name: role
        for name, role in data.roles.items()
        if role != "current_premium" or f"log_{name}" in wanted
    }
    return apply_variables(frame, data)


def _selected(stack: StackScorer, component_ids: tuple[str, ...], through: str | None):
    identifier = through or (component_ids[0] if component_ids else None)
    if identifier is None:
        raise ValueError("Select a physical component with through or component_ids")
    component = stack.component(identifier)
    if component.recipe.kind == "baseline" or not any(
        item is component for item in stack.components
    ):
        raise ValueError(
            "Internal baseline and virtual inspection components cannot be selected for scoring"
        )
    return component


def _model_targets(stack: StackScorer, component_ids: tuple[str, ...]) -> list[Any]:
    if not component_ids:
        raise ValueError("Select one or more physical components with component_ids")
    targets = [_selected(stack, (identifier,), None) for identifier in component_ids]
    positions = [stack.components.index(item) for item in targets]
    if len(set(positions)) != len(positions) or positions != list(
        range(min(positions), max(positions) + 1)
    ):
        raise ValueError(
            "Selected model components must be one contiguous accepted group"
        )
    return targets


def _dependency_lookups(
    stack: StackScorer, target: Any, *, native_catalogue: bool = False
) -> list[Any]:
    """Return only earlier catalogue joins that supply target raw features."""
    # A deployed lookup scores solely from its own key.  Its fitted attributes
    # are historical training inputs, not a dependency of accepted-file
    # scoring.  The raw retained teacher is the only path that needs them.
    if isinstance(target.scorer, LookupScorer) and not native_catalogue:
        return []
    before = stack.components[: stack.components.index(target)]
    needed = set(target.recipe.factors)
    selected: list[Any] = []
    for artifact in reversed(before):
        scorer = artifact.scorer
        if isinstance(scorer, LookupScorer) and needed.intersection(scorer.attributes):
            selected.append(artifact)
            needed.add(scorer.key)
    return list(reversed(selected))


def _requirements(
    scorer: RateModel | StackScorer, scope: Scope, target: Any | None
) -> set[str]:
    if isinstance(scorer, RateModel):
        rate_required = set(scorer.variables)
        rate_required.update(
            parent for table in scorer.pair_tables for parent in table.parents
        )
        if scorer.metadata.offset_col:
            rate_required.add(scorer.metadata.offset_col)
        return rate_required
    if scope == "full":
        full_required: set[str] = set()
        full_supplied: set[str] = set()
        for artifact in scorer.components:
            if isinstance(artifact.scorer, LookupScorer):
                full_required.add(artifact.scorer.key)
                full_supplied.update(artifact.scorer.attributes)
            else:
                full_required.update(_artifact_factors(artifact) - full_supplied)
        if scorer.offset_col:
            full_required.add(scorer.offset_col)
        return full_required
    elif scope == "through":
        if target is None:
            components = scorer.components
        else:
            components = scorer.components[: scorer.components.index(target) + 1]
    elif target is not None:
        components = [
            *_dependency_lookups(scorer, target, native_catalogue=scope == "catalogue"),
            target,
        ]
    else:
        components = []
    scoped_required: set[str] = set()
    scoped_supplied: set[str] = set()
    for artifact in components:
        if isinstance(artifact.scorer, LookupScorer):
            scoped_required.add(artifact.scorer.key)
            # A deployed lookup scores by key.  Its attributes are only needed
            # for a fresh native catalogue/master candidate.
            if scope == "catalogue" and artifact is target:
                scoped_required.update(set(artifact.recipe.factors) - scoped_supplied)
            scoped_supplied.update(artifact.scorer.attributes)
        else:
            scoped_required.update(_artifact_factors(artifact) - scoped_supplied)
    if scope in {"full", "through"} and scorer.offset_col:
        scoped_required.add(scorer.offset_col)
    return scoped_required


def _artifact_factors(artifact: Any) -> set[str]:
    """Raw columns required by an accepted physical scorer."""
    if isinstance(artifact.scorer, RateModelScorer):
        result = set(artifact.scorer.rate_model.variables)
        result.update(
            parent
            for table in artifact.scorer.rate_model.pair_tables
            for parent in table.parents
        )
        return result
    return set(artifact.recipe.factors)


def validate_scoring_data(
    scorer: RateModel | StackScorer,
    data: pl.DataFrame | Any,
    *,
    project: Any | None = None,
    scope: Scope = "full",
    component_ids: tuple[str, ...] = (),
    through: str | None = None,
    exposure_col: str | None = None,
    column_prefix: str = "easyglm_",
) -> dict[str, Any]:
    """Return JSON-friendly new-book requirements and blocking issues."""
    issues: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    coverage: list[dict[str, Any]] = []
    if scope not in {"full", "through", "model", "catalogue"}:
        return {
            "requirements": {},
            "issues": [
                {
                    "code": "scope",
                    "message": "scope must be full, through, model, or catalogue",
                }
            ],
            "warnings": [],
            "lookup_coverage": [],
            "ready": False,
        }
    raw = _frame(data)
    if isinstance(scorer, RateModel) and (
        scope != "full" or through is not None or component_ids
    ):
        return {
            "requirements": {},
            "issues": [
                {
                    "code": "scope",
                    "message": (
                        "Standalone RateModel scoring supports only scope='full' "
                        "without a selected component"
                    ),
                }
            ],
            "warnings": [],
            "lookup_coverage": [],
            "ready": False,
        }
    target = None
    if isinstance(scorer, StackScorer) and scope in {"through", "model", "catalogue"}:
        try:
            target = _selected(scorer, component_ids, through)
        except Exception as exc:
            issues.append({"code": "component", "message": str(exc)})
    targets = []
    if isinstance(scorer, StackScorer) and scope == "model":
        try:
            targets = _model_targets(scorer, component_ids)
        except Exception as exc:
            issues.append({"code": "component", "message": str(exc)})
    if any(issue["code"] == "component" for issue in issues):
        return {
            "requirements": {},
            "issues": issues,
            "warnings": [],
            "lookup_coverage": [],
            "ready": False,
        }
    required = _requirements(scorer, scope, target)
    if targets:
        required = set().union(
            *(_requirements(scorer, scope, item) for item in targets)
        )
    try:
        frame = _prepare(raw, project, required)
    except Exception as exc:  # preparation errors are actionable preflight output
        return {
            "requirements": {},
            "issues": [{"code": "preparation", "message": str(exc)}],
            "warnings": [],
            "lookup_coverage": [],
            "ready": False,
        }
    missing = sorted(required - set(frame.columns))
    if missing:
        issues.append(
            {
                "code": "columns",
                "message": "Missing required columns: " + ", ".join(missing),
            }
        )
    if exposure_col and exposure_col not in frame.columns:
        issues.append(
            {
                "code": "exposure",
                "message": f"Exposure column {exposure_col!r} is missing",
            }
        )
    if scope == "catalogue":
        if (
            not isinstance(scorer, StackScorer)
            or target is None
            or not isinstance(target.scorer, LookupScorer)
        ):
            issues.append(
                {
                    "code": "catalogue",
                    "message": "Catalogue scope requires one accepted catalogue component",
                }
            )
        elif target.scorer.native_scorer is None:
            issues.append(
                {
                    "code": "catalogue_native_unavailable",
                    "message": "This saved lookup has no native catalogue teacher; refit the catalogue component to score a new master.",
                }
            )
        elif target.scorer.key in frame.columns and (
            frame[target.scorer.key].null_count()
            or frame[target.scorer.key].n_unique() != frame.height
        ):
            issues.append(
                {
                    "code": "catalogue_key",
                    "message": f"Candidate catalogue key {target.scorer.key!r} must be present, non-null, and unique",
                }
            )
    existing = [name for name in raw.columns if name.startswith(column_prefix)]
    if existing:
        issues.append(
            {
                "code": "collision",
                "message": "Output columns already exist: " + ", ".join(existing),
            }
        )
    if not missing and isinstance(scorer, RateModel):
        try:
            scorer.predict(frame, exposure_col=None)
        except Exception as exc:
            issues.append({"code": "scoring_type", "message": str(exc)})
    if not missing and isinstance(scorer, StackScorer):
        try:
            candidates = (
                scorer.components
                if scope == "full"
                else (
                    scorer.components[: scorer.components.index(target) + 1]
                    if scope == "through" and target is not None
                    else (
                        [
                            item
                            for selected in targets
                            for item in [
                                *_dependency_lookups(scorer, selected),
                                selected,
                            ]
                        ]
                        if targets
                        else (
                            _dependency_lookups(
                                scorer,
                                target,
                                native_catalogue=scope == "catalogue",
                            )
                            if scope == "catalogue"
                            else _dependency_lookups(scorer, target)
                            + ([target] if target is not None else [])
                        )
                    )
                )
            )
            working = frame
            for artifact in candidates:
                if isinstance(artifact.scorer, LookupScorer):
                    detail = artifact.scorer.coverage(working)
                    detail.update(
                        component_id=artifact.recipe.component_id,
                        component_name=artifact.recipe.name,
                        key=artifact.scorer.key,
                    )
                    coverage.append(detail)
                    if detail["fallback_rows"]:
                        warnings.append(
                            {
                                "code": "lookup_fallback",
                                "message": (
                                    f"{artifact.recipe.name} used its saved fallback for "
                                    f"{detail['fallback_rows']} row(s) with unknown "
                                    f"{artifact.scorer.key} values."
                                ),
                            }
                        )
                    working = artifact.scorer.enrich(working)
                else:
                    artifact.scorer.linear_predictor(working)
            if scope == "catalogue" and target is not None:
                assert isinstance(target.scorer, LookupScorer)
                assert target.scorer.native_scorer is not None
                target.scorer.native_scorer.linear_predictor(working)
            if scope == "full":
                scorer.predict_unit(frame)
            elif scope == "through" and target is not None:
                scorer.prefix_through(target.recipe.component_id).predict_unit(frame)
            elif scope == "model" and targets:
                contribution = np.zeros(frame.height, dtype=float)
                if scorer.components.index(targets[0]) == 1 and isinstance(
                    scorer.components[0].scorer, ConstantScorer
                ):
                    contribution += scorer.components[0].scorer.linear_predictor(frame)
                for selected in targets:
                    contribution += selected.scorer.linear_predictor(working)
                with np.errstate(over="ignore", under="ignore"):
                    values = np.exp(contribution)
                if not np.all(np.isfinite(values) & (values > 0)):
                    raise ValueError("Selected contribution is not finite and positive")
        except Exception as exc:
            issues.append({"code": "scoring", "message": str(exc)})
    if exposure_col and exposure_col in frame.columns:
        exposure_values = frame[exposure_col].cast(pl.Float64, strict=False)
        bad = int(
            (~(exposure_values.is_finite() & (exposure_values >= 0)))
            .fill_null(True)
            .sum()
        )
        if bad:
            issues.append(
                {
                    "code": "exposure",
                    "message": f"Exposure column {exposure_col!r} has {bad} non-finite, missing, or negative value(s)",
                }
            )
    if (
        scope in {"full", "through"}
        and isinstance(scorer, StackScorer)
        and scorer.offset_col
        and scorer.offset_col in frame.columns
    ):
        offset = frame[scorer.offset_col].cast(pl.Float64, strict=False)
        bad = int((~offset.is_finite()).fill_null(True).sum())
        if bad:
            issues.append(
                {
                    "code": "offset",
                    "message": f"Offset column {scorer.offset_col!r} has {bad} missing or non-finite value(s)",
                }
            )
    return {
        "requirements": {
            "columns": sorted(required),
            "prepared_columns": frame.columns,
            "rows": frame.height,
        },
        "issues": issues,
        "warnings": warnings,
        "lookup_coverage": coverage,
        "ready": not issues,
    }


@dataclass
class ScoringResult:
    frame: pl.DataFrame
    metadata: dict[str, Any]
    issues: list[dict[str, str]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)

    def preview(self, rows: int = 20) -> pl.DataFrame:
        return self.frame.head(rows)

    def export(self, path: str | Path) -> Path:
        """Atomically write CSV/Parquet plus a JSON sidecar describing the score."""
        destination = Path(path)
        if destination.suffix.lower() not in {".csv", ".parquet"}:
            raise ValueError("Score export path must end in .csv or .parquet")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        if destination.suffix.lower() == ".csv":
            self.frame.write_csv(temporary)
        else:
            self.frame.write_parquet(temporary)
        os.replace(temporary, destination)
        metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
        metadata_temporary = metadata_path.with_name(f".{metadata_path.name}.tmp")
        metadata_temporary.write_text(
            json.dumps(self.metadata, indent=2, default=str) + "\n"
        )
        os.replace(metadata_temporary, metadata_path)
        return destination


def score_data(
    scorer: RateModel | StackScorer,
    data: pl.DataFrame | Any,
    *,
    project: Any | None = None,
    scope: Scope = "full",
    component_ids: tuple[str, ...] = (),
    through: str | None = None,
    exposure_col: str | None = None,
    column_prefix: str = "easyglm_",
) -> ScoringResult:
    """Score a fresh frame with frozen accepted tables; never fit or filter rows."""
    check = validate_scoring_data(
        scorer,
        data,
        project=project,
        scope=scope,
        component_ids=component_ids,
        through=through,
        exposure_col=exposure_col,
        column_prefix=column_prefix,
    )
    if check["issues"]:
        raise ValueError("; ".join(issue["message"] for issue in check["issues"]))
    raw = _frame(data)
    # Reuse the exact scope requirement set chosen by preflight so unrelated
    # project transformations cannot make scoring fail after validation.
    required = set(check["requirements"]["columns"])
    frame = _prepare(raw, project, required)
    target = None
    targets: list[Any] = []
    if isinstance(scorer, StackScorer) and scope in {"model", "catalogue"}:
        target = _selected(scorer, component_ids, through)
    if isinstance(scorer, StackScorer) and scope == "model":
        targets = _model_targets(scorer, component_ids)
    if isinstance(scorer, RateModel):
        values = scorer.predict(frame, exposure_col=exposure_col)
        label = "expected" if exposure_col else "prediction"
    elif scope == "catalogue":
        assert target is not None and isinstance(target.scorer, LookupScorer)
        working = frame
        for artifact in _dependency_lookups(scorer, target, native_catalogue=True):
            assert isinstance(artifact.scorer, LookupScorer)
            working = artifact.scorer.enrich(working)
        assert target.scorer.native_scorer is not None
        values = np.exp(target.scorer.native_scorer.linear_predictor(working))
        label = "candidate_multiplier"
    elif scope == "model":
        assert targets
        working = frame
        # A conditional root is part of its first correction's deployed level.
        value = np.ones(frame.height, dtype=float)
        index = scorer.components.index(targets[0])
        root_model = (index == 0 and targets[0].recipe.kind in {"glm", "legacy"}) or (
            index == 1 and isinstance(scorer.components[0].scorer, ConstantScorer)
        )
        if index == 1 and isinstance(scorer.components[0].scorer, ConstantScorer):
            value *= np.exp(scorer.components[0].scorer.linear_predictor(working))
        dependencies = []
        for selected in targets:
            for artifact in _dependency_lookups(scorer, selected):
                if artifact not in dependencies:
                    dependencies.append(artifact)
        for artifact in dependencies:
            assert isinstance(artifact.scorer, LookupScorer)
            working = artifact.scorer.enrich(working)
        for selected in targets:
            value *= np.exp(selected.scorer.linear_predictor(working))
        # The first selected model is the fitted rate/frequency itself.  Later
        # selected models are corrections to an already fitted score.
        label = "prediction" if root_model else "multiplier"
        values = value
    else:
        selected_scorer = (
            scorer.prefix_through(through) if scope == "through" else scorer
        )
        values = selected_scorer.predict(frame, exposure_col=exposure_col)
        label = "expected" if exposure_col else "prediction"
    output = raw.with_columns(
        pl.Series(f"{column_prefix}{label}", np.asarray(values, dtype=float))
    )
    metadata = {
        "scope": scope,
        "rows": raw.height,
        "output_column": f"{column_prefix}{label}",
        "through": through,
        "component_ids": list(component_ids),
        "scored_component_ids": (
            [item.recipe.component_id for item in targets]
            if scope == "model"
            else ([target.recipe.component_id] if target is not None else [])
        ),
        "exposure_col": exposure_col,
        "requirements": check["requirements"],
        "lookup_coverage": check["lookup_coverage"],
        "warnings": check["warnings"],
        "scorer_fingerprint": _scorer_fingerprint(scorer),
        "preparation_fingerprint": _preparation_fingerprint(project),
        "dataset_digest": _dataset_digest(raw),
        "source_schema": {name: str(dtype) for name, dtype in raw.schema.items()},
        "candidate_note": (
            "Raw retained-teacher catalogue multiplier; accepted lookup amendments are not applied."
            if scope == "catalogue"
            else None
        ),
    }
    return ScoringResult(output, metadata, warnings=check["warnings"])
