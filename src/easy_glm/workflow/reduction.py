"""Create a smaller model definition without changing the source model or data."""

from __future__ import annotations

from copy import deepcopy

from .project import Project, validate_model_name


def reduced_challenger(
    project: Project, source: str, name: str, predictors: list[str]
) -> Project:
    """Copy fitting settings, keep selected factors and drop dependent interactions.

    The fresh fit has no inherited rate overrides, adjustments or snapshots.
    Predictor order follows the source design, not the importance ranking.
    """
    if source not in project.models:
        raise ValueError("Choose an existing fitted model.")
    problem = validate_model_name(name, project.models)
    if problem:
        raise ValueError(problem)
    original = project.models[source]
    keep = set(predictors)
    if not keep or len(keep) != len(predictors):
        raise ValueError("Keep at least one predictor, with no duplicates.")
    if not keep.issubset(original.predictors):
        raise ValueError("Keep only predictors from the original model.")
    if len(keep) >= len(original.predictors):
        raise ValueError(
            "Remove at least one predictor to create a smaller challenger."
        )
    candidate = deepcopy(project)
    cfg = deepcopy(original)
    cfg.predictors = [p for p in original.predictors if p in keep]
    cfg.interactions = [i for i in cfg.interactions if i.a in keep and i.b in keep]
    # A parent used only by a pair is still an eligible input after reducing
    # GLM mains. If a selected main parent was removed, its dependent stages
    # are removed too, while the original model and its stage order are intact.
    removed_mains = set(original.predictors) - keep
    dropped_pairs = [
        stage
        for stage in cfg.pair_stages
        if stage.a in removed_mains or stage.b in removed_mains
    ]
    cfg.pair_stages = [
        stage
        for stage in cfg.pair_stages
        if stage.a not in removed_mains and stage.b not in removed_mains
    ]
    cfg.monotone = {p: direction for p, direction in cfg.monotone.items() if p in keep}
    cfg.adjustments = []
    cfg.snapshots = []
    cfg.base_rate_override = None
    cfg.notes = f"Reduced from {source}; freshly fitted with {len(keep)} predictors."
    if dropped_pairs:
        dropped = ", ".join(
            f"{stage.stage_id} ({stage.a} × {stage.b})" for stage in dropped_pairs
        )
        cfg.notes += f" Dropped pair stages whose main parent was removed: {dropped}."
    candidate.models[name] = cfg
    problems = candidate.validate(name)
    if problems:
        raise ValueError("; ".join(problems))
    return candidate
