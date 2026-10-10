"""Small, durable helpers for a GLM's optional interaction-table drafts.

The plan owns accepted interaction recipes.  Candidates live separately in the
project's workbench metadata until the reviewer explicitly keeps one.  Keeping
that distinction here is intentional: an unfinished CatBoost fit must never be
able to leak into the score used by a later model.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_KEY = "core_interaction_candidates"


def candidates_for(project: Any, model_id: str) -> list[dict[str, Any]]:
    """Return persisted candidates for one logical model, newest first."""
    entries = project.exploration.get(_KEY, [])
    if not isinstance(entries, list):
        return []
    return [deepcopy(entry) for entry in entries if entry.get("model_id") == model_id]


def all_candidates(project: Any) -> list[dict[str, Any]]:
    entries = project.exploration.get(_KEY, [])
    return deepcopy(entries) if isinstance(entries, list) else []


def replace_candidate(project: Any, candidate: dict[str, Any]) -> None:
    """Persist a candidate without changing any accepted model recipe."""
    entries = all_candidates(project)
    identifier = candidate["candidate_id"]
    project.exploration[_KEY] = [
        deepcopy(candidate) if item.get("candidate_id") == identifier else item
        for item in entries
    ]
    if not any(item.get("candidate_id") == identifier for item in entries):
        project.exploration[_KEY].append(deepcopy(candidate))


def remove_candidate(project: Any, candidate_id: str) -> bool:
    entries = all_candidates(project)
    kept = [item for item in entries if item.get("candidate_id") != candidate_id]
    if len(kept) == len(entries):
        return False
    project.exploration[_KEY] = kept
    return True


def public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return candidate state without private worker/project references."""
    keep = {
        "candidate_id",
        "model_id",
        "a",
        "b",
        "recipe",
        "source_model_name",
        "source_fit_id",
        "source_through_component_id",
        "source_fingerprint",
        "status",
        "message",
        "job_name",
        "job_id",
        "review",
        "created_at",
    }
    return {key: deepcopy(value) for key, value in candidate.items() if key in keep}
