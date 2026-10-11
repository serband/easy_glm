"""Contracts for residual discovery beneath an accepted logical model."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from easy_glm.desktop import core_search
from easy_glm.workflow.project import ModelConfig, Project


class _AcceptedRun:
    """Small accepted scorer: age is already accounted for, Missing is not."""

    def __init__(self, *, family: str = "poisson") -> None:
        self.config = ModelConfig(
            family=family,
            target="Claim",
            weight="Exposure",
        )
        self.fit = SimpleNamespace(coef=np.array([0.1, 0.2]))
        self.seen_rows: list[int] = []

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        self.seen_rows.append(frame.height)
        # This represents the accepted main-effects table.  The search must
        # use this (rather than a bare base rate) before finding what remains.
        return np.where(frame["Age"].to_numpy() == "old", 2.0, 1.0)


def _project_and_frame() -> tuple[Project, pl.DataFrame]:
    project = Project(name="residual discovery")
    project.data.roles = {
        "Claim": "target",
        "Exposure": "weight",
        "traintest": "split",
        "PolicyID": "id",
        "Age": "predictor",
        "Missing": "predictor",
        "Ignored": "ignore",
    }
    # Four balanced training cells. Age is already reflected in the accepted
    # scorer, while Missing has a clear remaining effect. PairOnly deliberately
    # has no role: it is eligible for interaction investigation but never
    # promoted to a GLM main effect by the helper.
    rows: list[dict[str, object]] = []
    for age, base in (("young", 1.0), ("old", 2.0)):
        for missing, multiplier in (("no", 0.5), ("yes", 2.0)):
            for index in range(120):
                rows.append(
                    {
                        "Claim": base * multiplier,
                        "Exposure": 1.0,
                        "traintest": 1,
                        "PolicyID": f"train-{age}-{missing}-{index}",
                        "Age": age,
                        "Missing": missing,
                        "PairOnly": "left" if index % 2 else "right",
                        "Ignored": "control",
                    }
                )
    # A conspicuous holdout pattern would distort a search if it leaked in.
    rows.extend(
        {
            "Claim": 1_000.0,
            "Exposure": 1.0,
            "traintest": 0,
            "PolicyID": f"holdout-{index}",
            "Age": "old",
            "Missing": "holdout-only",
            "PairOnly": "holdout-only",
            "Ignored": "control",
        }
        for index in range(40)
    )
    return project, pl.DataFrame(rows)


def test_factor_search_uses_accepted_training_scorer_and_preserves_roles() -> None:
    project, frame = _project_and_frame()
    before = project.to_dict()
    run = _AcceptedRun()

    result = core_search.discover(
        project,
        frame,
        run,
        action="factors",
        main_factors=["Age"],
        n_bins=6,
    )

    assert run.seen_rows == [480]
    assert result["subset"] == "train"
    assert result["source"] == "Complete accepted scorer on training rows"
    assert result["rows"][0]["variable"] == "Missing"
    assert "Age" not in [row["variable"] for row in result["rows"]]
    assert {"Missing", "PairOnly"} <= set(result["eligible_variables"])
    assert {"Claim", "Exposure", "traintest", "PolicyID", "Ignored"}.isdisjoint(
        result["eligible_variables"]
    )
    assert project.to_dict() == before


def test_interaction_search_excludes_accepted_pairs_and_control_columns(
    monkeypatch,
) -> None:
    project, frame = _project_and_frame()
    run = _AcceptedRun()
    seen: dict[str, object] = {}

    def search(data, variables, actual, expected, weight, *, pairs, **kwargs):
        seen["rows"] = data.height
        seen["variables"] = variables
        seen["pairs"] = pairs
        return pl.DataFrame(
            {
                "pair": ["Age × PairOnly"],
                "a": ["Age"],
                "b": ["PairOnly"],
                "signal": [1.0],
            }
        )

    monkeypatch.setattr(core_search, "residual_pair_search", search)
    result = core_search.discover(
        project,
        frame,
        run,
        action="interactions",
        main_factors=["Age"],
        accepted_pairs=[("Missing", "Age")],
    )

    assert seen["rows"] == 480
    assert seen["variables"] == ["Age", "Missing", "PairOnly"]
    assert ("Age", "Missing") not in seen["pairs"]
    assert ("Age", "PairOnly") in seen["pairs"]
    assert result["rows"][0]["pair"] == "Age × PairOnly"


def test_gaussian_search_refuses_count_scale_pearson_signal() -> None:
    project, frame = _project_and_frame()
    with pytest.raises(ValueError, match="not available for Gaussian"):
        core_search.discover(
            project,
            frame,
            _AcceptedRun(family="gaussian"),
            action="factors",
            main_factors=["Age"],
        )
