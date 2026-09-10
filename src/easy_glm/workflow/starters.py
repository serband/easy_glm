"""Editable, unfitted starter projects for the public insurance examples."""

from __future__ import annotations

from .project import Project


def starter_project(example: str, source_path: str) -> Project:
    """Match the Streamlit starter roles, split and model configuration."""
    if example == "french_motor":
        project = Project(name="French motor claim frequency")
        predictors = ["DrivAge", "Region", "BonusMalus", "Density"]
        project.data.roles = {
            "ClaimNb": "target",
            "Exposure": "weight",
            "IDpol": "id",
            **dict.fromkeys(predictors, "predictor"),
        }
        project.new_model(
            "frequency",
            family="poisson",
            divide_target_by_weight=True,
            predictors=predictors,
        )
    elif example == "swedish_motorcycle":
        project = Project(name="Swedish motorcycle claims cost")
        predictors = ["OwnerAge", "Gender", "Area", "RiskClass", "VehAge", "BonusClass"]
        project.data.roles = {
            "ClaimAmount": "target",
            "Exposure": "weight",
            "ClaimNb": "ignore",
            **dict.fromkeys(predictors, "predictor"),
        }
        project.data.filters = ["pl.col('Exposure') > 0"]
        project.new_model(
            "BurnCost",
            family="tweedie",
            tweedie_power=1.5,
            divide_target_by_weight=True,
            predictors=predictors,
        )
    else:
        raise ValueError("Choose the French motor or Swedish motorcycle example.")
    project.data.source.path = source_path
    project.data.source.type = "parquet"
    project.data.split.mode = "random"
    project.data.split.column = "traintest"
    project.data.split.fraction = 0.7
    project.data.split.seed = 42
    return project
