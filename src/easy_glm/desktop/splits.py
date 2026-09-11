"""Split settings shared by the Variables editors and modelling workflow."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import polars as pl

from easy_glm.desktop.modeling import SplitEdit
from easy_glm.workflow.prep import add_split_column, apply_variables, train_holdout
from easy_glm.workflow.project import Project, Split


def split_setup(project: Project) -> dict[str, Any]:
    """Use source names in the Variables draft, like the role assignments."""
    result = asdict(project.data.split)
    if result["mode"] == "column":
        reverse = {new: old for old, new in project.data.renames.items()}
        result["column"] = reverse.get(result["column"], result["column"])
    return result


def apply_split_setup(project: Project, setup: dict[str, Any]) -> None:
    parsed = SplitEdit.model_validate({**setup, "session_id": "draft", "revision": 0})
    values = parsed.model_dump(exclude={"session_id", "revision"})
    if values["mode"] == "column":
        name = values["column"]
        values["column"] = project.data.renames.get(name, name)
        assigned = project.column_with_role("split")
        if assigned and assigned != values["column"]:
            raise ValueError(
                "The split settings must use the column assigned the split role."
            )
    elif project.column_with_role("split"):
        raise ValueError(
            "Remove the existing split role before choosing a random split."
        )
    project.data.split = Split(**values)


def split_counts(project: Project, raw: pl.DataFrame) -> dict[str, int]:
    if project.data.split.mode == "column":
        options = split_values(project, raw)
        if options["distinct"] != 2:
            raise ValueError(
                f'This column has {options["distinct"]} distinct values. '
                "A train/test column must have exactly two."
            )
        if options["missing"]:
            raise ValueError("Correct or filter missing split values before applying.")
        train = project.data.split.train_value

        def matches(value: Any) -> bool:
            if train is None:
                return False
            if isinstance(value, bool):
                return str(value).lower() == str(train).lower()
            if isinstance(value, (int, float)):
                try:
                    return float(value) == float(train)
                except (ValueError, TypeError):
                    return False
            return str(value) == str(train)

        observed = [item["value"] for item in options["values"]]
        selected = [value for value in observed if matches(value)]
        if len(selected) != 1:
            raise ValueError("Choose a training value from this column on Variables.")
        project.data.split.train_value = selected[0]
        project.data.split.holdout_value = next(
            value for value in observed if not matches(value)
        )
    frame = add_split_column(apply_variables(raw, project.data), project.data.split)
    train, holdout = train_holdout(frame, project.data.split)
    return {"train": train.height, "holdout": holdout.height}


def split_values(project: Project, raw: pl.DataFrame) -> dict[str, Any]:
    frame = apply_variables(raw, project.data)
    column = project.data.split.column
    if column not in frame.columns:
        raise ValueError("Choose an existing split column on Variables.")
    series = frame[column]
    missing = series.is_null()
    if series.dtype.is_float():
        missing = missing | ~series.is_finite()
    observed = series.filter(~missing)
    counts = observed.value_counts(name="rows").sort("rows", descending=True)
    return {
        "column": column,
        "values": [
            {"value": row[column], "rows": row["rows"]}
            for row in counts.head(100).to_dicts()
        ],
        "missing": int(missing.sum()),
        "distinct": counts.height,
        "rows": frame.height,
    }
