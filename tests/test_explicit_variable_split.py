"""An explicit split is one atomic Variables change used by every consumer."""

from copy import deepcopy

import polars as pl
import pytest
from fastapi.testclient import TestClient

from easy_glm.desktop.server import create_app
from easy_glm.workflow import Project, Split, add_split_column


@pytest.fixture
def session():
    project = Project(name="Text split")
    project.data.roles = {"claims": "target", "age": "predictor", "train_test": "split"}
    project.data.split = Split(column="train_test")
    raw = pl.DataFrame(
        {
            "claims": [0, 1, 2, 99],
            "age": [20, 30, 40, 70],
            "train_test": ["TRAIN", "TRAIN", "TRAIN", "test"],
        }
    )
    with TestClient(
        create_app(project, raw, port=8765), base_url="http://127.0.0.1:8765"
    ) as client:
        client.headers["x-easyglm-token"] = client.get("/api/session").json()["token"]
        state = client.get("/api/variables").json()
        yield client, {
            "session_id": state["session_id"],
            "revision": state["revision"],
            "setup": state["setup"],
        }


def mapped(body):
    result = deepcopy(body)
    result["setup"]["split"].update(train_value="TRAIN", holdout_value="test")
    return result


def test_mapping_required_even_for_previously_assigned_split(session):
    client, body = session
    response = client.post("/api/variables/preview", json=body)
    assert response.status_code == 422
    assert "Choose a training value" in response.text
    assert client.get("/api/variables").json()["revision"] == 0
    options = client.post("/api/variables/split-values", json=body)
    assert options.status_code == 200
    assert options.json()["values"] == [
        {"value": "TRAIN", "rows": 3},
        {"value": "test", "rows": 1},
    ]


def test_atomic_mapping_reaches_explore_model_project_and_reload(session):
    client, body = session
    body = mapped(body)
    body["setup"]["split"].pop("holdout_value")
    preview = client.post("/api/variables/preview", json=body)
    assert preview.status_code == 200, preview.text
    assert len(preview.json()["changes"]) == 1
    assert client.get("/api/variables").json()["setup"]["split"]["train_value"] == 1
    applied = client.post("/api/variables/apply", json=body)
    assert applied.status_code == 200, applied.text
    saved = client.get("/api/project").json()
    assert saved["data"]["split"]["holdout_value"] == "test"
    assert Project.from_dict(saved).data.split.train_value == "TRAIN"
    explored = client.get("/api/explore", params={"column": "age"})
    assert explored.status_code == 200, explored.text
    assert explored.json()["training_rows"] == 3
    assert client.get("/api/workbench").json()["counts"] == {"train": 3, "holdout": 1}
    assert client.post("/api/variables/apply", json=body).status_code == 409


@pytest.mark.parametrize(
    "train,holdout", [("unknown", "TRAIN"), ("train", "test"), (None, "test")]
)
def test_bad_mapping_does_not_change_project(session, train, holdout):
    client, body = session
    before = client.get("/api/project").json()
    body["setup"]["split"].update(train_value=train, holdout_value=holdout)
    assert client.post("/api/variables/apply", json=body).status_code == 422
    assert client.get("/api/project").json() == before


def test_rename_uses_source_name_in_draft_and_final_name_in_project(session):
    client, body = session
    body = mapped(body)
    body["setup"]["renames"]["train_test"] = "Dataset"
    response = client.post("/api/variables/apply", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["setup"]["split"]["column"] == "train_test"
    assert client.get("/api/project").json()["data"]["split"]["column"] == "Dataset"
    assert (
        client.get("/api/explore", params={"column": "age"}).json()["training_rows"]
        == 3
    )


@pytest.mark.parametrize(
    "values,train,holdout",
    [
        ([2, 2, 9], 2, 9),
        ([True, True, False], True, False),
        (["Train", "Train", "TEST"], "Train", "TEST"),
        (["", "", "test"], "", "test"),
    ],
)
def test_types_and_arbitrary_labels(values, train, holdout):
    raw = pl.DataFrame({"s": values})
    result = add_split_column(
        raw, Split(column="s", train_value=train, holdout_value=holdout)
    )
    assert result["s"].to_list() == [1, 1, 0]
    assert raw["s"].to_list() == values


@pytest.mark.parametrize("extra", [None, "validation"])
def test_unmapped_and_missing_values_are_not_silently_holdout(extra):
    raw = pl.DataFrame({"s": ["train", "test", extra]})
    with pytest.raises(ValueError, match="missing or unmapped"):
        add_split_column(
            raw, Split(column="s", train_value="train", holdout_value="test")
        )


@pytest.mark.parametrize("labels", [["a", "b", "c"], ["a", "a"], ["a", "b", None]])
def test_binary_column_required_for_automatic_holdout(labels):
    from easy_glm.desktop.splits import split_counts

    project = Project()
    project.data.split = Split(column="s", train_value="a")
    with pytest.raises(ValueError, match="exactly two|missing split"):
        split_counts(project, pl.DataFrame({"s": labels}))


@pytest.mark.parametrize(
    "values,train,holdout",
    [
        ([2, 9], 2, 9),
        ([True, False], True, False),
        (["TRAIN", "test"], "test", "TRAIN"),
    ],
)
def test_holdout_is_derived_and_replaces_previous_mapping(values, train, holdout):
    from easy_glm.desktop.splits import split_counts

    project = Project()
    project.data.split = Split(column="s", train_value=train, holdout_value=train)
    assert split_counts(project, pl.DataFrame({"s": values})) == {
        "train": 1,
        "holdout": 1,
    }
    assert project.data.split.holdout_value == holdout
