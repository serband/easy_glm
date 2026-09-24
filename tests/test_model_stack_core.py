"""Accepted stack scoring, catalogue ownership, and persistence contracts."""

from __future__ import annotations

import copy

import numpy as np
import polars as pl
import pytest

from easy_glm.engine.models import (
    FromToRow,
    PairCellRow,
    PairTableConfig,
    VariableConfig,
)
from easy_glm.workflow.catalogue import load_catalogue
from easy_glm.workflow.project import (
    ModelConfig,
    PairStageConfig,
    Project,
    VariableDesign,
)
from easy_glm.workflow.run import run_model
from easy_glm.workflow.stack import amend_stack, fit_stack, stack_from_legacy
from easy_glm.workflow.stack_types import ComponentRecipe, LookupScorer, StackScorer


def _book() -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(51)
    keys = np.array(["001", "002"])[rng.integers(0, 2, 100)]
    exposure = rng.choice([0.5, 2.0], size=100)
    counts = rng.poisson(exposure * np.where(keys == "001", 1.5, 0.9))
    book = pl.DataFrame({"VehicleCode": keys, "Exposure": exposure, "Claims": counts})
    catalogue = pl.DataFrame(
        {"VehicleCode": ["001", "002", "003"], "Power": [80, 120, 180]}
    )
    return book, catalogue


def test_standalone_catalogue_level_roundtrip_and_amendment() -> None:
    book, catalogue = _book()
    project = Project()
    config = ModelConfig(
        target="Claims",
        weight="Exposure",
        divide_target_by_weight=True,
        components=[
            ComponentRecipe(
                "vehicle",
                "Vehicle",
                "catalogue",
                ("Power",),
                {
                    "key": "VehicleCode",
                    "catalogue": catalogue.to_dict(as_series=False),
                    "iterations": 5,
                    "depth": 2,
                    "tune": False,
                },
            )
        ],
    )
    project.models["Vehicle"] = config
    run = fit_stack(project, book, config)
    assert [item.name for item in run.stack.components] == ["Vehicle"]
    lookup = run.stack.component("vehicle").scorer.table
    assert lookup.height == catalogue.height
    assert lookup.filter(pl.col("VehicleCode") == "003")["train_rows"].item() == 0
    np.testing.assert_allclose(
        run.predict(book), run.stack.component_multiplier("vehicle", book)
    )
    np.testing.assert_allclose(
        run.predict(book), StackScorer.from_dict(run.stack.to_dict()).predict_unit(book)
    )
    amended = amend_stack(project, book, run, "vehicle", "cap", lower=1.0, upper=1.3)
    assert np.max(amended.predict(book)) <= 1.3 + 1e-12
    np.testing.assert_allclose(run.predict(book), run.stack.predict_unit(book))


def test_catalogue_then_glm_scores_pinned_attribute_and_exposure_once() -> None:
    book, catalogue = _book()
    project = Project()
    project.design.variables["Power"] = VariableDesign(kind="continuous")
    config = ModelConfig(
        target="Claims",
        weight="Exposure",
        divide_target_by_weight=True,
        components=[
            ComponentRecipe(
                "vehicle",
                "Vehicle",
                "catalogue",
                ("Power",),
                {
                    "key": "VehicleCode",
                    "catalogue": catalogue.to_dict(as_series=False),
                    "iterations": 5,
                    "depth": 2,
                    "tune": False,
                },
            ),
            ComponentRecipe(
                "power_again", "Power again", "glm", ("Power",), {"alpha": 0.01}
            ),
        ],
    )
    project.models["Mixed"] = config
    run = fit_stack(project, book, config)
    unit = run.predict(book)
    assert np.all(np.isfinite(unit) & (unit > 0))
    np.testing.assert_allclose(
        run.stack.predict(book, exposure_col="Exposure"),
        unit * book["Exposure"].to_numpy(),
    )
    np.testing.assert_allclose(
        StackScorer.from_dict(run.stack.to_dict()).predict_unit(book), unit
    )


def test_catalogue_rejects_conflicting_history_and_lookup_key_type() -> None:
    book, catalogue = _book()
    project = Project()
    config = ModelConfig(
        target="Claims",
        weight="Exposure",
        divide_target_by_weight=True,
        components=[
            ComponentRecipe(
                "vehicle",
                "Vehicle",
                "catalogue",
                ("Power",),
                {
                    "key": "VehicleCode",
                    "catalogue": catalogue.to_dict(as_series=False),
                    "iterations": 5,
                    "tune": False,
                },
            )
        ],
    )
    project.models["Vehicle"] = config
    wrong = book.with_columns(pl.lit(999).alias("Power"))
    with pytest.raises(ValueError, match="conflicts"):
        fit_stack(project, wrong, config)
    lookup = LookupScorer(
        "VehicleCode", catalogue.with_columns(pl.lit(1.2).alias("multiplier"))
    )
    with pytest.raises(ValueError, match="different type"):
        lookup.linear_predictor(pl.DataFrame({"VehicleCode": [1]}))


def test_lookup_explicit_unknown_fallback() -> None:
    table = pl.DataFrame({"Code": ["001", "002"], "multiplier": [1.2, 0.8]})
    scorer = LookupScorer("Code", table, fallback=1.1)
    np.testing.assert_allclose(
        np.exp(scorer.linear_predictor(pl.DataFrame({"Code": ["001", "999"]}))),
        [1.2, 1.1],
    )
    assert scorer.coverage(pl.DataFrame({"Code": ["001", "999"]})) == {
        "rows": 2,
        "missing_key_rows": 1,
        "fallback_rows": 1,
        "unknown_examples": ["999"],
    }


def test_gaussian_stack_requires_explicit_log_link() -> None:
    book, _ = _book()
    project = Project()
    config = ModelConfig(
        family="gaussian",
        target="Claims",
        components=[
            ComponentRecipe("g", "Gaussian", "glm", ("Exposure",), {"alpha": 0.01})
        ],
    )
    project.models["Gaussian"] = config
    with pytest.raises(ValueError, match="log link"):
        fit_stack(project, book, config)


def test_legacy_pair_stages_remain_individually_inspectable() -> None:
    book, _ = _book()
    frame = book.with_columns(
        pl.Series("Region", np.where(np.arange(book.height) % 2, "north", "south")),
        pl.Series("traintest", np.ones(book.height, dtype=np.int64)),
    )
    project = Project()
    project.data.roles = {
        "VehicleCode": "predictor",
        "Region": "predictor",
        "Claims": "target",
        "Exposure": "weight",
        "traintest": "split",
    }
    config = project.new_model("Main", predictors=["VehicleCode"])
    config.target = "Claims"
    config.weight = "Exposure"
    config.divide_target_by_weight = True
    config.penalty.alpha = 0.01
    run = run_model(project, frame, "Main")
    first = VariableConfig(
        "categorical",
        [
            FromToRow("001", "001", 1),
            FromToRow("002", "002", 1),
            FromToRow(None, None, 1),
        ],
    )
    second = VariableConfig(
        "categorical",
        [
            FromToRow("north", "north", 1),
            FromToRow("south", "south", 1),
            FromToRow(None, None, 1),
        ],
    )
    pair = PairTableConfig(
        "pair-1",
        ("VehicleCode", "Region"),
        (first, second),
        [PairCellRow(0, 0, 1.2, 1, 1, 1)],
    )
    run.rate_model.add_pair_table(pair)
    run.config = copy.deepcopy(run.config)
    run.config.pair_stages = [PairStageConfig("pair-1", "VehicleCode", "Region")]
    project.models["First interaction"] = run.config
    run.name = "First interaction"
    stack = stack_from_legacy(project, run)
    assert [item.kind for item in stack.virtual_components] == ["glm", "interaction"]
    assert stack.component("Main").kind == "glm"
    assert stack.component("First interaction").kind == "interaction"
    np.testing.assert_allclose(stack.predict_unit(frame), run.predict(frame))
    np.testing.assert_allclose(
        StackScorer.from_dict(stack.to_dict()).predict_unit(frame), run.predict(frame)
    )


def test_lookup_persistence_preserves_typed_key_and_rejects_bad_values() -> None:
    table = pl.DataFrame(
        {
            "Code": pl.Series([1, 2], dtype=pl.Int32),
            "multiplier": [1.2, 0.8],
        }
    )
    saved = LookupScorer("Code", table).to_dict()
    loaded = LookupScorer.from_dict(saved)
    assert loaded.table.schema["Code"] == pl.Int32
    np.testing.assert_allclose(
        np.exp(
            loaded.linear_predictor(
                pl.DataFrame({"Code": pl.Series([1], dtype=pl.Int32)})
            )
        ),
        [1.2],
    )
    saved["table"]["multiplier"][0] = 0.0
    with pytest.raises(ValueError, match="nonpositive"):
        LookupScorer.from_dict(saved)
    loaded_master = load_catalogue(
        {
            "catalogue": table.drop("multiplier").to_dict(as_series=False),
            "catalogue_schema": {"Code": "Int32"},
        }
    )
    assert loaded_master.schema["Code"] == pl.Int32
