"""Accepted stack scoring, catalogue ownership, and persistence contracts."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import polars as pl
import pytest

from easy_glm.engine.models import (
    FromToRow,
    PairCellRow,
    PairTableConfig,
    VariableConfig,
)
from easy_glm.workflow.catalogue import join_catalogue, load_catalogue
from easy_glm.workflow.project import (
    ModelConfig,
    PairStageConfig,
    Project,
    VariableDesign,
)
from easy_glm.workflow.run import run_model
from easy_glm.workflow.stack import (
    _cat_features_for_recipe,
    _recipe_categorical,
    amend_stack,
    fit_stack,
    raw_features,
    stack_from_legacy,
)
from easy_glm.workflow.stack_types import (
    ComponentArtifact,
    ComponentRecipe,
    ConstantScorer,
    LookupScorer,
    StackScorer,
)


def _local_design(*, n_bins: int, kind: str = "step") -> dict[str, object]:
    return {
        "defaults": {
            "n_bins": n_bins,
            "min_level_share": 0.0025,
            "null_indicator": True,
            "max_integer_knots": 150,
        },
        "variables": {
            "x": {
                "kind": kind,
                "knots": "quantile",
                "clamp": None,
                "n_bins": n_bins,
                "null_indicator": True,
                "min_level_share": 0.0025,
                "max_levels": None,
                "levels": None,
                "monotone": None,
                "penalty_weight": 1.0,
            }
        },
    }


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


def test_csv_numeric_catalogue_key_aligns_without_losing_leading_zero_keys() -> None:
    experience = pl.DataFrame({"PolicyId": [1, 2]})
    numeric_csv = pl.DataFrame({"PolicyId": ["1", "2"], "Group": ["A", "B"]})
    joined = join_catalogue(experience, numeric_csv, "PolicyId", ("Group",))
    assert joined.schema["PolicyId"] == pl.Int64
    assert joined["Group"].to_list() == ["A", "B"]

    leading_zero_csv = pl.DataFrame({"PolicyId": ["001", "002"], "Group": ["A", "B"]})
    with pytest.raises(ValueError, match="different type"):
        join_catalogue(experience, leading_zero_csv, "PolicyId", ("Group",))


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


def test_component_design_is_local_and_legacy_serialization_stays_unchanged() -> None:
    legacy = ComponentRecipe("legacy", "Legacy", "glm", ("x",), {"alpha": 0.01})
    assert "design" not in legacy.to_dict()
    project = Project()
    project.models["Legacy"] = ModelConfig(components=[legacy])
    assert "design" not in project.to_dict()["models"]["Legacy"]["components"][0]
    stack = StackScorer("poisson", "log", "Claims", "Exposure", None, True, None)
    stack.components.append(
        ComponentArtifact(legacy, "", ConstantScorer(0.0), {"legacy": True})
    )
    expected = {
        "format_version": 1,
        "family": "poisson",
        "link": "log",
        "target": "Claims",
        "weight": "Exposure",
        "offset_col": None,
        "divide_target_by_weight": True,
        "tweedie_power": None,
        "components": [
            {
                "recipe": {
                    "component_id": "legacy",
                    "name": "Legacy",
                    "kind": "glm",
                    "factors": ("x",),
                    "settings": {"alpha": 0.01},
                    "amendments": (),
                },
                "prefix_fingerprint": "",
                "scorer": {"type": "constant", "log_value": 0.0},
                "evidence": {"legacy": True},
                "status": "current",
            }
        ],
        "virtual_components": [],
    }
    assert stack.to_dict() == expected
    assert (
        stack.fingerprint()
        == hashlib.sha256(
            json.dumps(expected, sort_keys=True, default=str).encode()
        ).hexdigest()
    )
    assert StackScorer.from_dict(expected).to_dict() == expected

    recipe = ComponentRecipe(
        "local", "Local", "glm", ("x",), {"alpha": 0.01}, design=_local_design(n_bins=3)
    )
    restored = ComponentRecipe.from_dict(recipe.to_dict())
    assert restored.design == recipe.design
    with pytest.raises(ValueError, match="selected factors"):
        ComponentRecipe(
            "bad",
            "Bad",
            "glm",
            ("x",),
            {"alpha": 0.01},
            design={
                **_local_design(n_bins=3),
                "variables": {"other": _local_design(n_bins=3)["variables"]["x"]},
            },
        )
    bad = recipe.to_dict()
    bad["design"]["defaults"]["unexpected"] = True
    with pytest.raises(ValueError, match="defaults"):
        ComponentRecipe.from_dict(bad)
    with pytest.raises(ValueError, match="duplicates"):
        ComponentRecipe("duplicate", "Duplicate", "glm", ("x", "x"))


def test_catboost_uses_local_type_override_or_raw_type_inference() -> None:
    numeric = pl.DataFrame({"x": [1, 2]})
    categorical_recipe = ComponentRecipe(
        "cat",
        "Cat",
        "catboost",
        ("x",),
        design=_local_design(n_bins=3, kind="categorical"),
    )
    categorical = _recipe_categorical(categorical_recipe, numeric)
    raw, indices = raw_features(numeric, categorical_recipe.factors, categorical)
    assert indices == (0,)
    assert raw[:, 0].tolist() == ["1", "2"]

    strings = pl.DataFrame({"x": ["a", "b"]})
    inferred_recipe = ComponentRecipe(
        "inferred",
        "Inferred",
        "catboost",
        ("x",),
        design=_local_design(n_bins=3, kind=None),
    )
    assert _recipe_categorical(inferred_recipe, strings) == (0,)

    # A saved legacy index cannot silently overrule the visible local choice.
    numeric_recipe = ComponentRecipe(
        "numeric",
        "Numeric",
        "catboost",
        ("x",),
        {"cat_features": [0]},
        design=_local_design(n_bins=3, kind="step"),
    )
    assert _cat_features_for_recipe(numeric_recipe, numeric) == ()
    legacy_recipe = ComponentRecipe(
        "legacy", "Legacy", "catboost", ("x",), {"cat_features": [0]}
    )
    assert _cat_features_for_recipe(legacy_recipe, numeric) == (0,)


def test_cumulative_glms_keep_their_own_numeric_bands() -> None:
    rng = np.random.default_rng(19)
    frame = pl.DataFrame(
        {
            "x": np.arange(180, dtype=float),
            "Exposure": np.ones(180),
            "Claims": rng.poisson(np.exp(-1.5 + np.arange(180) / 400)),
        }
    )
    project = Project()
    project.design.defaults.n_bins = 9  # Neither component may inherit this.
    config = ModelConfig(
        target="Claims",
        weight="Exposure",
        divide_target_by_weight=True,
        family="poisson",
        link="log",
        components=[
            ComponentRecipe(
                "first",
                "First",
                "glm",
                ("x",),
                {"alpha": 0.01},
                design=_local_design(n_bins=2),
            ),
            ComponentRecipe(
                "second",
                "Second",
                "glm",
                ("x",),
                {"alpha": 0.01},
                design=_local_design(n_bins=5),
            ),
        ],
    )
    project.models["Stack"] = config
    run = fit_stack(project, frame, config, tuning=False)
    first = run.stack.component("first").scorer.rate_model.variables["x"].table
    second = run.stack.component("second").scorer.rate_model.variables["x"].table
    assert (
        len([row for row in first if not (row.from_ is None and row.to_ is None)]) == 2
    )
    assert (
        len([row for row in second if not (row.from_ is None and row.to_ is None)]) == 5
    )
    assert project.design.defaults.n_bins == 9


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
