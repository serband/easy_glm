"""Public frozen-score contracts for fresh data books."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from easy_glm.engine.models import FromToRow, ModelMetadata, VariableConfig
from easy_glm.engine.rate_model import RateModel
from easy_glm.workflow.project import ModelConfig, Project
from easy_glm.workflow.scoring import score_data, validate_scoring_data
from easy_glm.workflow.stack import fit_stack
from easy_glm.workflow.stack_types import (
    ComponentArtifact,
    ComponentRecipe,
    ConstantScorer,
    LookupScorer,
    RateModelScorer,
    StackScorer,
)


def test_rate_model_score_preserves_new_book_and_export(tmp_path) -> None:
    scorer = RateModel(
        2.0,
        {"Band": VariableConfig("categorical", [FromToRow("A", "A", 1.5)])},
    )
    book = pl.DataFrame({"Band": ["A", "new"], "keep": [3, 4]})
    result = score_data(scorer, book)
    assert result.frame.columns == ["Band", "keep", "easyglm_prediction"]
    np.testing.assert_allclose(result.frame["easyglm_prediction"], [3.0, 2.0])
    assert len(result.metadata["scorer_fingerprint"]) == 64
    assert len(result.metadata["dataset_digest"]) == 64
    assert result.metadata["preparation_fingerprint"] is None
    saved = result.export(tmp_path / "new-book.parquet")
    np.testing.assert_allclose(pl.read_parquet(saved)["easyglm_prediction"], [3.0, 2.0])
    assert saved.with_suffix(".parquet.metadata.json").exists()
    csv = result.export(tmp_path / "new-book.csv")
    np.testing.assert_allclose(pl.read_csv(csv)["easyglm_prediction"], [3.0, 2.0])
    assert csv.with_suffix(".csv.metadata.json").exists()


def test_stack_selected_contribution_needs_only_its_lookup_dependencies() -> None:
    first = LookupScorer(
        "Vehicle",
        pl.DataFrame({"Vehicle": ["v1"], "Power": [80], "multiplier": [1.2]}),
        ("Power",),
    )
    second = LookupScorer(
        "Postcode",
        pl.DataFrame({"Postcode": ["p1"], "Area": ["N"], "multiplier": [0.9]}),
        ("Area",),
    )
    stack = StackScorer("poisson", "log", "Claims", None, None, False, None)
    stack.components = [
        ComponentArtifact(
            ComponentRecipe("vehicle", "Vehicle", "catalogue", ("Power",), {}),
            "",
            first,
        ),
        ComponentArtifact(
            ComponentRecipe("area", "Area", "catalogue", ("Area",), {}),
            "",
            second,
        ),
    ]
    # The second lookup does not depend on Vehicle/Power, so its contribution
    # should not demand an unrelated earlier catalogue key.
    result = score_data(
        stack,
        pl.DataFrame({"Postcode": ["p1"]}),
        scope="model",
        component_ids=("area",),
    )
    np.testing.assert_allclose(result.frame["easyglm_multiplier"], [0.9])


def test_catalogue_native_teacher_roundtrips_and_deployed_lookup_stays_amended() -> (
    None
):
    book = pl.DataFrame(
        {"Vehicle": ["a", "b", "a", "b"], "Exposure": [1.0] * 4, "Claims": [3, 1, 2, 1]}
    )
    master = pl.DataFrame({"Vehicle": ["a", "b", "c"], "Power": [70, 120, 180]})
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
                    "key": "Vehicle",
                    "catalogue": master.to_dict(as_series=False),
                    "iterations": 4,
                    "depth": 2,
                    "tune": False,
                },
            )
        ],
    )
    project.models["Vehicle"] = config
    stack = fit_stack(project, book, config).stack
    lookup = stack.component("vehicle").scorer
    assert lookup.native_scorer is not None
    restored = StackScorer.from_dict(stack.to_dict())
    native = score_data(restored, master, scope="catalogue", component_ids=("vehicle",))
    assert native.frame.height == master.height
    lookup.table = lookup.table.with_columns(pl.lit(1.0).alias("multiplier"))
    accepted = score_data(stack, book.select("Vehicle"))
    np.testing.assert_allclose(accepted.frame["easyglm_prediction"], 1.0)
    old = stack.to_dict()
    del old["components"][0]["scorer"]["native_scorer"]
    issues = validate_scoring_data(
        StackScorer.from_dict(old),
        master,
        scope="catalogue",
        component_ids=("vehicle",),
    )["issues"]
    assert issues[0]["code"] == "catalogue_native_unavailable"


def test_preflight_reports_missing_columns_without_neutral_scoring() -> None:
    scorer = RateModel(1.0, {"x": VariableConfig("numeric", [FromToRow(0, 1, 1.0)])})
    issues = validate_scoring_data(scorer, pl.DataFrame({"other": [1]}))["issues"]
    assert issues and issues[0]["code"] == "columns"
    assert not validate_scoring_data(scorer, pl.DataFrame({"other": [1]}))["ready"]
    with pytest.raises(ValueError, match="Missing required columns"):
        score_data(scorer, pl.DataFrame({"other": [1]}))


def test_scoring_scope_does_not_evaluate_unrelated_current_premium() -> None:
    project = Project()
    project.data.roles = {"Premium": "current_premium"}
    scorer = RateModel(
        1.0,
        {"Vehicle": VariableConfig("categorical", [FromToRow("a", "a", 1.2)])},
    )
    result = score_data(scorer, pl.DataFrame({"Vehicle": ["a"]}), project=project)
    np.testing.assert_allclose(result.frame["easyglm_prediction"], [1.2])


def test_rate_model_preflight_includes_external_offset() -> None:
    scorer = RateModel(
        1.0,
        {},
        metadata=ModelMetadata(offset_col="existing_rate", offset_is_log=True),
    )
    issues = validate_scoring_data(scorer, pl.DataFrame({"x": [1]}))["issues"]
    assert issues[0]["code"] == "columns"


@pytest.mark.parametrize("scope", ["through", "model", "catalogue"])
def test_standalone_rate_model_rejects_unimplemented_selected_scopes(
    scope: str,
) -> None:
    scorer = RateModel(1.0, {})
    check = validate_scoring_data(
        scorer,
        pl.DataFrame({"Policy": [1]}),
        scope=scope,  # type: ignore[arg-type]
        component_ids=("not-a-component",),
        through="not-a-component" if scope == "through" else None,
    )
    assert not check["ready"]
    assert "Standalone RateModel" in check["issues"][0]["message"]
    with pytest.raises(ValueError, match="Standalone RateModel"):
        score_data(
            scorer,
            pl.DataFrame({"Policy": [1]}),
            scope=scope,  # type: ignore[arg-type]
            component_ids=("not-a-component",),
            through="not-a-component" if scope == "through" else None,
        )


def test_frozen_transforms_close_dependencies_without_filters_or_split() -> None:
    project = Project()
    project.data.derived = [
        type("Derived", (), {"name": "x1", "expr": "pl.col('raw') + 1"})(),
        type("Derived", (), {"name": "x2", "expr": "pl.col('x1') * 2"})(),
        type("Derived", (), {"name": "unrelated", "expr": "pl.col('missing') + 1"})(),
    ]
    project.data.filters = ["pl.col('raw') > 100"]
    project.data.split.column = "Split"
    scorer = RateModel(1.0, {"x2": VariableConfig("numeric", [FromToRow(0, 10, 1.5)])})
    result = score_data(scorer, pl.DataFrame({"raw": [1, 2]}), project=project)
    assert result.frame.height == 2
    np.testing.assert_allclose(result.frame["easyglm_prediction"], [1.5, 1.5])


def test_lookup_keeps_leading_zero_key_and_rejects_output_collision() -> None:
    lookup = LookupScorer(
        "Code",
        pl.DataFrame({"Code": ["001"], "multiplier": [1.4]}),
    )
    stack = StackScorer("poisson", "log", "Claims", None, None, False, None)
    stack.components = [
        ComponentArtifact(
            ComponentRecipe("codes", "Codes", "catalogue", (), {}), "", lookup
        )
    ]
    result = score_data(stack, pl.DataFrame({"Code": ["001"]}))
    np.testing.assert_allclose(result.frame["easyglm_prediction"], [1.4])
    with pytest.raises(ValueError, match="Output columns already exist"):
        score_data(
            stack,
            pl.DataFrame({"Code": ["001"], "easyglm_prediction": [0.0]}),
        )


def test_later_accepted_lookup_needs_its_own_key_not_a_reused_attribute_source() -> (
    None
):
    first = LookupScorer(
        "VehicleCode",
        pl.DataFrame({"VehicleCode": ["v1"], "Power": [80], "multiplier": [1.1]}),
        ("Power",),
    )
    later = LookupScorer(
        "GarageCode",
        pl.DataFrame({"GarageCode": ["g1"], "Power": [80], "multiplier": [0.9]}),
        ("Power",),
    )
    stack = StackScorer("poisson", "log", "Claims", None, None, False, None)
    stack.components = [
        ComponentArtifact(
            ComponentRecipe("vehicle", "Vehicle", "catalogue", ("Power",), {}),
            "",
            first,
        ),
        ComponentArtifact(
            ComponentRecipe("garage", "Garage", "catalogue", ("Power",), {}),
            "",
            later,
        ),
    ]
    result = score_data(
        stack,
        pl.DataFrame({"GarageCode": ["g1"]}),
        scope="model",
        component_ids=("garage",),
    )
    np.testing.assert_allclose(result.frame["easyglm_multiplier"], [0.9])


def test_lookup_fallback_is_reported_but_still_scores() -> None:
    lookup = LookupScorer(
        "Code", pl.DataFrame({"Code": ["known"], "multiplier": [1.2]}), fallback=0.8
    )
    stack = StackScorer("poisson", "log", "Claims", None, None, False, None)
    stack.components = [
        ComponentArtifact(
            ComponentRecipe("code", "Code", "catalogue", (), {}), "", lookup
        )
    ]
    result = score_data(stack, pl.DataFrame({"Code": ["missing"]}))
    np.testing.assert_allclose(result.frame["easyglm_prediction"], [0.8])
    assert result.warnings[0]["code"] == "lookup_fallback"
    assert result.metadata["lookup_coverage"][0]["fallback_rows"] == 1
    assert validate_scoring_data(stack, pl.DataFrame({"Code": ["missing"]}))["ready"]


def test_model_contribution_ignores_an_unused_bad_external_offset() -> None:
    stack = StackScorer("poisson", "log", "Claims", None, "offset", False, None)
    stack.components = [
        ComponentArtifact(
            ComponentRecipe("adjustment", "Adjustment", "catboost", (), {}),
            "",
            ConstantScorer(np.log(1.25)),
        )
    ]
    result = score_data(
        stack,
        pl.DataFrame({"offset": [float("nan")]}),
        scope="model",
        component_ids=("adjustment",),
    )
    np.testing.assert_allclose(result.frame["easyglm_multiplier"], [1.25])


def test_nonfinite_contribution_is_blocked_in_preflight() -> None:
    stack = StackScorer("poisson", "log", "Claims", None, None, False, None)
    stack.components = [
        ComponentArtifact(
            ComponentRecipe("overflow", "Overflow", "catboost", (), {}),
            "",
            ConstantScorer(1_000.0),
        )
    ]
    check = validate_scoring_data(
        stack,
        pl.DataFrame({"Policy": [1]}),
        scope="model",
        component_ids=("overflow",),
    )
    assert not check["ready"]
    assert any(issue["code"] == "scoring" for issue in check["issues"])


def test_full_stack_discovers_columns_inside_a_legacy_rate_model_scorer() -> None:
    legacy = RateModel(
        1.0,
        {"Region": VariableConfig("categorical", [FromToRow("N", "N", 1.1)])},
    )
    stack = StackScorer("poisson", "log", "Claims", None, None, False, None)
    stack.components = [
        ComponentArtifact(
            ComponentRecipe("legacy", "Accepted legacy scorer", "legacy", (), {}),
            "",
            RateModelScorer(legacy),
        )
    ]
    check = validate_scoring_data(stack, pl.DataFrame({"Policy": [1]}))
    assert check["requirements"]["columns"] == ["Region"]
    assert not check["ready"]


def test_score_accepts_pandas_and_logit_rate_model_without_training_data() -> None:
    pd = pytest.importorskip("pandas")
    scorer = RateModel(
        2.0,
        {},
        metadata=ModelMetadata(link="logit"),
    )
    result = score_data(scorer, pd.DataFrame({"Policy": [1, 2]}))
    # Logit tables store odds: the deployed response conversion happens here.
    expected = 2.0 / 3.0
    np.testing.assert_allclose(result.frame["easyglm_prediction"], [expected, expected])


@pytest.mark.parametrize("scope", ["model", "through", "catalogue"])
def test_internal_baseline_is_never_a_selectable_scoring_component(scope: str) -> None:
    stack = StackScorer("poisson", "log", "Claims", None, None, False, None)
    stack.components = [
        ComponentArtifact(
            ComponentRecipe("__baseline__", "Starting level", "baseline"),
            "",
            ConstantScorer(0.2),
        )
    ]
    check = validate_scoring_data(
        stack,
        pl.DataFrame({"Policy": [1]}),
        scope=scope,  # type: ignore[arg-type]
        component_ids=("__baseline__",),
        through="__baseline__" if scope == "through" else None,
    )
    assert any(issue["code"] == "component" for issue in check["issues"])
    with pytest.raises(ValueError, match="Internal baseline"):
        score_data(
            stack,
            pl.DataFrame({"Policy": [1]}),
            scope=scope,  # type: ignore[arg-type]
            component_ids=("__baseline__",),
            through="__baseline__" if scope == "through" else None,
        )
