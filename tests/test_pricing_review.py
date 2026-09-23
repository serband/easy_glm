from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from easy_glm.engine.models import (
    FromToRow,
    PairCellRow,
    PairTableConfig,
    VariableConfig,
)
from easy_glm.pricing import PricingSession
from easy_glm.pricing.model import PricingModel
from easy_glm.workflow.pair_stages import PairStageArtifact, fit_pair_stages
from easy_glm.workflow.project import PairStageConfig, Project, VariableDesign
from easy_glm.workflow.run import rate_model_for


def _data() -> pl.DataFrame:
    rows = 80
    return pl.DataFrame(
        {
            "claims": [index % 3 == 0 for index in range(rows)],
            "exposure": np.linspace(0.5, 1.0, rows),
            "group": ["A", "B"] * (rows // 2),
            "region": ["N", "N", "S", "S"] * (rows // 4),
            "vehicle": ["new", "old", "old", "new"] * (rows // 4),
            "split": [1] * 64 + [0] * 16,
        }
    ).with_columns(pl.col("claims").cast(pl.Int64))


def _main_model() -> PricingModel:
    session = PricingSession(
        _data(),
        claims="claims",
        exposure="exposure",
        split="split",
    )
    session.categories("group", levels=["A", "B"])
    session.categories("region", levels=["N", "S"])
    session.categories("vehicle", levels=["new", "old"])
    return session.fit_glm("Main", factors=["group"], cv=2, n_alphas=3)


def _axis(levels: list[str]) -> VariableConfig:
    return VariableConfig(
        type="categorical",
        table=[
            *(FromToRow(level, level, 1.0) for level in levels),
            FromToRow(None, None, 1.0),
        ],
    )


def _table(stage_id: str, parents: tuple[str, str], value: float) -> PairTableConfig:
    levels = {
        "group": ["A", "B"],
        "region": ["N", "S"],
        "vehicle": ["new", "old"],
    }
    return PairTableConfig(
        stage_id=stage_id,
        parents=parents,
        axes=(_axis(levels[parents[0]]), _axis(levels[parents[1]])),
        cells=[PairCellRow(0, 0, value, row_count=10, fitting_weight=8.0)],
        provenance={"source": "test"},
    )


def _with_pairs(main: PricingModel) -> PricingModel:
    project = Project.from_dict(main._project.to_dict())
    config = project.models[main.name]
    config.pair_method = "sequential_catboost"
    config.pair_stages = [
        PairStageConfig("driver_region", "group", "region"),
        PairStageConfig("driver_vehicle", "group", "vehicle"),
    ]
    for variable, levels in {
        "group": ["A", "B"],
        "region": ["N", "S"],
        "vehicle": ["new", "old"],
    }.items():
        project.design.variables[variable] = VariableDesign(
            kind="categorical", levels=levels
        )
    run = copy.copy(main._run)
    run.config = config
    run.project_snapshot = project.to_dict()
    run.rate_model = main._run.rate_model.clone()
    tables = [
        _table("driver_region", ("group", "region"), 0.8),
        _table("driver_vehicle", ("group", "vehicle"), 1.1),
    ]
    for table in tables:
        run.rate_model.add_pair_table(copy.deepcopy(table))
    run.pair_stages = [
        PairStageArtifact(
            stage_id=stage.stage_id,
            parents=(stage.a, stage.b),
            table=copy.deepcopy(table),
            chosen_candidate=None,
            prefix_fingerprint=f"prefix-{index}",
        )
        for index, (stage, table) in enumerate(
            zip(config.pair_stages, tables, strict=True)
        )
    ]
    return PricingModel(main._session, project, run)


def test_main_rate_preview_and_apply_are_detached_and_do_not_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _main_model()
    original = model.predict(model._frame("train"))
    review = model.edit_rates("Reviewed").set_relativity("group", level="B", value=1.25)
    preview = review.preview()
    assert preview.changes.to_dicts() == [
        {"table": "group", "row": "B", "before": pytest.approx(1.0), "proposed": 1.25}
    ]
    assert preview.summary["expected"][1] != preview.summary["expected"][0]

    def unexpected_fit(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("keeping existing tables must not fit")

    monkeypatch.setattr("easy_glm.pricing.review.run_model", unexpected_fit)
    adjusted = review.apply(refit_later_interactions=False)
    np.testing.assert_allclose(model.predict(model._frame("train")), original)
    assert not np.allclose(adjusted.predict(adjusted._frame("train")), original)
    assert adjusted._run.config is adjusted._project.models[adjusted.name]
    assert adjusted.history[-1]["action"] == "rate_review"


def test_loaded_checkpoint_can_preview_and_apply_without_a_fit(tmp_path: Path) -> None:
    original = _main_model()
    path = original.save(tmp_path / "main.easyglm.json")
    loaded = PricingModel.load(path, data=_data())
    assert loaded._run.fit is None
    loaded._run.metrics["train"]["mean_deviance"] = -123.0

    review = loaded.edit_rates("Loaded review").set_relativity(
        "group", level="B", value=1.18
    )
    preview = review.preview()
    assert preview.changes["proposed"].to_list() == [1.18]
    adjusted = review.apply(refit_later_interactions=False)

    assert loaded._run.rate_model.variables["group"].table[1].relativity != 1.18
    assert adjusted._run.rate_model.variables["group"].table[1].relativity == 1.18
    assert adjusted._run.fit is None
    assert adjusted._run.metrics["train"]["mean_deviance"] != -123.0
    assert adjusted._run.metrics["train"]["expected"] == pytest.approx(
        adjusted._run.predict(adjusted._frame("train")).dot(
            adjusted._frame("train")["exposure"].to_numpy()
        )
    )


def test_rate_edit_moves_prior_validation_out_of_current_evidence(
    tmp_path: Path,
) -> None:
    model = _main_model()
    model.validate_holdout()
    prior = model.evidence["validation"]

    adjusted = (
        model.edit_rates("After validation")
        .set_relativity("group", level="B", value=1.23)
        .apply(refit_later_interactions=False)
    )

    assert "validation" not in adjusted.evidence
    historical = adjusted.evidence["historical_validation"][-1]
    assert historical["status"] == "superseded_by_rate_review"
    assert historical["source_model"] == model.name
    assert historical["validation"] == prior
    saved = json.loads(adjusted.save(tmp_path / "adjusted.easyglm.json").read_text())
    assert "validation" not in saved["evidence"]
    assert saved["evidence"]["historical_validation"][-1]["validation"] == prior


def test_loaded_first_pair_keeps_exact_saved_mains_and_base(tmp_path: Path) -> None:
    model = _main_model()
    model._run.rate_model.base_rate *= 1.07
    group = model._run.rate_model.variables["group"]
    b_row = next(row for row in group.table if row.from_ == "B")
    model._run.rate_model.update_relativity(
        "group", b_row.from_, b_row.to_, b_row.relativity * 1.11
    )
    saved_base = model._run.rate_model.base_rate
    saved_rows = [row.relativity for row in group.table]
    loaded = PricingModel.load(
        model.save(tmp_path / "literal-main.easyglm.json"), data=_data()
    )

    with_pair = loaded.fit_interaction(
        "group", "region", name="Loaded pair", trials=1, prefix_trials=1
    )

    assert with_pair._run.rate_model.base_rate == pytest.approx(saved_base)
    assert [
        row.relativity for row in with_pair._run.rate_model.variables["group"].table
    ] == pytest.approx(saved_rows)
    assert with_pair._run.pair_stages[0].reused is False


def test_rebalance_is_explicit_and_restores_original_training_expected() -> None:
    model = _main_model()
    review = model.edit_rates("Balanced").set_relativity("group", level="B", value=1.4)
    changed = review.preview().summary
    assert changed["expected"][1] != pytest.approx(changed["expected"][0])
    review.rebalance()
    balanced = review.preview().summary
    assert balanced["expected"][1] == pytest.approx(balanced["expected"][0], rel=1e-12)
    assert any(
        row["table"] == "base rate" for row in review.preview().changes.to_dicts()
    )


def test_rebalance_rejects_a_non_multiplicative_link() -> None:
    session = PricingSession(
        _data(),
        family="binomial",
        target="claims",
        split="split",
    )
    session.categories("group", levels=["A", "B"])
    model = session.fit_glm("Gaussian", factors=["group"], alpha=0.01)
    assert model._run.rate_model.metadata.link == "logit"
    review = model.edit_rates("Not proportional").set_relativity(
        "group", level="B", value=1.2
    )
    with pytest.raises(ValueError, match="multiplicative log-link"):
        review.rebalance()


def test_keep_later_tables_marks_historical_cv_and_preserves_pair_edit() -> None:
    model = _with_pairs(_main_model())
    original_first = model._run.rate_model.get_pair_table(
        "driver_region"
    ).cell_matrix.copy()
    review = model.edit_rates("Kept tables").set_pair_relativity(
        "region",
        "group",
        level_a="N",
        level_b="A",
        value=0.92,
    )
    adjusted = review.apply(refit_later_interactions=False)
    assert (
        model._run.rate_model.get_pair_table("driver_region").cell_matrix[0, 0] == 0.8
    )
    assert (
        adjusted._run.rate_model.get_pair_table("driver_region").cell_matrix[0, 0]
        == 0.92
    )
    assert adjusted._run.pair_stages[0].table.cell_matrix[0, 0] == 0.92
    np.testing.assert_allclose(
        adjusted._run.rate_model.get_pair_table("driver_region").cell_matrix[1:, :],
        original_first[1:, :],
    )
    assert adjusted._frozen_stages == ["driver_region", "driver_vehicle"]
    assert all(
        artifact.status == "pricing_adjustment"
        for artifact in adjusted._run.pair_stages
    )
    assert adjusted.evidence["rate_review"]["cv_evidence"] == {
        "driver_region": "historical_pre_edit",
        "driver_vehicle": "historical_pre_edit",
    }


def test_refit_later_stages_passes_edited_full_prefix_and_replaces_only_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _with_pairs(_main_model())
    old_second = model._run.rate_model.get_pair_table(
        "driver_vehicle"
    ).cell_matrix.copy()
    review = model.edit_rates("Refitted suffix").set_pair_relativity(
        "group", "region", level_a="A", level_b="N", value=0.93
    )
    calls: list[list[str]] = []

    def fake_run_model(
        project: Project,
        frame: pl.DataFrame,
        name: str,
        **kwargs: Any,
    ):
        frozen = kwargs["frozen_pair_prefix"]
        frozen_scorer = kwargs["frozen_pair_rate_model"]
        calls.append([artifact.stage_id for artifact in frozen])
        assert frame.height == model._frame("train").height
        assert frozen[0].table.cells[0].relativity == 0.8
        assert frozen_scorer.get_pair_table("driver_region").cell_matrix[
            0, 0
        ] == pytest.approx(0.93)
        run = copy.copy(model._run)
        run.name = name
        run.config = project.models[name]
        run.project_snapshot = project.to_dict()
        run.rate_model = rate_model_for(
            model._project,
            model._run,
            run.config.adjustments,
            base_rate_override=run.config.base_rate_override,
        )
        run.rate_model.update_pair_cell("driver_vehicle", 0, 0, 1.35)
        run.pair_stages = copy.deepcopy(model._run.pair_stages)
        run.pair_stages[1].table.cells[0].relativity = 1.35
        return run

    monkeypatch.setattr("easy_glm.pricing.review.run_model", fake_run_model)
    adjusted = review.apply(refit_later_interactions=True)
    assert calls == [["driver_region"]]
    first = adjusted._run.rate_model.get_pair_table("driver_region").cell_matrix
    assert first[0, 0] == 0.93
    np.testing.assert_allclose(
        first[1:, :], model._run.rate_model.pair_tables[0].cell_matrix[1:, :]
    )
    second = adjusted._run.rate_model.get_pair_table("driver_vehicle").cell_matrix
    assert second[0, 0] == 1.35
    assert not np.array_equal(second, old_second)
    assert adjusted._frozen_stages == ["driver_region"]
    assert adjusted.history[-1]["proposed_before_refit"] is not None
    assert adjusted.history[-1]["after"]["expected"] == pytest.approx(
        adjusted._run.rate_model.predict(
            adjusted._frame("train"), exposure_col=None
        ).dot(adjusted._frame("train")["exposure"].to_numpy())
    )
    np.testing.assert_allclose(
        adjusted.predict(adjusted._frame("train")),
        adjusted._run.rate_model.predict(adjusted._frame("train"), exposure_col=None),
    )


def test_refit_rejects_pair_edit_with_data_derived_levels() -> None:
    model = _with_pairs(_main_model())
    model._project.design.variables["region"].levels = None
    review = model.edit_rates("Ambiguous").set_pair_relativity(
        "group", "region", level_a="A", level_b="N", value=0.91
    )
    with pytest.raises(ValueError, match="data-derived levels"):
        review.apply(refit_later_interactions=True)


def test_fit_pair_stages_can_keep_a_reviewed_full_prefix_without_teacher_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _with_pairs(_main_model())
    project = model._project
    config = project.models[model.name]
    config.adjustments = [
        copy.deepcopy(
            model.edit_rates("x")
            .set_pair_relativity(
                "group", "region", level_a="A", level_b="N", value=0.94
            )
            ._config.adjustments[-1]
        )
    ]
    main = model._run.rate_model.clone()
    main.pair_tables = []

    def no_teacher(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a fully frozen prefix must not fit a teacher")

    monkeypatch.setattr("easy_glm.workflow.pair_stages._fit_table", no_teacher)
    scorer, artifacts = fit_pair_stages(
        project,
        model._frame("train"),
        config,
        main,
        frozen_prefix_artifacts=model._run.pair_stages,
        frozen_prefix_rate_model=rate_model_for(
            project, model._run, config.adjustments
        ),
    )
    assert scorer.get_pair_table("driver_region").cell_matrix[0, 0] == 0.94
    assert [artifact.status for artifact in artifacts] == [
        "pricing_adjustment",
        "up_to_date",
    ]
    assert artifacts[0].table.provenance["cv_evidence"] == "historical_pre_edit"
    assert "cv_evidence" not in artifacts[1].table.provenance


def test_real_suffix_refit_keeps_reviewed_first_table_and_refits_second(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(91)
    rows = 240
    x = rng.normal(size=rows)
    z = rng.normal(size=rows)
    w = rng.normal(size=rows)
    exposure = rng.uniform(0.4, 1.0, size=rows)
    mean = exposure * np.exp(-1.0 + 0.2 * x + 0.18 * x * z + 0.12 * x * w)
    data = pl.DataFrame(
        {
            "claims": rng.poisson(mean),
            "exposure": exposure,
            "x": x,
            "z": z,
            "w": w,
        }
    )
    session = PricingSession(data, claims="claims", exposure="exposure", seed=9)
    for variable in ("x", "z", "w"):
        session.bands(variable, cuts=[-0.75, 0.0, 0.75])
    main = session.fit_glm("Main", factors=["x"], alpha=0.01)
    first = main.fit_interaction("x", "z", name="First", trials=1, prefix_trials=1)
    second = first.fit_interaction("x", "w", name="Second", trials=1, prefix_trials=1)
    assert second._run.pair_stages[0].reused is True
    assert second._run.pair_stages[0].status != "pricing_adjustment"
    first_table = second._run.rate_model.pair_tables[0]
    axis_a_row = 0
    axis_b_row = 0
    axis_a = first_table.axes[0].table[axis_a_row]
    axis_b = first_table.axes[1].table[axis_b_row]
    original_value = float(first_table.cell_matrix[axis_a_row, axis_b_row])
    reviewed_value = original_value * 1.08
    second.validate_holdout()
    prior_validation = second.evidence["validation"]

    adjusted = (
        second.edit_rates("Reviewed suffix")
        .set_pair_relativity(
            "x",
            "z",
            lower_a=axis_a.from_,
            upper_a=axis_a.to_,
            lower_b=axis_b.from_,
            upper_b=axis_b.to_,
            value=reviewed_value,
        )
        .apply(refit_later_interactions=True)
    )

    assert adjusted._run.pair_stages[0].reused is True
    assert adjusted._run.pair_stages[1].reused is False
    assert "validation" not in adjusted.evidence
    assert (
        adjusted.evidence["historical_validation"][-1]["validation"] == prior_validation
    )
    assert adjusted._run.pair_stages[0].status == "pricing_adjustment"
    assert adjusted._run.pair_stages[0].table.cell_matrix[
        axis_a_row, axis_b_row
    ] == pytest.approx(reviewed_value)
    assert adjusted._run.rate_model.pair_tables[0].cell_matrix[
        axis_a_row, axis_b_row
    ] == pytest.approx(reviewed_value)
    assert second._run.rate_model.pair_tables[0].cell_matrix[
        axis_a_row, axis_b_row
    ] == pytest.approx(original_value)
    assert adjusted.history[-1]["after"]["expected"] == pytest.approx(
        adjusted._run.predict(adjusted._frame("train")).dot(
            adjusted._frame("train")["exposure"].to_numpy()
        )
    )

    loaded = PricingModel.load(second.save(tmp_path / "pairs.easyglm.json"), data=data)
    assert loaded._run.fit is None
    loaded_adjusted = (
        loaded.edit_rates("Loaded reviewed suffix")
        .set_pair_relativity(
            "x",
            "z",
            lower_a=axis_a.from_,
            upper_a=axis_a.to_,
            lower_b=axis_b.from_,
            upper_b=axis_b.to_,
            value=reviewed_value,
        )
        .apply(refit_later_interactions=True)
    )
    assert loaded_adjusted._run.pair_stages[0].reused is True
    assert loaded_adjusted._run.pair_stages[1].reused is False
    assert loaded_adjusted._run.rate_model.pair_tables[0].cell_matrix[
        axis_a_row, axis_b_row
    ] == pytest.approx(reviewed_value)
