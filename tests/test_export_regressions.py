"""Execute exported scripts and compare their scores and reporting basis."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from easy_glm import fit_glm, rate_tables, to_rate_model
from easy_glm.core.fit import TwoStageFit
from easy_glm.engine import RateModel
from easy_glm.workflow import (
    Adjustment,
    Interaction,
    ModelRun,
    Project,
    VariableDesign,
    build_design,
    prepare,
    run_model,
    to_script,
    totals,
    train_holdout,
)


def _project(
    tmp_path: Path,
    *,
    family: str = "poisson",
    divide: bool = True,
    weighted: bool = True,
) -> Project:
    rng = np.random.default_rng(318)
    n = 600
    x = rng.uniform(-1.0, 1.0, n)
    region = rng.choice(["A", "B", "C"], n)
    weight = rng.integers(2, 12, n).astype(float) if weighted else np.ones(n)
    offset = rng.uniform(-0.5, 0.5, n)
    eta = (
        0.6
        + 0.7 * (x > 0)
        - 0.4 * (region == "C")
        + 1.6 * ((x > 0.5) & (region == "B"))
        + offset
    )
    if family == "binomial":
        probability = 1 / (1 + np.exp(-eta))
        y = rng.binomial(weight.astype(int), probability) / weight
    else:
        count = rng.poisson(np.exp(eta) * weight).astype(float)
        y = count if divide else count / weight
    frame = pl.DataFrame(
        {
            "x": x,
            "region": region,
            "weight": weight,
            "exposure": rng.uniform(0.1, 0.9, n),
            "offset": offset,
            "target": y,
            "traintest": (np.arange(n) < 450).astype(int),
        }
    )
    path = tmp_path / "book.parquet"
    frame.write_parquet(path)
    project = Project(name="Export regression")
    project.data.source.type = "parquet"
    project.data.source.path = str(path)
    project.data.roles = {
        "x": "predictor",
        "region": "predictor",
        "target": "target",
        "offset": "offset",
        "weight": "weight" if weighted else "ignore",
        "exposure": "exposure" if family != "binomial" else "ignore",
        "traintest": "split",
    }
    project.design.variables["x"] = VariableDesign(knots=[-0.5, 0.0, 0.5])
    project.new_model(
        "test",
        family=family,
        divide_target_by_weight=divide,
        predictors=["x", "region"],
    )
    project.models["test"].penalty.alpha = 0.01
    return project


def _execute(project, run, tmp_path, monkeypatch):
    source = to_script(project, "test", run=run, output_prefix="exported")
    path = tmp_path / "rebuild.py"
    path.write_text(source)
    monkeypatch.chdir(tmp_path)
    namespace = runpy.run_path(str(path))
    return namespace, RateModel.from_json(tmp_path / "exported.easyglm")


def test_cv_interaction_script_reproduces_final_fit_and_adjusted_scorer(
    tmp_path, monkeypatch
):
    project = _project(tmp_path)
    cfg = project.models["test"]
    cfg.penalty.alpha = None
    cfg.penalty.cv = 3
    cfg.penalty.n_alphas = 6
    cfg.interactions = [Interaction("x", "region", min_cell_exposure=0.0)]
    cfg.base_rate_override = 1.4
    cfg.adjustments = [Adjustment("region", "B", "B", 1.15)]
    frame = prepare(project)
    run = run_model(project, frame, "test")
    assert isinstance(run.fit, TwoStageFit)
    assert np.count_nonzero(run.fit.stage2.coef) > 0

    namespace, rebuilt = _execute(project, run, tmp_path, monkeypatch)

    # The final cells use the final full-training mains, not the OOF predictors
    # that were used only to choose alpha. The original offset is included once.
    np.testing.assert_allclose(
        namespace["fit"].predict(frame), run.fit.predict(frame), rtol=1e-10
    )
    np.testing.assert_allclose(
        # CV warms the main fit; a cold resolved-alpha refit can move its
        # coefficients at solver tolerance, visible after manual overrides.
        rebuilt.predict(frame, exposure_col=None),
        run.predict(frame),
        rtol=1e-4,
    )
    assert rebuilt.base_rate == pytest.approx(cfg.base_rate_override)
    assert (tmp_path / "exported_rate_tables.xlsx").is_file()


@pytest.mark.parametrize(
    "family,divide,weighted",
    [
        ("poisson", True, True),
        ("poisson", False, True),
        ("binomial", False, True),
        ("poisson", False, False),
    ],
)
def test_script_prints_ae_on_canonical_model_basis(
    family, divide, weighted, tmp_path, monkeypatch, capsys
):
    project = _project(tmp_path, family=family, divide=divide, weighted=weighted)
    cfg = project.models["test"]
    cfg.adjustments = [Adjustment("region", "B", "B", 1.15)]
    frame = prepare(project)
    if family == "binomial":
        # Isolate exported scoring from the unrelated intercept-only benchmark
        # used by run_model's diagnostics. Build the same fit with the public API.
        train, _ = train_holdout(frame, project.data.split)
        spec = build_design(project, train, cfg.predictors, weight_col=cfg.weight)
        fit = fit_glm(
            train,
            spec,
            cfg.target,
            family=cfg.family,
            weight_col=cfg.weight,
            offset_col=cfg.offset,
            alpha=cfg.penalty.alpha,
        )
        rm = to_rate_model(fit, exposure_col=None)
        for adjustment in cfg.adjustments:
            rm.update_relativity(
                adjustment.variable,
                adjustment.from_,
                adjustment.to_,
                adjustment.relativity,
            )
        run = ModelRun(
            "test", cfg, spec, fit, rm, rate_tables(fit), {}, project.to_dict()
        )
    else:
        run = run_model(project, frame, "test")
    _, holdout = train_holdout(frame, project.data.split)
    actual, expected, _ = totals(holdout, cfg, run.predict(holdout))
    expected_ae = actual.sum() / expected.sum()

    namespace, rebuilt = _execute(project, run, tmp_path, monkeypatch)

    printed = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("holdout A/E:")
    )
    assert float(printed.split(":", 1)[1]) == pytest.approx(expected_ae, rel=1e-10)
    assert namespace["holdout_ae"] == pytest.approx(expected_ae, rel=1e-10)
    np.testing.assert_allclose(
        rebuilt.predict(holdout, exposure_col=None), run.predict(holdout), rtol=1e-10
    )
    if family == "binomial":
        assert rebuilt.metadata.exposure_col is None


def test_script_export_refuses_a_pathless_source():
    project = Project(name="In-memory data")
    project.new_model("test")
    with pytest.raises(ValueError, match="Save the source data to a file"):
        to_script(project, "test")


def test_script_replays_saved_feature_screen_before_final_fit(tmp_path, monkeypatch):
    project = _project(tmp_path)
    frame = prepare(project)
    # Save the candidate context first.  The final model is deliberately
    # reviewed down to x, while the screen must still assess region too.
    screening_project = project.to_dict()
    screening_project["models"] = {}
    screening_project["champion"] = None
    screening_project["exploration"] = {}
    project.models["test"].predictors = ["x"]
    project.data.roles["region"] = "ignore"
    project.exploration["feature_selection"] = {
        "version": 1,
        "project": screening_project,
        "options": {
            "family": "poisson",
            "link": None,
            "divide_target_by_weight": False,
            "n_alphas": 2,
            "repeats": 1,
            "seed": 11,
            "include_unassigned": False,
        },
        "result": {"rows": [{"variable": "old_browser_result"}]},
    }
    run = run_model(project, frame, "test")
    source = to_script(project, "test", run=run, output_prefix="screened")

    assert "RUN_FEATURE_SELECTION = True" in source
    assert source.index("select_variables(screening_project, df") < source.index(
        "# --------------------------------------------------------------- 2. split"
    )
    assert "old_browser_result" not in source
    assert "# final reviewed predictors: ['x']" in source

    path = tmp_path / "screened_rebuild.py"
    path.write_text(source)
    monkeypatch.chdir(tmp_path)
    namespace = runpy.run_path(str(path))
    report = json.loads((tmp_path / "screened_feature_selection.json").read_text())
    assert namespace["selection_options"]["importance_sample_pct"] == 100.0
    assert namespace["selection_options"]["link"] == "log"
    assert report["training_rows"] == 450
    assert report["tested_count"] == 2
    assert {row["variable"] for row in report["rows"]} == {"x", "region"}
    assert {row["status"] for row in report["rows"]} <= {"signal", "no_signal"}
    rebuilt = RateModel.from_json(tmp_path / "screened.easyglm")
    np.testing.assert_allclose(
        rebuilt.predict(frame, exposure_col=None), run.predict(frame), rtol=1e-10
    )
    assert namespace["spec"].main_effects == ["x"]

    # Changing only holdout outcomes and predictor values cannot affect the search.
    raw = pl.read_parquet(project.data.source.path)
    raw.with_columns(
        pl.when(pl.col("traintest") == 0).then(999.0).otherwise(pl.col("x")).alias("x"),
        pl.when(pl.col("traintest") == 0)
        .then(10000.0)
        .otherwise(pl.col("target"))
        .alias("target"),
    ).write_parquet(project.data.source.path)
    rerun = runpy.run_path(str(path))["feature_selection"]
    for before, after in zip(report["rows"], rerun["rows"], strict=True):
        assert before["variable"] == after["variable"]
        assert before["status"] == after["status"]
        assert before["importance"] == pytest.approx(after["importance"], abs=1e-10)
        assert before["threshold"] == pytest.approx(after["threshold"], abs=1e-10)


@pytest.mark.parametrize(
    "override",
    [VariableDesign(n_bins=3), VariableDesign(knots="integer", n_bins=3)],
)
def test_unfitted_export_preserves_per_variable_binning(
    tmp_path, monkeypatch, override
):
    project = _project(tmp_path)
    project.design.defaults.n_bins = 12
    project.design.variables["x"] = override
    frame = prepare(project)
    expected = run_model(project, frame, "test")
    namespace, rebuilt = _execute(project, None, tmp_path, monkeypatch)
    assert namespace["spec"].to_dict() == expected.spec.to_dict()
    np.testing.assert_allclose(
        rebuilt.predict(frame, exposure_col=None), expected.predict(frame), rtol=1e-10
    )


@pytest.mark.parametrize(
    "recipe", [{"version": 2}, {"version": 1, "project": {}, "options": []}]
)
def test_export_rejects_invalid_saved_search_recipe(tmp_path, recipe):
    project = _project(tmp_path)
    project.exploration["feature_selection"] = recipe
    with pytest.raises(ValueError, match="Saved feature-selection recipe"):
        to_script(project, "test")
