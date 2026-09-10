"""Execute exported scripts and compare their scores and reporting basis."""

from __future__ import annotations

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
