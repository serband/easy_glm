"""Report diagnostics are original-fit evidence, not adjusted rate-table effects."""

from __future__ import annotations

import copy

import polars as pl
import pytest
import test_d3_d4_compare_report as fixtures

from easy_glm.workflow._report_diagnostics import fitted_diagnostics_section
from easy_glm.workflow.diagnostics import permutation_importance
from easy_glm.workflow.prep import train_holdout
from easy_glm.workflow.report import to_report_html
from easy_glm.workflow.run import run_model

workspace = fixtures.workspace


@pytest.fixture(scope="module")
def cv_report(workspace):
    project = copy.deepcopy(workspace["project"])
    project.design.defaults.n_bins = 4
    cfg = project.models[fixtures.MODEL_B]
    cfg.penalty.alpha = None
    cfg.penalty.cv = 3
    cfg.penalty.n_alphas = 4
    run = run_model(project, workspace["frame"], fixtures.MODEL_B)
    return to_report_html(
        project, {run.name: run}, workspace["frame"], champion=run.name
    )


def test_importance_and_actual_coefficient_paths_precede_individual_factors(cv_report):
    assert cv_report.index('id="variable-importance"') < cv_report.index(
        'id="coefficient-paths"'
    )
    assert cv_report.index('id="coefficient-paths"') < cv_report.index(
        'id="var-drivage"'
    )
    assert "Training permutation importance — original fit" in cv_report
    assert "Main effects: coefficients versus lambda" in cv_report
    assert "Interaction cells: coefficients versus lambda" in cv_report
    assert "Mean coefficients across 3 validation folds." in cv_report
    assert cv_report.count("Selected lambda:") == 2
    assert "<script" not in cv_report
    assert len(cv_report.encode()) < 5_000_000


def test_fixed_lambda_does_not_fabricate_a_coefficient_path(workspace):
    run = workspace["runs"][fixtures.MODEL_A]
    train, _ = train_holdout(workspace["frame"], workspace["project"].data.split)
    section = fitted_diagnostics_section(workspace["project"], run, train)
    assert (
        "Fitted at a single lambda (2e-3); no coefficient path was recorded." in section
    )
    assert "Main effects: coefficients versus lambda" not in section


def test_adjustments_do_not_change_report_importance_or_stored_paths(workspace):
    original = workspace["runs"][fixtures.MODEL_A]
    adjusted = copy.deepcopy(original)
    row = adjusted.rate_model.variables["DrivAge"].table[1]
    adjusted.rate_model.update_relativity("DrivAge", row.from_, row.to_, 9.876)
    train, _ = train_holdout(workspace["frame"], workspace["project"].data.split)
    assert fitted_diagnostics_section(
        workspace["project"], original, train
    ) == fitted_diagnostics_section(workspace["project"], adjusted, train)
    assert original.rate_model.variables["DrivAge"].table[1].relativity != 9.876


def test_report_uses_training_only_and_accepts_matching_cached_importance(
    workspace, monkeypatch
):
    project = workspace["project"]
    run = workspace["runs"][fixtures.MODEL_A]
    frame = workspace["frame"]
    train, _ = train_holdout(frame, project.data.split)
    cached = permutation_importance(run.fit, train)
    expected = fitted_diagnostics_section(project, run, train, importance=cached)

    def no_scoring(*args, **kwargs):
        raise AssertionError("Cached importance must avoid repeated scoring")

    monkeypatch.setattr(
        "easy_glm.workflow._report_diagnostics.permutation_importance", no_scoring
    )
    changed = frame.with_columns(
        pl.when(pl.col("traintest") == 0)
        .then(999_999)
        .otherwise(pl.col("DrivAge"))
        .alias("DrivAge")
    )
    report = to_report_html(
        project, {run.name: run}, changed, champion=run.name, importance=cached
    )
    assert expected in report


def test_one_training_row_has_an_explicit_importance_message(workspace):
    run = workspace["runs"][fixtures.MODEL_A]
    section = fitted_diagnostics_section(
        workspace["project"], run, workspace["frame"].head(1)
    )
    assert "At least two training rows are needed." in section


def test_report_importance_uses_diagnostics_protected_roles(workspace, monkeypatch):
    project = copy.deepcopy(workspace["project"])
    project.data.roles["Premium"] = "current_premium"
    run = workspace["runs"][fixtures.MODEL_A]
    train, _ = train_holdout(workspace["frame"], project.data.split)
    calls = []

    def importance(fit, data, **kwargs):
        calls.append((fit, data, kwargs))
        return pl.DataFrame()

    monkeypatch.setattr(
        "easy_glm.workflow._report_diagnostics.permutation_importance", importance
    )
    section = fitted_diagnostics_section(project, run, train)
    fit, data, options = calls[0]
    assert fit is run.fit and data is train
    assert set(options["protected_columns"]) == {
        "ClaimNb",
        "Exposure",
        "IDpol",
        "traintest",
        "Premium",
    }
    assert options["seed"] == 42 and options["repeats"] == 5
    assert "No eligible predictors in this fit." in section
