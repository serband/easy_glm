"""Unchanged mains are reused; interaction fits and CV stay statistically unchanged."""

from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from easy_glm.core.fit import TwoStageFit, fit_glm
from easy_glm.workflow.prep import prepare
from easy_glm.workflow.project import Interaction, Project, VariableDesign
from easy_glm.workflow.run import run_model


@pytest.fixture
def problem():
    rng = np.random.default_rng(482)
    n = 800
    a = rng.choice(["a", "b", "c"], n)
    b = rng.choice(["x", "y"], n)
    raw = pl.DataFrame(
        {
            "a": a,
            "b": b,
            "y": rng.poisson(np.exp(0.2 + 0.4 * (a == "b") + 0.3 * (b == "y")), n),
            "w": rng.uniform(0.5, 1.5, n),
            "off": rng.normal(0, 0.1, n),
        }
    )
    project = Project()
    project.data.roles = {
        "a": "predictor",
        "b": "predictor",
        "y": "target",
        "w": "weight",
    }
    project.data.split.mode = "random"
    cfg = project.new_model("m", family="poisson")
    cfg.penalty.alpha = 0.02
    cfg.penalty.cv = None
    cfg.offset = "off"
    return project, raw


def test_add_edit_remove_interactions_reuses_exact_mains(problem):
    project, raw = problem
    frame = prepare(project, raw)
    cache = {}
    original = run_model(project, frame, "m", main_effects_cache=cache)
    original_spec = original.fit.spec
    project.models["m"].interactions = [Interaction("a", "b", min_cell_exposure=0)]
    with patch("easy_glm.core.fit.fit_glm", wraps=fit_glm) as calls:
        adjusted = run_model(project, frame, "m", main_effects_cache=cache)
    assert calls.call_count == 1
    assert calls.call_args.args[1].interactions
    assert isinstance(adjusted.fit, TwoStageFit)
    np.testing.assert_array_equal(original.fit.coef, adjusted.fit.stage1.coef)
    assert original.fit.intercept == adjusted.fit.intercept
    assert original.fit.spec is original_spec
    fresh = run_model(project, frame, "m")
    np.testing.assert_allclose(
        adjusted.fit.predict(frame), fresh.fit.predict(frame), rtol=1e-10
    )
    project.models["m"].interactions[0].penalty_weight = 2
    with patch("easy_glm.core.fit.fit_glm", wraps=fit_glm) as calls:
        run_model(project, frame, "m", main_effects_cache=cache)
    assert calls.call_count == 1
    project.models["m"].interactions = []
    with patch("easy_glm.core.fit.fit_glm", wraps=fit_glm) as calls:
        restored = run_model(project, frame, "m", main_effects_cache=cache)
    calls.assert_not_called()
    np.testing.assert_array_equal(
        original.fit.predict(frame), restored.fit.predict(frame)
    )


@pytest.mark.parametrize(
    "change",
    [
        "target",
        "weight",
        "offset",
        "row_order",
        "split",
        "penalty",
        "predictors",
        "design",
        "family",
    ],
)
def test_changed_main_problem_refits(problem, change):
    project, raw = problem
    cache = {}
    run_model(project, prepare(project, raw), "m", main_effects_cache=cache)
    if change in {"target", "weight", "offset"}:
        col = {"target": "y", "weight": "w", "offset": "off"}[change]
        raw = raw.with_columns((pl.col(col) + 0.1).alias(col))
    elif change == "row_order":
        raw = raw.reverse()
    elif change == "split":
        project.data.split.seed += 1
    elif change == "penalty":
        project.models["m"].penalty.alpha = 0.04
    elif change == "predictors":
        project.models["m"].predictors = ["a"]
    elif change == "family":
        project.models["m"].family = "gaussian"
    else:
        project.design.variables["a"] = VariableDesign(
            kind="categorical", penalty_weight=2
        )
    with patch("easy_glm.core.fit.fit_glm", wraps=fit_glm) as calls:
        run_model(project, prepare(project, raw), "m", main_effects_cache=cache)
    assert calls.call_count == 1
    assert not calls.call_args.args[1].interactions


def test_cv_interaction_edits_reuse_main_and_cross_fitted_offsets(problem):
    project, raw = problem
    pen = project.models["m"].penalty
    pen.alpha, pen.cv, pen.n_alphas = None, 2, 3
    frame = prepare(project, raw)
    cache = {}
    main = run_model(project, frame, "m", main_effects_cache=cache)
    project.models["m"].interactions = [Interaction("a", "b", min_cell_exposure=0)]
    with patch("easy_glm.core.fit.fit_glm", wraps=fit_glm) as calls:
        first = run_model(project, frame, "m", main_effects_cache=cache)
    # Two fold-specific mains plus CV and final interaction fits. No full main refit.
    assert calls.call_count == 4
    np.testing.assert_array_equal(main.fit.coef, first.fit.stage1.coef)
    oof = cache["oof"].copy()
    project.models["m"].interactions[0].penalty_weight = 2
    with patch("easy_glm.core.fit.fit_glm", wraps=fit_glm) as calls:
        second = run_model(project, frame, "m", main_effects_cache=cache)
    assert calls.call_count == 2
    assert all(call.args[1].interactions for call in calls.call_args_list)
    np.testing.assert_array_equal(oof, cache["oof"])
    fresh = run_model(project, frame, "m")
    np.testing.assert_allclose(
        second.fit.predict(frame), fresh.fit.predict(frame), rtol=1e-9
    )
    pen.n_alphas = 4
    with patch("easy_glm.core.fit.fit_glm", wraps=fit_glm) as calls:
        run_model(project, frame, "m", main_effects_cache=cache)
    assert calls.call_count == 5


def test_worker_cache_survives_separate_worker_calls_and_corruption(problem, tmp_path):
    from easy_glm.desktop.fit_worker import fit_result

    project, raw = problem
    path = tmp_path / "main-cache.pkl"
    fit_result(project, raw, "m", lambda _: None, main_cache_path=path)
    project.models["m"].interactions = [Interaction("a", "b", min_cell_exposure=0)]
    result = fit_result(project, raw, "m", lambda _: None, main_cache_path=path)
    assert result["reuse"]["main_effects"] is True
    path.write_bytes(b"corrupt disposable cache")
    result = fit_result(project, raw, "m", lambda _: None, main_cache_path=path)
    assert result["reuse"]["main_effects"] is False


def test_live_jobs_retain_mains_when_saved_interactions_invalidate_result(problem):
    import time

    from easy_glm.desktop.jobs import FitJobs

    project, raw = problem
    jobs = FitJobs()

    def finish():
        for _ in range(600):
            status = jobs.status(project)["m"]["status"]
            if status in ("complete", "failed"):
                assert status == "complete", jobs.status(project)
                return jobs.result(project, "m")
            time.sleep(0.05)
        pytest.fail("Local fitting worker did not complete")

    try:
        jobs.start(project, raw, "m")
        assert finish()["reuse"]["main_effects"] is False
        project.models["m"].interactions = [Interaction("a", "b", min_cell_exposure=0)]
        jobs.invalidate(project)
        jobs.start(project, raw, "m")
        assert finish()["reuse"]["main_effects"] is True
    finally:
        jobs.close()
