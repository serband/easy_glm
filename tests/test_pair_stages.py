"""Sequential pair fitting uses deployed prefixes and fold-local search."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from easy_glm.engine.models import ModelMetadata
from easy_glm.engine.rate_model import RateModel
from easy_glm.workflow import pair_stages
from easy_glm.workflow import run as workflow_run
from easy_glm.workflow.export import to_script
from easy_glm.workflow.pair_distillation import distill_pair_cells
from easy_glm.workflow.project import (
    Adjustment,
    PairCandidateConfig,
    PairSearchConfig,
    PairStageConfig,
    Project,
    VariableDesign,
)
from easy_glm.workflow.run import rate_model_for, run_model


def _case() -> tuple[Project, pl.DataFrame]:
    rng = np.random.default_rng(102)
    n = 155
    main = rng.normal(size=n)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.normal(size=n)
    offset = rng.normal(0, 0.3, n)
    mu = np.exp(0.15 + 0.2 * main + 0.35 * a * b - 0.25 * b * c + offset)
    frame = pl.DataFrame(
        {
            "main": main,
            "A": a,
            "B": b,
            "C": c,
            "offset": offset,
            "y": rng.poisson(mu).astype(float),
            "split": [1] * 130 + [0] * 25,
        }
    )
    project = Project()
    project.data.roles = {
        "main": "predictor",
        "A": "predictor",
        "B": "predictor",
        "C": "predictor",
        "offset": "offset",
        "y": "target",
        "split": "split",
    }
    project.data.split.column = "split"
    cfg = project.new_model("pair")
    cfg.predictors = ["main"]
    cfg.penalty.alpha = 0.01
    candidate = PairCandidateConfig(2, 5, 0.1, 3)
    cfg.pair_stages = [
        PairStageConfig("ab", "A", "B", candidates=[candidate]),
        PairStageConfig("bc", "B", "C", candidates=[candidate]),
    ]
    assert not project.validate("pair")
    return project, frame


@pytest.mark.parametrize("with_main", [True, False])
def test_unassigned_pair_parents_fit_score_and_roundtrip(
    monkeypatch, tmp_path, with_main
):
    project, frame = _case()
    if not with_main:
        project.models["pair"].predictors = []
        project.models["pair"].pair_method = "sequential_catboost"
    for parent in ("A", "B", "C"):
        project.data.roles.pop(parent)
    assert not project.validate("pair", columns=frame.columns)
    real_distill = pair_stages.distill_pair_cells
    captured: list[np.ndarray] = []

    def spy(cell_ids, baseline, teacher, **kwargs):
        if len(baseline) == 130:
            captured.append(baseline.copy())
        return real_distill(cell_ids, baseline, teacher, **kwargs)

    monkeypatch.setattr(pair_stages, "distill_pair_cells", spy)
    run = run_model(project, frame, "pair")
    assert len(run.pair_stages) == len(run.rate_model.pair_tables) == 2
    assert len(captured) == 2
    for artifact in run.pair_stages:
        if artifact.chosen_candidate is not None:
            assert artifact.approximation_loss is not None
            assert artifact.approximation_loss >= -1e-10
            assert artifact.observed_table_minus_teacher_cv_loss == pytest.approx(
                artifact.table_cv_loss - artifact.teacher_cv_loss
            )
    first_only = copy.deepcopy(run.rate_model)
    first_only.pair_tables = first_only.pair_tables[:1]
    train = frame[:130]
    np.testing.assert_allclose(
        captured[1], first_only.predict(train, exposure_col=None), rtol=1e-12
    )
    np.testing.assert_allclose(
        run.predict(frame),
        rate_model_for(project, run).predict(frame, exposure_col=None),
        rtol=1e-12,
    )
    scorer_path = tmp_path / "pairs.easyglm"
    run.rate_model.to_json(scorer_path)
    np.testing.assert_allclose(
        run.predict(frame),
        RateModel.from_json(scorer_path).predict(frame, exposure_col=None),
        rtol=1e-12,
    )
    expected_mains = {"main"} if with_main else set()
    assert set(run.rate_model.variables) == expected_mains
    assert all(parent not in project.data.roles for parent in ("A", "B", "C"))
    assert run.pair_stages[1].baseline_stage_ids == ("ab",)
    if not with_main:
        data_path = tmp_path / "pair_only.parquet"
        frame.write_parquet(data_path)
        project.data.source.path = str(data_path)
        script_path = tmp_path / "pair_only_training.py"
        output_prefix = str(tmp_path / "pair_only_retrained")
        script_path.write_text(
            to_script(project, "pair", run=run, output_prefix=output_prefix)
        )
        assert "'pair_time_limit_minutes': 15.0" in script_path.read_text()
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=90,
        )
        exported = RateModel.from_json(output_prefix + ".easyglm")
        assert exported.variables == {}
        np.testing.assert_allclose(
            exported.predict(frame, exposure_col=None),
            run.predict(frame),
            rtol=1e-10,
        )


def test_append_and_holdout_poison_reuse_full_training_prefix():
    project, frame = _case()
    cfg = project.models["pair"]
    second = cfg.pair_stages.pop()
    cache: dict = {}
    first = run_model(project, frame, "pair", pair_stages_cache=cache)
    first_table = first.rate_model.to_dict()["pair_tables"][0]
    cfg.pair_stages.append(second)
    combined = run_model(project, frame, "pair", pair_stages_cache=cache)
    assert combined.pair_stages[0].reused is True
    assert combined.rate_model.to_dict()["pair_tables"][0] == first_table
    poisoned = frame.with_columns(
        pl.when(pl.col("split") == 0).then(1e6).otherwise(pl.col("y")).alias("y")
    )
    repeated = run_model(project, poisoned, "pair", pair_stages_cache=cache)
    assert all(artifact.reused for artifact in repeated.pair_stages)
    np.testing.assert_allclose(
        combined.predict(frame), repeated.predict(frame), rtol=1e-12
    )


def test_uniform_twenty_by_twenty_cells_clear_default_support_threshold():
    cell = np.repeat(np.arange(400), 10)
    result = distill_pair_cells(
        cell,
        np.ones(len(cell)),
        np.ones(len(cell)),
        n_cells=400,
        min_weight_share=PairStageConfig("s", "a", "b").min_weight_share,
    )
    assert result.fallback_reason == (None,) * 400
    assert np.all(result.relativities == 1)


def test_configured_deadline_is_absolute_and_forwarded_unchanged(monkeypatch):
    project, frame = _case()
    project.models["pair"].pair_time_limit_minutes = 2.0
    estimate = pair_stages.preflight_pair_stages(
        project.models["pair"].pair_stages,
        project,
        frame[:130],
        project.models["pair"],
    )
    assert estimate["deadline_seconds"] == 120.0
    settings = pair_stages._main_settings(project.models["pair"])
    project.models["pair"].pair_time_limit_minutes = 3.0
    assert pair_stages._main_settings(project.models["pair"]) == settings
    project.models["pair"].pair_time_limit_minutes = 2.0

    class RunClock:
        @staticmethod
        def monotonic():
            return 100.0

    class PairClock:
        current = 101.0

        @classmethod
        def monotonic(cls):
            cls.current += 0.001
            return cls.current

        @classmethod
        def perf_counter(cls):
            cls.current += 0.001
            return cls.current

    monkeypatch.setattr(workflow_run, "time", RunClock)
    monkeypatch.setattr(pair_stages, "time", PairClock)
    seen: list[float] = []
    original_check = pair_stages._check_deadline

    def record(deadline, cfg):
        seen.append(deadline)
        return original_check(deadline, cfg)

    monkeypatch.setattr(pair_stages, "_check_deadline", record)
    run_model(project, frame, "pair")

    assert len(seen) > 20  # fold-local mains, candidates and both full stage fits
    assert set(seen) == {220.0}


def test_tiny_configured_limit_fails_with_retry_message_before_teacher(monkeypatch):
    project, frame = _case()
    cfg = project.models["pair"]
    cfg.pair_time_limit_minutes = 0.000001
    called = False

    def teacher(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError(
            "teacher should not start after the main fit used the limit"
        )

    monkeypatch.setattr(pair_stages, "fit_catboost_pair_raw", teacher)
    with pytest.raises(
        TimeoutError,
        match=(
            r"The fit reached its 1e-06-minute time limit\. Increase Fit time limit "
            r"in Model > Fit settings, save, and retry\."
        ),
    ):
        run_model(project, frame, "pair")
    assert called is False


def test_direct_staged_fit_respects_an_explicit_zero_deadline() -> None:
    project, frame = _case()
    cfg = project.models["pair"]
    with pytest.raises(TimeoutError, match="15-minute time limit"):
        pair_stages.fit_pair_stages(
            project,
            frame[:130],
            cfg,
            RateModel(1.0, {}),
            deadline_monotonic=0.0,
        )


def test_sequential_mode_without_pair_stages_has_no_pair_deadline() -> None:
    project, frame = _case()
    cfg = project.models["pair"]
    cfg.pair_stages = []
    cfg.pair_method = "sequential_catboost"
    cfg.pair_time_limit_minutes = 0.000001
    run = run_model(project, frame, "pair")
    assert run.pair_stages == []


def test_rejects_invalid_numeric_teacher_baseline():
    project, frame = _case()
    cfg = project.models["pair"]
    prefix = RateModel(1.0, {}, metadata=ModelMetadata(offset_col="offset"))
    with pytest.raises(ValueError, match="prefix baseline"):
        broken = frame.with_columns(pl.lit(1000.0).alias("offset"))
        # An extreme external offset cannot be clipped into an apparently valid fit.
        pair_stages._fit_table(
            project,
            cfg,
            cfg.pair_stages[0],
            cfg.pair_stages[0].candidates[0],
            broken[:130],
            prefix,
            deadline=time.monotonic() + 10,
        )


def test_pair_edit_refit_requires_clear_or_explicit_recipe_replay():
    project, frame = _case()
    cfg = project.models["pair"]
    second = cfg.pair_stages.pop()
    for name in ("A", "B", "C"):
        project.design.variables[name] = VariableDesign(knots=[-1.0, 0.0, 1.0])
    first = run_model(project, frame, "pair")
    table = first.rate_model.pair_tables[0]
    row_a, row_b = table.axes[0].table[0], table.axes[1].table[0]
    cfg.adjustments.append(
        Adjustment(
            variable="pair:ab",
            from_=row_a.from_,
            to_=row_a.to_,
            relativity=2.0,
            from_b=row_b.from_,
            to_b=row_b.to_,
            cell=True,
            stage_id="ab",
            axis_a_row=0,
            axis_b_row=0,
        )
    )
    with pytest.raises(ValueError, match="replaces its manual cell edits"):
        run_model(project, frame, "pair")
    replay = run_model(project, frame, "pair", replay_pair_adjustments=True)
    assert replay.rate_model.get_pair_table("ab").cell_matrix[0, 0] == 2.0
    cfg.pair_stages.append(second)
    cache = {
        "full_prefix": {first.pair_stages[0].prefix_fingerprint: first.pair_stages[0]}
    }
    appended = run_model(project, frame, "pair", pair_stages_cache=cache)
    assert appended.pair_stages[0].reused
    assert appended.rate_model.get_pair_table("ab").cell_matrix[0, 0] == 2.0


def test_training_export_replays_adjusted_prefix_before_downstream(tmp_path):
    project, frame = _case()
    for parent in ("A", "B", "C"):
        project.data.roles.pop(parent)
    assert not project.validate("pair", columns=frame.columns)
    data_path = tmp_path / "book.parquet"
    frame.write_parquet(data_path)
    project.data.source.type = "parquet"
    project.data.source.path = str(data_path)
    cfg = project.models["pair"]
    second = cfg.pair_stages.pop()
    for name in ("A", "B", "C"):
        project.design.variables[name] = VariableDesign(knots=[-1.0, 0.0, 1.0])
    initial = run_model(project, frame, "pair")
    first_table = initial.rate_model.get_pair_table("ab")
    a_row, b_row = first_table.axes[0].table[0], first_table.axes[1].table[0]
    cfg.adjustments.append(
        Adjustment(
            variable="pair:ab",
            from_=a_row.from_,
            to_=a_row.to_,
            relativity=2.0,
            from_b=b_row.from_,
            to_b=b_row.to_,
            cell=True,
            stage_id="ab",
            axis_a_row=0,
            axis_b_row=0,
        )
    )
    cfg.pair_stages.append(second)
    direct = run_model(project, frame, "pair", replay_pair_adjustments=True)
    script_path = tmp_path / "replay.py"
    output_prefix = str(tmp_path / "retrained")
    script_path.write_text(
        to_script(project, "pair", run=direct, output_prefix=output_prefix)
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    saved = RateModel.from_json(output_prefix + ".easyglm")
    assert len(saved.pair_tables) == 2
    assert saved.get_pair_table("ab").cell_matrix[0, 0] == 2.0
    np.testing.assert_allclose(
        saved.predict(frame, exposure_col=None),
        direct.predict(frame),
        rtol=1e-7,
        atol=1e-8,
    )


def test_grid_limit_fails_before_main_fit(monkeypatch):
    project, frame = _case()
    project.models["pair"].pair_stages.pop()
    levels = [str(index) for index in range(101)]
    project.design.variables["A"] = VariableDesign(kind="categorical", levels=levels)
    project.design.variables["B"] = VariableDesign(kind="categorical", levels=levels)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("main fit started before pair preflight")

    monkeypatch.setattr("easy_glm.workflow.run._fit_main_effects", forbidden_fit)
    with pytest.raises(ValueError, match="above the 10,000 limit"):
        run_model(project, frame, "pair")


@pytest.mark.parametrize("auto_stage", ["ab", "bc"])
def test_optuna_mixed_chain_keeps_fold_local_prefixes(auto_stage):
    project, frame = _case()
    stages = project.models["pair"].pair_stages
    for stage in stages:
        if stage.stage_id == auto_stage:
            stage.candidates = []
            stage.search = (
                PairSearchConfig(trials=4, prefix_trials=3)
                if auto_stage == "ab"
                else PairSearchConfig(trials=2, prefix_trials=2)
            )
    run = run_model(project, frame, "pair")
    first, second = run.pair_stages
    assert len(run.rate_model.pair_tables) == 2
    assert len(first.selected_prefix_parameters) == 5
    assert len(second.selected_prefix_parameters) == 5
    assert all(len(parameters) == 1 for parameters in second.selected_prefix_parameters)
    if auto_stage == "ab":
        assert len(first.search_trials) == 4
        assert len(first.cv_candidates) == 5  # neutral plus four TPE proposals
        assert len(second.search_trials) == 0
        assert len(second.prefix_search_trials) == 20  # five independent studies
        assert [
            sum(record.outer_fold == fold for record in second.prefix_search_trials)
            for fold in range(5)
        ] == [
            4
        ] * 5  # neutral plus three proposals per outer-training partition
    else:
        assert len(first.search_trials) == 0
        assert len(second.search_trials) == 2
        assert len(second.prefix_search_trials) == 0
        assert len(second.cv_candidates) == 3
        assert all(len(choices) == 1 for choices in second.selected_prefix_configs)
    for artifact in run.pair_stages:
        assert artifact.table_cv_loss <= artifact.prefix_cv_loss + 1e-8
        if artifact.search_trials:
            assert all(record.state == "COMPLETE" for record in artifact.search_trials)
            assert all(len(record.folds) == 5 for record in artifact.search_trials)


def test_optuna_holdout_target_poison_leaves_every_search_trace_unchanged():
    project, frame = _case()
    stages = project.models["pair"].pair_stages
    stages[0].candidates = []
    stages[0].search = PairSearchConfig(trials=2, prefix_trials=2)
    stages[1].candidates = []
    stages[1].search = PairSearchConfig(trials=2, prefix_trials=2)
    original = run_model(project, frame, "pair")
    poisoned = frame.with_columns(
        pl.when(pl.col("split") == 0)
        .then(pl.lit(100_000.0))
        .otherwise(pl.col("y"))
        .alias("y")
    )
    repeated = run_model(project, poisoned, "pair")
    for first, second in zip(original.pair_stages, repeated.pair_stages, strict=True):
        assert first.chosen_candidate == second.chosen_candidate
        assert first.cv_candidates == second.cv_candidates
        assert first.search_trials == second.search_trials
        assert first.prefix_search_trials == second.prefix_search_trials
        assert first.selected_prefix_parameters == second.selected_prefix_parameters
        assert first.prefix_fingerprint == second.prefix_fingerprint


def test_prefix_study_cache_changes_when_current_stage_seed_changes():
    project, frame = _case()
    first, second = project.models["pair"].pair_stages
    first.search = PairSearchConfig(trials=1, prefix_trials=1)
    first.candidates = []
    cache: dict = {}
    run_model(project, frame, "pair", pair_stages_cache=cache)
    assert len(cache["prefix_selection"]) == 5
    second.seed += 1
    run_model(project, frame, "pair", pair_stages_cache=cache)
    assert len(cache["prefix_selection"]) == 10
