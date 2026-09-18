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
from easy_glm.workflow.export import to_script
from easy_glm.workflow.pair_distillation import distill_pair_cells
from easy_glm.workflow.project import (
    Adjustment,
    PairCandidateConfig,
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


def test_next_teacher_receives_exact_deployed_prefix(monkeypatch):
    project, frame = _case()
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
    assert set(run.rate_model.variables) == {"main"}  # pair-only parents stay pair-only
    assert run.pair_stages[1].baseline_stage_ids == ("ab",)


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
