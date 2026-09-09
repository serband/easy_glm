"""Cached A/E equals canonical diagnostics and tracks the full scoring basis."""

import copy
import pickle

import numpy as np
import polars as pl
import pytest

from easy_glm.desktop.ae_cache import (
    build_packet,
    cache_path,
    read_packet,
    variable_view,
)
from easy_glm.desktop.fit_worker import json_safe
from easy_glm.desktop.review_worker import review
from easy_glm.desktop.reviews import ReviewJobs
from easy_glm.workflow.prep import prepare
from easy_glm.workflow.project import Adjustment, ModelConfig, Project, VariableDesign
from easy_glm.workflow.run import rebuild_rate_model, run_model


def fixture(tmp_path, kind="step"):
    rng = np.random.default_rng(3)
    x = rng.integers(1, 8, 400).astype(float)
    x[::9] = np.nan
    raw = pl.DataFrame(
        {
            "x": (
                x
                if kind != "categorical"
                else [str(v) if np.isfinite(v) else None for v in x]
            ),
            "z": rng.integers(1, 5, 400),
            "y": rng.poisson(1, 400),
            "w": rng.uniform(0.1, 2, 400),
        }
    )
    p = Project(name="Cached A/E")
    p.data.roles = {"x": "predictor", "z": "predictor", "y": "target", "w": "weight"}
    p.data.split.mode = "random"
    p.design.variables["x"] = VariableDesign(kind=kind, n_bins=4)
    p.models["M"] = ModelConfig(
        target="y", weight="w", divide_target_by_weight=True, predictors=["x", "z"]
    )
    frame = prepare(p, raw)
    run = run_model(p, frame, "M")
    source = tmp_path / "fit-1"
    source.mkdir()
    (source / "fit.pkl").write_bytes(pickle.dumps(run))
    raw.write_parquet(source / "raw.parquet")
    request = {
        "model": "M",
        "action": "variable",
        "variable": "x",
        "subset": "holdout",
        "options": {"both_subsets": True},
    }
    return p, raw, frame, run, source, request


@pytest.mark.parametrize("kind", ["step", "linear", "categorical"])
def test_cached_rows_equal_on_demand_for_every_subset(tmp_path, kind):
    p, raw, frame, run, source, request = fixture(tmp_path, kind)
    packet = build_packet(p, run, frame, source, request)
    for subset in ["train", "holdout", "all"]:
        req = {**request, "subset": subset}
        cached = variable_view(packet, req)
        direct = json_safe(review(copy.deepcopy(p), run, raw, req))
        assert cached["subset"] == subset
        assert cached["rows"] == direct["rows"]
        assert cached["ae_sets"] == direct["ae_sets"]
        assert cached["book_impact"] == pytest.approx(direct["book_impact"])
    assert read_packet(p, source, request) == packet


def test_any_factor_or_base_edit_invalidates_adjusted_only_and_fast_hit_ignores_worker(
    tmp_path,
):
    p, raw, frame, run, source, request = fixture(tmp_path)
    original = cache_path(p, source, request, original=True)
    packet = build_packet(p, run, frame, source, request)
    original_bytes = original.read_bytes()
    jobs = ReviewJobs()
    jobs.tasks["unrelated"] = {"status": "running", "cancel": False, "process": None}
    task = jobs.start(p, source, request, 0)
    assert jobs.get(task["id"])["status"] == "complete"
    assert "thread" not in jobs.get(task["id"])
    jobs.tasks.pop("unrelated")
    jobs.close()
    row = run.rate_model.variables["z"].table[1]
    p.models["M"].adjustments = [Adjustment("z", row.from_, row.to_, 2.7)]
    assert read_packet(p, source, request) is None
    assert cache_path(p, source, request, original=True) == original
    rebuild_rate_model(p, run, frame)
    changed = build_packet(p, run, frame, source, request)
    assert original.read_bytes() == original_bytes
    assert [
        r["fitted_rate"] for r in changed["variables"]["x"]["subsets"]["train"]
    ] == [r["fitted_rate"] for r in packet["variables"]["x"]["subsets"]["train"]]
    assert (
        changed["variables"]["x"]["subsets"]["train"]
        != packet["variables"]["x"]["subsets"]["train"]
    )
    p.models["M"].base_rate_override = run.rate_model.base_rate * 2
    assert read_packet(p, source, request) is None
    assert cache_path(p, source, request, original=True) == original
    p.models["M"].base_rate_override = None
    p.models["M"].adjustments = []
    assert (
        read_packet(p, source, request) == packet
    )  # Undo reuses the earlier applied basis.
    p.models["M"].notes = "metadata only"
    assert read_packet(p, source, request) == packet


def test_challenger_uses_selected_groups_and_adjustment_identity(tmp_path):
    p, raw, frame, run, source, request = fixture(tmp_path)
    p.models["C"] = copy.deepcopy(p.models["M"])
    alternative = copy.deepcopy(p)
    alternative.design.variables["x"].n_bins = 2
    challenger = run_model(alternative, frame, "C")
    other_source = tmp_path / "fit-2"
    other_source.mkdir()
    (other_source / "fit.pkl").write_bytes(pickle.dumps(challenger))
    raw.write_parquet(other_source / "raw.parquet")
    request.update(challenger="C", _challenger_source=str(other_source))
    packet = build_packet(p, run, frame, source, request, challenger)
    direct = json_safe(review(copy.deepcopy(p), run, raw, request, challenger))
    assert variable_view(packet, request)["rows"] == direct["rows"]
    original_key = cache_path(p, source, request, original=True)
    row = challenger.rate_model.variables["z"].table[1]
    p.models["C"].adjustments = [Adjustment("z", row.from_, row.to_, 3)]
    assert read_packet(p, source, request) is None
    assert cache_path(p, source, request, original=True) == original_key


def test_cache_failure_does_not_fail_fit(tmp_path, monkeypatch):
    from easy_glm.desktop import ae_cache
    from easy_glm.desktop.fit_worker import fit_result

    p, raw, _, _, source, _ = fixture(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("Cache unavailable")

    monkeypatch.setattr(ae_cache, "build_packet", fail)
    result = fit_result(p, raw, "M", lambda message: None, source)
    assert result["summary"]["name"] == "M"
    assert (source / "fit.pkl").exists()


def test_fit_completion_writes_ready_cache(tmp_path):
    from easy_glm.desktop.fit_worker import fit_result

    p, raw, _, _, source, request = fixture(tmp_path)
    fit_result(p, raw, "M", lambda message: None, source)
    packet = read_packet(p, source, request)
    assert set(packet["variables"]) == {"x", "z"}
    assert set(packet["variables"]["x"]["subsets"]) == {"train", "holdout", "all"}


def test_probability_link_cache_matches_canonical_without_exposure_scaling(tmp_path):
    p, raw, _, _, source, request = fixture(tmp_path)
    raw = raw.with_columns((pl.col("y") > 0).cast(pl.Float64))
    cfg = p.models["M"]
    cfg.family, cfg.link, cfg.weight, cfg.divide_target_by_weight = (
        "binomial",
        "logit",
        None,
        False,
    )
    frame = prepare(p, raw)
    run = run_model(p, frame, "M")
    (source / "fit.pkl").write_bytes(pickle.dumps(run))
    raw.write_parquet(source / "raw.parquet")
    packet = build_packet(p, run, frame, source, request)
    for subset in ["train", "holdout", "all"]:
        req = {**request, "subset": subset}
        assert (
            variable_view(packet, req)["rows"]
            == json_safe(review(copy.deepcopy(p), run, raw, req))["rows"]
        )
