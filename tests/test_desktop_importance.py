"""Original-fit importance caches and review routes never apply table changes."""

import copy
import pickle

import pytest
import test_desktop_models as fixtures
from test_desktop_ae_cache import fixture

from easy_glm.desktop.fit_worker import fit_result
from easy_glm.desktop.importance_cache import build_packet, cache_path, read_packet
from easy_glm.desktop.review_worker import review
from easy_glm.desktop.reviews import ReviewJobs
from easy_glm.workflow.prep import train_holdout
from easy_glm.workflow.project import Adjustment
from easy_glm.workflow.run import rebuild_rate_model

importance_session = fixtures.model_session


def test_cache_is_training_original_and_survives_adjustments(tmp_path, monkeypatch):
    p, raw, frame, run, source, _ = fixture(tmp_path)
    p.to_json(source / "project.json")
    packet = build_packet(p, run, frame, source)
    assert packet["training_rows"] == train_holdout(frame, p.data.split)[0].height
    assert packet["subset"] == "train" and packet["basis"] == "original"
    assert packet["repeats"] == 5 and packet["seed"] == 42
    assert packet["fit_id"] == source.name
    original_bytes = (source / "fit.pkl").read_bytes()
    original_packet = cache_path(source).read_bytes()
    row = run.rate_model.variables["x"].table[1]
    p.models["M"].adjustments = [Adjustment("x", row.from_, row.to_, 5)]
    p.models["M"].base_rate_override = run.rate_model.base_rate * 2
    rebuild_rate_model(p, run, frame)
    assert build_packet(p, run, frame) == {**packet, "fit_id": None}
    # A cache hit skips preparation, rebuild, and any challenger checks.
    monkeypatch.setattr(
        "easy_glm.desktop.review_worker.prepare",
        lambda *args: pytest.fail("prepared on hit"),
    )
    for action, options in [
        ("importance", {}),
        ("coefficients", {"view": "importance"}),
    ]:
        request = {
            "action": action,
            "options": options,
            "subset": "holdout",
            "model": "M",
        }
        assert review(p, run, raw, request, object(), source) == packet
        jobs = ReviewJobs()
        jobs.tasks["unrelated"] = {
            "status": "running",
            "cancel": False,
            "process": None,
        }
        task = jobs.get(jobs.start(p, source, request, 2)["id"])
        assert task["status"] == "complete" and task["data"] == packet
        assert "thread" not in task and "project" not in task["data"]
        jobs.tasks.pop("unrelated")
        jobs.close()
    assert (source / "fit.pkl").read_bytes() == original_bytes
    assert cache_path(source).read_bytes() == original_packet
    (source / "fit.pkl").write_bytes(pickle.dumps(run))
    assert read_packet(source) is None


def test_lazy_route_uses_artifact_training_project_and_cache(tmp_path):
    p, raw, frame, run, source, _ = fixture(tmp_path)
    p.to_json(source / "project.json")
    expected = build_packet(p, run, frame)
    changed = copy.deepcopy(p)
    changed.data.split.fraction = 0.1
    changed.data.roles["x"] = "ignore"
    packet = review(
        changed,
        run,
        raw,
        {"action": "coefficients", "options": {"view": "importance"}},
        source=source,
    )
    assert packet == {**expected, "fit_id": source.name}
    assert read_packet(source) == packet


def test_optional_diagnostic_failure_cannot_discard_good_fit(tmp_path, monkeypatch):
    from easy_glm.desktop import importance_cache

    p, raw, _, _, source, _ = fixture(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(importance_cache, "build_packet", fail)
    result = fit_result(p, raw, "M", lambda message: None, source)
    assert result["summary"]["name"] == "M"
    assert (source / "fit.pkl").exists()


def test_fit_prepares_packet_and_read_only_storage_still_returns_result(
    tmp_path, monkeypatch
):
    p, raw, frame, run, source, _ = fixture(tmp_path)
    fit_result(p, raw, "M", lambda message: None, source)
    assert read_packet(source)["rows"]

    def fail(*args, **kwargs):
        raise OSError("read-only")

    monkeypatch.setattr("easy_glm.desktop.fit_worker.write_json", fail)
    assert build_packet(p, run, frame, source)["rows"]


def test_importance_api_is_read_only_and_reuses_original_after_edit(importance_session):
    from test_desktop_models import revision, save_model, wait_fit
    from test_desktop_reviews import apply
    from test_desktop_reviews import review as request_review

    client, _, _ = importance_session
    save_model(client)
    client.post("/api/models/Frequency/fit", json=revision(client))
    assert wait_fit(client)["status"] == "complete"
    snapshot = client.get("/api/project").json()
    first = request_review(
        client, "importance", subset="holdout", challenger="Frequency"
    )
    assert first["status"] == "complete" and not first["can_apply"]
    packet = first["data"]
    assert packet["subset"] == "train" and packet["basis"] == "original"
    assert client.get("/api/project").json() == snapshot
    edit = request_review(client, "edit", variable="Age", edits={"1": 2.1})
    apply(client, edit)
    second = request_review(client, "importance")
    assert second["data"] == packet
    bridged = request_review(client, "coefficients", options={"view": "importance"})
    assert bridged["data"] == packet and not bridged["can_apply"]
    # An earlier completed importance remains valid after overlay edits.
    assert client.get("/api/reviews/" + first["id"]).json()["status"] == "complete"
    client.post("/api/models/Frequency/fit", json=revision(client))
    assert client.get("/api/reviews/" + first["id"]).json()["status"] == "stale"
