"""Fit progress uses a bounded pipe without affecting durable fit results."""

from __future__ import annotations

import io
import json
import os
import pickle
import subprocess
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from easy_glm.desktop import fit_worker
from easy_glm.desktop.fit_progress import ProgressWriter, progress_packets
from easy_glm.desktop.jobs import FitJobs
from easy_glm.workflow.project import ModelConfig, Project


def _project_and_raw() -> tuple[Project, pl.DataFrame]:
    project = Project()
    project.data.roles = {"y": "target"}
    project.data.split.mode = "random"
    project.models["m"] = ModelConfig(target="y")
    return project, pl.DataFrame({"y": [0.0, 1.0, 0.0, 2.0]})


def _wait(jobs: FitJobs, project: Project, *, timeout: float = 10) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        packet = jobs.status(project)["m"]
        if packet["status"] not in ("queued", "running"):
            if jobs.thread is not None:
                jobs.thread.join(timeout=5)
            return packet
        time.sleep(0.01)
    pytest.fail("fit worker did not finish")


def _fake_worker(monkeypatch, body: str) -> None:
    import easy_glm.desktop.jobs as jobs_module

    real_popen = subprocess.Popen

    def launch(command, **kwargs):
        folder = command[3]
        return real_popen(
            [sys.executable, "-c", body, folder],
            **kwargs,
        )

    monkeypatch.setattr(jobs_module.subprocess, "Popen", launch)


def test_progress_writer_frames_concurrent_messages_and_ignores_closed_pipe():
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "rb")
    writer = ProgressWriter(os.fdopen(write_fd, "wb", buffering=0))

    threads = [
        threading.Thread(
            target=lambda start=start: [
                writer({"message": f"{start + index}"}) for index in range(250)
            ]
        )
        for start in (0, 250, 500, 750)
    ]
    for thread in threads:
        thread.start()
    received: list[dict] = []
    drain = threading.Thread(target=lambda: received.extend(progress_packets(reader)))
    drain.start()
    for thread in threads:
        thread.join()
    writer.close()
    drain.join(timeout=5)

    assert len(received) == 1_000
    assert {packet["message"] for packet in received} == {
        str(index) for index in range(1_000)
    }
    writer("after close")  # Best effort means a closed transport is harmless.


def test_progress_reader_discards_malformed_non_dict_and_oversized_frames():
    payload = (
        b"not-json\n"
        + b"[1,2]\n"
        + json.dumps({"message": "x" * 80}).encode()
        + b"\n"
        + b'{"message":"kept"}\n'
    )
    assert list(progress_packets(io.BytesIO(payload), maximum_bytes=40)) == [
        {"message": "kept"}
    ]
    output = io.BytesIO()
    writer = ProgressWriter(output, maximum_bytes=40)
    writer("x" * 80)
    writer("kept")
    assert output.getvalue() == b'{"message":"kept"}\n'


def test_worker_reserves_progress_stdout_and_routes_python_and_native_output():
    code = """
import os
from easy_glm.desktop.fit_progress import reserve_progress_stdout
progress = reserve_progress_stdout()
print('python noise', flush=True)
os.write(1, b'native noise\\n')
progress({'message': 'live'})
progress.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": "src"},
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        check=True,
    )
    assert completed.stdout == b'{"message":"live"}\n'
    assert b"python noise" in completed.stderr
    assert b"native noise" in completed.stderr


def test_actual_worker_main_never_writes_progress_files_and_pipe_failure_is_free(
    tmp_path,
):
    project, raw = _project_and_raw()
    project.to_json(tmp_path / "project.json")
    raw.write_parquet(tmp_path / "raw.parquet")
    code = r"""
import pathlib, sys
from easy_glm.desktop import fit_worker

original_write_text = pathlib.Path.write_text
def guarded_write_text(path, *args, **kwargs):
    if path.name.startswith('progress'):
        raise AssertionError('progress must not use files')
    return original_write_text(path, *args, **kwargs)
pathlib.Path.write_text = guarded_write_text

def stub_fit(project, raw, name, progress, *args, **kwargs):
    progress('first packet')
    progress({'message': 'stage packet', 'stage_number': 2})
    return {'ok': True}
fit_worker.fit_result = stub_fit
fit_worker.main()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path), "m"],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": "src"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None and process.stderr is not None
    process.stdout.close()  # Simulate a parent that lost the progress transport.
    stderr = process.stderr.read()
    assert process.wait(timeout=10) == 0, stderr.decode(errors="replace")
    assert json.loads((tmp_path / "result.json").read_text()) == {"ok": True}
    assert not list(tmp_path.glob("progress*"))


def test_actual_worker_main_treats_result_write_failure_as_fatal(tmp_path):
    project, raw = _project_and_raw()
    project.to_json(tmp_path / "project.json")
    raw.write_parquet(tmp_path / "raw.parquet")
    (tmp_path / "result.tmp").mkdir()
    code = r"""
import sys
from easy_glm.desktop import fit_worker
fit_worker.fit_result = lambda *args, **kwargs: {'ok': True}
fit_worker.main()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), "m"],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode != 0
    assert not (tmp_path / "result.json").exists()


def test_jobs_drains_live_stage_and_burst_without_progress_files(monkeypatch):
    body = r"""
import json, pathlib, sys, time
folder = pathlib.Path(sys.argv[1])
print(json.dumps({'message': 'Pair stage 1/2: fold 1/5', 'stage_number': 2}), flush=True)
time.sleep(.25)
for index in range(12000):
    print(json.dumps({'message': 'burst-' + str(index)}))
sys.stdout.flush()
(folder / 'result.json').write_text(json.dumps({'ok': True}))
"""
    _fake_worker(monkeypatch, body)
    project, raw = _project_and_raw()
    jobs = FitJobs()
    try:
        jobs.start(project, raw, "m")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = jobs.status(project)["m"]
            if status.get("progress", {}).get("stage_number") == 2:
                break
            time.sleep(0.01)
        else:
            pytest.fail("live stage progress was not observed")
        complete = _wait(jobs, project)
        assert complete["status"] == "complete"
        assert jobs.result(project, "m") == {"ok": True}
        artifact = Path(jobs.folder.name) / complete["id"]
        assert not list(artifact.glob("progress*"))
    finally:
        jobs.close()


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            "import json,pathlib,sys; pathlib.Path(sys.argv[1], 'result.json').write_text(json.dumps({'error':'real model error'}))",
            "real model error",
        ),
        ("pass", "result.json"),
        (
            "import json,pathlib,sys; pathlib.Path(sys.argv[1], 'result.json').write_text(json.dumps({'ok':True})); raise SystemExit(7)",
            "exited with code 7",
        ),
    ],
)
def test_failed_missing_or_nonzero_worker_is_never_published(
    monkeypatch, body, expected
):
    _fake_worker(monkeypatch, body)
    project, raw = _project_and_raw()
    jobs = FitJobs()
    try:
        jobs.start(project, raw, "m")
        status = _wait(jobs, project)
        assert status["status"] == "failed"
        assert expected in status["message"]
        with pytest.raises(ValueError, match="No completed fit"):
            jobs.result(project, "m")
    finally:
        jobs.close()


def test_cancel_and_stale_messages_ignore_late_progress(monkeypatch):
    body = r"""
import json, sys, time
for index in range(100):
    print(json.dumps({'message': 'late-' + str(index), 'stage_number': 2}), flush=True)
    time.sleep(.02)
time.sleep(30)
"""
    _fake_worker(monkeypatch, body)
    project, raw = _project_and_raw()
    jobs = FitJobs()
    try:
        jobs.start(project, raw, "m")
        deadline = time.monotonic() + 5
        while jobs.status(project)["m"]["status"] == "queued":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        jobs.cancel("m")
        assert _wait(jobs, project)["message"] == "Fit cancelled"

        jobs.start(project, raw, "m")
        changed = deepcopy(project)
        changed.models["m"].penalty.alpha = 0.02
        jobs.invalidate(changed)
        status = _wait(jobs, changed)
        assert status["status"] == "stale"
        assert status["message"] == "Settings changed. Fit this model again."
    finally:
        jobs.close()


def test_optional_cache_write_failure_is_nonfatal_and_preserves_existing(
    monkeypatch, tmp_path
):
    cache = tmp_path / "cache.pkl"
    cache.write_bytes(b"existing")
    monkeypatch.setattr(
        fit_worker.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("denied"))
    )
    fit_worker._write_optional_pickle(cache, {"new": True})
    assert cache.read_bytes() == b"existing"
    assert not cache.with_suffix(".tmp").exists()


def test_cache_write_failure_does_not_hide_success_or_original_fit_error(
    monkeypatch, tmp_path
):
    class StubProject:
        def __init__(self, with_pairs: bool):
            self.models = {
                "m": SimpleNamespace(pair_stages=[object()] if with_pairs else [])
            }

        def validate(self, name, columns):
            return []

    monkeypatch.setattr("easy_glm.workflow.prep.prepare", lambda project, raw: raw)
    monkeypatch.setattr(
        fit_worker.os,
        "replace",
        lambda *args: (_ for _ in ()).throw(OSError("cache denied")),
    )
    monkeypatch.setattr(fit_worker, "result_for", lambda *args: {"ok": True})
    monkeypatch.setattr(
        "easy_glm.workflow.run.run_model", lambda *args, **kwargs: object()
    )
    result = fit_worker.fit_result(
        StubProject(False),
        pl.DataFrame({"y": [1]}),
        "m",
        lambda message: None,
        main_cache_path=tmp_path / "main.pkl",
    )
    assert result["ok"] is True

    def fail(*args, **kwargs):
        raise RuntimeError("original fitting error")

    monkeypatch.setattr("easy_glm.workflow.run.run_model", fail)
    with pytest.raises(RuntimeError, match="original fitting error"):
        fit_worker.fit_result(
            StubProject(True),
            pl.DataFrame({"y": [1]}),
            "m",
            lambda message: None,
            pair_cache_path=tmp_path / "pair.pkl",
        )


def test_required_fit_artifact_write_failure_is_fatal(monkeypatch, tmp_path):
    class StubProject:
        models = {"m": SimpleNamespace(pair_stages=[])}

        def validate(self, name, columns):
            return []

    monkeypatch.setattr("easy_glm.workflow.prep.prepare", lambda project, raw: raw)
    monkeypatch.setattr(
        "easy_glm.workflow.run.run_model", lambda *args, **kwargs: object()
    )
    (tmp_path / "fit.pkl").mkdir()
    with pytest.raises(OSError):
        fit_worker.fit_result(
            StubProject(),
            pl.DataFrame({"y": [1]}),
            "m",
            lambda message: None,
            artifact=tmp_path,
        )


def test_optional_cache_normal_replacement(tmp_path):
    cache = tmp_path / "cache.pkl"
    fit_worker._write_optional_pickle(cache, {"format": 9})
    with cache.open("rb") as handle:
        assert pickle.load(handle) == {"format": 9}
