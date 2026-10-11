"""CLI and generated scripts preserve an accepted ordered scorer."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from easy_glm.cli import main
from easy_glm.pricing import PricingModel
from easy_glm.workflow import Project, prepare, run_model, to_scoring_script, to_script
from easy_glm.workflow.project import ModelConfig
from easy_glm.workflow.stack_types import ComponentRecipe


def _project(tmp_path: Path) -> tuple[Path, pl.DataFrame]:
    rng = np.random.default_rng(82)
    count = 90
    vehicle = np.array(["001", "002"])[rng.integers(0, 2, count)]
    region = np.array(["north", "south"])[rng.integers(0, 2, count)]
    response = rng.poisson(
        np.where(vehicle == "001", 1.5, 0.9) * np.where(region == "north", 1.1, 0.95)
    )
    frame = pl.DataFrame(
        {
            "VehicleCode": vehicle,
            "Region": region,
            "Claims": response,
        }
    )
    source = tmp_path / "experience.parquet"
    frame.write_parquet(source)
    master = pl.DataFrame(
        {
            "VehicleCode": ["001", "002", "003"],
            "Power": [80, 120, 160],
        }
    )
    project = Project(name="Stack CLI")
    project.data.source.path = str(source)
    project.data.split.mode = "random"
    project.data.roles = {
        "VehicleCode": "predictor",
        "Region": "predictor",
        "Claims": "target",
    }
    config = ModelConfig(
        family="poisson",
        target="Claims",
        components=[
            ComponentRecipe(
                "trees",
                "Trees",
                "catboost",
                ("Region",),
                {"iterations": 5, "depth": 2, "trials": 0},
            ),
            ComponentRecipe(
                "vehicle",
                "Vehicle",
                "catalogue",
                ("Power",),
                {
                    "key": "VehicleCode",
                    "catalogue": master.to_dict(as_series=False),
                    "catalogue_schema": {
                        name: str(dtype) for name, dtype in master.schema.items()
                    },
                    "iterations": 5,
                    "depth": 2,
                    "trials": 0,
                },
            ),
        ],
    )
    project.models["Mixed"] = config
    project.champion = "Mixed"
    path = tmp_path / "project.json"
    project.to_json(path)
    return path, frame


def _python_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    return env


def test_cli_run_bundle_and_fresh_process_scripts(tmp_path: Path) -> None:
    project_path, frame = _project(tmp_path)
    out = tmp_path / "out"
    assert main(["run", str(project_path), "--model", "Mixed", "--out", str(out)]) == 0
    stem = out / "Stack CLI_Mixed"
    bundle = stem.with_name(f"{stem.name}.easyglm")
    workbook = stem.with_name(f"{stem.name}_rate_tables.xlsx")
    training_script = stem.with_name(f"{stem.name}.py")
    assert bundle.is_file() and workbook.is_file() and training_script.is_file()
    accepted = PricingModel.load(bundle)
    expected = accepted.predict(frame)

    project = Project.from_json(project_path)
    prepared = pl.read_parquet(project.data.source.path)
    from easy_glm.workflow import prepare

    run = run_model(project, prepare(project, prepared), "Mixed")
    np.testing.assert_allclose(run.predict(frame), expected, rtol=1e-5)
    frozen = tmp_path / "frozen.py"
    frozen.write_text(to_scoring_script(run))
    scored = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import json,runpy,polars as pl,sys; ns=runpy.run_path(sys.argv[1]); "
            "data=pl.read_parquet(sys.argv[2]); print(json.dumps(ns['predict'](data).tolist()))",
            str(frozen),
            str(tmp_path / "experience.parquet"),
        ],
        env=_python_env(),
        text=True,
    )
    np.testing.assert_allclose(json.loads(scored), run.predict(frame), rtol=1e-12)

    subprocess.run(
        [sys.executable, str(training_script)],
        cwd=tmp_path,
        env=_python_env(),
        check=True,
    )
    rebuilt = PricingModel.load(tmp_path / bundle.name)
    np.testing.assert_allclose(rebuilt.predict(frame), expected, rtol=1e-5)


def test_cli_stack_report_gate_precedes_output(tmp_path: Path) -> None:
    project_path, _ = _project(tmp_path)
    out = tmp_path / "out"
    assert main(["export", str(project_path), "--report", "--out", str(out)]) == 1
    assert not out.exists()


def test_training_script_pins_catalogue_source(tmp_path: Path) -> None:
    project_path, _ = _project(tmp_path)
    project = Project.from_json(project_path)
    recipe = project.models["Mixed"].components[1]
    master = pl.DataFrame(recipe.settings.pop("catalogue"))
    master_path = tmp_path / "vehicle.csv"
    master.write_csv(master_path)
    recipe.settings.pop("catalogue_schema", None)
    recipe.settings["source_path"] = str(master_path)
    recipe.settings["source_sha256"] = hashlib.sha256(
        master_path.read_bytes()
    ).hexdigest()
    run = run_model(project, prepare(project), "Mixed")
    saved_config = copy.deepcopy(run.config)
    # The current working project is deliberately changed after fitting.  A
    # workflow export must retain the captured preparation/split snapshot.
    project.data.split.seed = 99
    saved_project = project.to_dict()
    master_path.unlink()
    source = to_script(project, "Mixed", run=run, output_prefix="pinned")
    assert "catalogue_schema" in source
    assert str(master_path) not in source
    assert '"seed": 42' in source
    assert '"seed": 99' not in source
    assert run.config == saved_config
    assert project.to_dict() == saved_project
    training_script = tmp_path / "pinned.py"
    training_script.write_text(source)
    subprocess.run(
        [sys.executable, str(training_script)],
        cwd=tmp_path,
        env=_python_env(),
        check=True,
    )
    assert (tmp_path / "pinned.easyglm").is_file()
