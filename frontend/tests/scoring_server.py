"""Serve a saved pricing stack and scoring files for browser checks.

The fixture deliberately restores a real immutable checkpoint rather than having
Playwright fit a model.  It contains a main GLM, an accepted legacy pair table,
and vehicle and occupation catalogue multipliers.  The scoring frames contain no
claim outcome or train/holdout split: they are new business inputs.
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path

import polars as pl
import uvicorn

from easy_glm import PricingSession, load_motor_catalogue_example
from easy_glm.desktop.fit_worker import result_for, write_json
from easy_glm.desktop.jobs import model_key
from easy_glm.desktop.server import create_app
from easy_glm.workflow.prep import prepare

PORT = int(os.environ.get("EASYGLM_SCORING_PORT", "8891"))


def new_business_frames(data: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return an eligible and deliberately incomplete new-business extract."""
    required_inputs = [
        "PolicyID",
        "Exposure",
        "DriverAge",
        "BonusMalus",
        "Region",
        "VehicleCode",
        "OccupationCode",
    ]
    renewal = data.select(required_inputs).tail(4)
    return {
        "Renewal extract": renewal,
        "Missing rating factor": renewal.drop("Region"),
    }


def saved_stack() -> tuple[object, pl.DataFrame, object]:
    """Fit the small deterministic accepted stack used by the scoring screens."""
    data, vehicles, occupations, _ = load_motor_catalogue_example(rows=320, seed=887)
    session = PricingSession(
        data,
        family="poisson",
        claims="Claims",
        exposure="Exposure",
        id="PolicyID",
        split="traintest",
    )
    main = session.fit_glm(
        "Main effects",
        factors=["DriverAge", "BonusMalus", "Region"],
        alpha=0.02,
    )
    pair = main.fit_interaction(
        "DriverAge",
        "Region",
        name="Age and region correction",
        trials=1,
        prefix_trials=1,
    )
    vehicle = pair.fit_catalogue(
        "Vehicle file",
        catalogue=vehicles,
        key="VehicleCode",
        factors=["Make", "Power"],
        tune=False,
        iterations=8,
        depth=2,
        thread_count=1,
    )
    occupation = vehicle.fit_catalogue(
        "Occupation file",
        catalogue=occupations,
        key="OccupationCode",
        factors=["OccupationGroup", "PhysicalWork"],
        tune=False,
        iterations=8,
        depth=2,
        thread_count=1,
    )
    return occupation._project, data, occupation._run


def write_restored_fit(
    folder: Path,
    project: object,
    raw: pl.DataFrame,
    run: object,
) -> None:
    """Write the application-owned artefacts accepted by ``FitJobs.restore``."""
    name = run.name
    source = folder / "fits" / "accepted-stack"
    source.mkdir(parents=True)
    with (source / "fit.pkl").open("wb") as handle:
        pickle.dump(run, handle, protocol=5)
    raw.write_parquet(source / "raw.parquet")
    project.to_json(source / "project.json")
    result = result_for(project, prepare(project, raw), run)
    write_json(source / "result.json", result)
    record = {
        "id": "accepted-stack",
        "name": name,
        "status": "complete",
        "message": "Saved accepted stack is ready.",
        "elapsed": 0.0,
        "key": model_key(project, name),
        "applicable": True,
        "artifact_available": True,
    }
    (folder / "jobs.json").write_text(json.dumps({name: record}), encoding="utf-8")


def main() -> None:
    project, raw, run = saved_stack()
    with tempfile.TemporaryDirectory(prefix="easyglm_scoring_browser_") as temporary:
        restore_folder = Path(temporary)
        write_restored_fit(restore_folder, project, raw, run)
        uvicorn.run(
            create_app(
                project,
                raw,
                port=PORT,
                restore_folder=restore_folder,
                scoring_frames=new_business_frames(raw),
            ),
            host="127.0.0.1",
            port=PORT,
            log_level="warning",
        )


if __name__ == "__main__":
    main()
