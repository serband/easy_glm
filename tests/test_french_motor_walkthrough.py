"""Developer verification for the interactive French motor Python lesson."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import plotly.graph_objects as go

from easy_glm import RateModel

ROOT = Path(__file__).resolve().parents[1]
LESSON = ROOT / "docs" / "examples" / "french_motor_walkthrough.py"
GUIDE = ROOT / "examples" / "pricing_walkthrough.md"
CELL_MARKER = re.compile(r"(?m)^# %% (\d+) — (.+)$")
PYTHON_FENCE = re.compile(r"(?ms)^```python[^\n]*\n(.*?)^```[ \t]*$")


def lesson_cells(source: str) -> list[tuple[int, str, str]]:
    """Return numbered cell source without treating the file as one-shot code."""
    matches = list(CELL_MARKER.finditer(source))
    return [
        (
            int(match.group(1)),
            match.group(2),
            (
                source[match.end() : matches[index + 1].start()]
                if index + 1 < len(matches)
                else source[match.end() :]
            ),
        )
        for index, match in enumerate(matches)
    ]


def markdown_python_blocks(source: str) -> list[str]:
    """Extract the guide's executable Python without reading its companion."""
    return [match.group(1) for match in PYTHON_FENCE.finditer(source)]


def test_walkthrough_executes_as_reviewable_cells_on_full_fixture(
    tmp_path, monkeypatch
):
    matplotlib.use("Agg")
    lesson_directory = tmp_path / "installed-package-lesson"
    lesson_directory.mkdir()
    shutil.copy2(
        ROOT / "tests" / "fixtures" / "french_motor_50k.parquet",
        lesson_directory / "french_motor_50k.parquet",
    )
    monkeypatch.chdir(lesson_directory)
    monkeypatch.delenv("EASY_GLM_FRENCH_MOTOR_DATA", raising=False)
    monkeypatch.setenv("EASY_GLM_LESSON_OUTPUT", str(lesson_directory / "outputs"))
    source = LESSON.read_text(encoding="utf-8")
    cells = lesson_cells(source)

    assert [number for number, _title, _body in cells] == list(range(1, 16))
    assert "def run_all" not in source
    assert "if __name__" not in source
    assert "easy_glm.app" not in source
    assert "easy_glm.desktop" not in source
    assert "input(" not in source
    assert all("locked_holdout" not in body for _n, _t, body in cells[2:11])

    namespace: dict[str, object] = {"__name__": "__main__"}
    for number, title, body in cells:
        exec(compile(body, f"{LESSON.name}:cell-{number}:{title}", "exec"), namespace)

    assert namespace["split_counts"] == {"train": 34_887, "holdout": 15_113}
    assert (
        namespace["settings_roundtrip"].to_dict()
        == namespace["settings_project"].to_dict()
    )
    assert namespace["settings_project"].data.roles.get("Area") is None
    assert namespace["settings_project"].data.roles.get("Region") is None
    assert (
        namespace["binning_examples"]["literal_custom_DrivAge"]
        == namespace["CUSTOM_KNOTS"]["DrivAge"]
    )
    assert len(namespace["binning_examples"]["per_variable_6_bins_VehPower"]) <= 5

    reviewed_main = namespace["reviewed_main"]
    mains_run = namespace["mains_run"]
    pair1_run = namespace["pair1_run"]
    pair2_run = namespace["pair2_run"]
    training_only = namespace["training_only"]
    assert mains_run.spec.to_dict() == reviewed_main.spec.to_dict()
    np.testing.assert_allclose(
        mains_run.predict(training_only),
        reviewed_main.predict(training_only).to_numpy(),
        rtol=1e-8,
    )
    assert set(mains_run.metrics) == {"train"}
    assert set(pair1_run.metrics) == {"train"}
    assert set(pair2_run.metrics) == {"train"}
    assert len(pair1_run.config.pair_stages) == 1
    assert len(pair2_run.config.pair_stages) == 2
    assert len(pair1_run.rate_model.pair_tables) == 1
    assert len(pair2_run.rate_model.pair_tables) == 2
    assert (
        pair1_run.rate_model.to_dict()["pair_tables"][0]
        == pair2_run.rate_model.to_dict()["pair_tables"][0]
    )
    assert (
        mains_run.rate_model.to_dict()["variables"]
        == pair2_run.rate_model.to_dict()["variables"]
    )
    np.testing.assert_allclose(
        namespace["final_unit_prediction"],
        namespace["main_unit_prediction"]
        * namespace["pair1_factor"]
        * namespace["pair2_factor"],
        rtol=1e-12,
    )

    final_metrics = namespace["final_metrics"]
    assert set(final_metrics) == {"train", "holdout"}
    for subset in ("train", "holdout"):
        assert final_metrics[subset]["rows"] > 0
        assert np.isfinite(final_metrics[subset]["mean_deviance"])
        assert np.isfinite(final_metrics[subset]["ae"])

    assert namespace["accepted_run"] is pair2_run
    replayed_prepared = namespace["replayed_prepared"]
    np.testing.assert_array_equal(
        replayed_prepared.filter(namespace["pl"].col("traintest") == 1)[
            "IDpol"
        ].to_numpy(),
        training_only["IDpol"].to_numpy(),
    )

    artifact_paths = namespace["artifact_paths"]
    assert all(path.is_file() for path in artifact_paths.values())
    assert (lesson_directory / "outputs" / "skinny_train_ae_DrivAge.png").is_file()
    assert (lesson_directory / "outputs" / "pair1_relativity_heatmap.png").is_file()
    assert (lesson_directory / "outputs" / "lesson_results.json").is_file()

    # A reviewer may accept the main-effects model without presentation failing.
    namespace["accepted_run"] = mains_run
    exec(
        compile(cells[12][2], f"{LESSON.name}:cell-13:no-pairs", "exec"),
        namespace,
    )
    assert namespace["accepted_pair_heatmap"] is None


def test_markdown_walkthrough_executes_independently_on_full_fixture(
    tmp_path, monkeypatch
):
    """The main guide alone must reproduce the complete staged workflow."""
    matplotlib.use("Agg")
    lesson_directory = tmp_path / "installed-package-guide"
    lesson_directory.mkdir()
    shutil.copy2(
        ROOT / "tests" / "fixtures" / "french_motor_50k.parquet",
        lesson_directory / "french_motor_50k.parquet",
    )
    monkeypatch.chdir(lesson_directory)
    monkeypatch.delenv("EASY_GLM_FRENCH_MOTOR_DATA", raising=False)
    monkeypatch.setenv(
        "EASY_GLM_LESSON_OUTPUT", str(lesson_directory / "guide-outputs")
    )
    monkeypatch.setattr(go.Figure, "show", lambda self, *args, **kwargs: None)

    source = GUIDE.read_text(encoding="utf-8")
    blocks = markdown_python_blocks(source)
    assert blocks
    assert all("french_motor_walkthrough" not in block for block in blocks)

    namespace: dict[str, object] = {"__name__": "__main__"}
    for index, block in enumerate(blocks, start=1):
        exec(
            compile(block, f"{GUIDE.name}:python-block-{index}", "exec"),
            namespace,
        )

    skinny = namespace["skinny"]
    reviewed_main = namespace["reviewed_main"]
    mains_run = namespace["mains_run"]
    pair1_run = namespace["pair1_run"]
    pair2_run = namespace["pair2_run"]
    training_only = namespace["training_only"]
    accepted_run = namespace["accepted_run"]

    assert skinny is not reviewed_main
    assert mains_run.spec.to_dict() == reviewed_main.spec.to_dict()
    np.testing.assert_allclose(
        mains_run.predict(training_only),
        reviewed_main.predict(training_only).to_numpy(),
        rtol=1e-8,
    )
    for run in (mains_run, pair1_run, pair2_run):
        assert set(run.metrics) == {"train"}
        assert run.train_rows == training_only.height
        assert run.holdout_rows == 0

    assert len(mains_run.rate_model.pair_tables) == 0
    assert len(pair1_run.rate_model.pair_tables) == 1
    assert len(pair2_run.rate_model.pair_tables) == 2
    assert (
        mains_run.rate_model.to_dict()["variables"]
        == pair1_run.rate_model.to_dict()["variables"]
        == pair2_run.rate_model.to_dict()["variables"]
    )
    assert (
        pair1_run.rate_model.to_dict()["pair_tables"][0]
        == pair2_run.rate_model.to_dict()["pair_tables"][0]
    )

    artifact_paths = namespace["artifact_paths"]
    assert artifact_paths
    assert all(path.is_file() for path in artifact_paths.values())

    locked_holdout = namespace["locked_holdout"]
    accepted_prediction = accepted_run.predict(locked_holdout)
    restored = RateModel.from_json(artifact_paths["model"])
    np.testing.assert_allclose(
        restored.predict(locked_holdout, exposure_col=None),
        accepted_prediction,
        rtol=1e-12,
    )

    scorer_source = artifact_paths["scorer"].read_text(encoding="utf-8")
    scorer_namespace: dict[str, object] = {"__name__": "guide_frozen_scorer"}
    exec(
        compile(scorer_source, str(artifact_paths["scorer"]), "exec"),
        scorer_namespace,
    )
    np.testing.assert_allclose(
        scorer_namespace["predict"](locked_holdout, exposure_col=None),
        accepted_prediction,
        rtol=1e-12,
    )
