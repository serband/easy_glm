"""Developer verification for the interactive French motor Python lesson."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LESSON = ROOT / "docs" / "examples" / "french_motor_walkthrough.py"
CELL_MARKER = re.compile(r"(?m)^# %% (\d+) — (.+)$")


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


def test_walkthrough_executes_as_reviewable_cells_on_full_fixture(
    tmp_path, monkeypatch
):
    matplotlib.use("Agg")
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("EASY_GLM_LESSON_OUTPUT", str(tmp_path / "outputs"))
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
    assert (tmp_path / "outputs" / "skinny_train_ae_DrivAge.png").is_file()
    assert (tmp_path / "outputs" / "pair1_relativity_heatmap.png").is_file()
    assert (tmp_path / "outputs" / "lesson_results.json").is_file()

    # A reviewer may accept the main-effects model without presentation failing.
    namespace["accepted_run"] = mains_run
    exec(
        compile(cells[12][2], f"{LESSON.name}:cell-13:no-pairs", "exec"),
        namespace,
    )
    assert namespace["accepted_pair_heatmap"] is None
