"""Execute the public worked workflows as the reader sees them."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

import easy_glm
from easy_glm.datasets import load_motor_catalogue_example
from easy_glm.pricing.review import ReviewPreview
from easy_glm.pricing.views import DisplayResult
from tests.desktop_onboarding_fixtures import example_frames

ROOT = Path(__file__).resolve().parents[1]
PYTHON_FENCE = re.compile(r"(?ms)^```python[^\n]*\n(.*?)^```[ \t]*$")


def _blocks(page: Path) -> list[str]:
    blocks = [match.group(1) for match in PYTHON_FENCE.finditer(page.read_text())]
    assert blocks, f"{page} has no runnable Python fences"
    return blocks


def _run_page(page: Path) -> dict[str, object]:
    namespace: dict[str, object] = {"__name__": "__main__"}
    for index, block in enumerate(_blocks(page), start=1):
        exec(compile(block, f"{page.name}:block-{index}", "exec"), namespace)
    return namespace


def _quiet_notebook_output(monkeypatch) -> None:
    monkeypatch.setattr(DisplayResult, "show", lambda self: self)
    monkeypatch.setattr(ReviewPreview, "show", lambda self: self)
    monkeypatch.setattr(go.Figure, "show", lambda self, *args, **kwargs: None)


def _small_catalogue_book(*_args, **_kwargs):
    """Keep CI fast while the page itself still calls the public loader literally."""
    return load_motor_catalogue_example(rows=600, seed=42)


def test_stacked_models_page_runs_literal_chain_and_scores_unseen_catalogue_key(
    tmp_path, monkeypatch
):
    """Keep the GLM, two pairs, and three catalogue corrections executable."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(easy_glm, "load_motor_catalogue_example", _small_catalogue_book)
    _quiet_notebook_output(monkeypatch)

    namespace = _run_page(ROOT / "examples" / "stacked_models.md")
    main = namespace["main"]
    first_pair = namespace["first_pair"]
    second_pair = namespace["second_pair"]
    complete = namespace["complete"]
    accepted_vehicle = namespace["accepted_vehicle"]
    saved = namespace["saved"]
    data = namespace["data"]
    unseen_vehicle = namespace["unseen_vehicle"]

    assert len(first_pair._run.rate_model.pair_tables) == 1
    assert len(second_pair._run.rate_model.pair_tables) == 2
    assert (
        first_pair._run.rate_model.to_dict()["variables"]
        == main._run.rate_model.to_dict()["variables"]
    )
    assert [component["name"] for component in complete.summary()["components"]] == [
        "Main effects",
        "Age and region",
        "Age, region and bonus-malus",
        "Vehicle",
        "Occupation",
        "Postcode",
    ]
    vehicle_rates = saved.component("Vehicle").show().table
    assert vehicle_rates.height == 80
    assert saved.component("Occupation").show().table.height == 30
    assert saved.component("Postcode").show().table.height == 120
    assert vehicle_rates.filter(pl.col("VehicleCode") == "0061").height == 1
    accepted_rates = accepted_vehicle.component("Vehicle").show().table["multiplier"]
    assert accepted_rates.min() >= 0.70
    np.testing.assert_allclose(accepted_rates, np.round(accepted_rates, 4))
    unit = saved.predict(data)
    np.testing.assert_allclose(
        saved.predict(data, expected=True), unit * data["Exposure"].to_numpy()
    )
    assert np.isfinite(saved.predict(unseen_vehicle)).all()
    for name in (
        "motor_experience.parquet",
        "vehicle_catalogue.parquet",
        "occupation_catalogue.parquet",
        "postcode_catalogue.parquet",
        "vehicle_factors.csv",
        "occupation_factors.csv",
        "postcode_factors.parquet",
        "motor_rating_tables.xlsx",
        "motor_model.easyglm",
    ):
        assert (tmp_path / name).is_file()


def test_model_variations_page_keeps_fixed_and_refitted_suffixes_distinct(
    tmp_path, monkeypatch
):
    """A table revision must leave its source checkpoint intact and show both policies."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(easy_glm, "load_motor_catalogue_example", _small_catalogue_book)
    _quiet_notebook_output(monkeypatch)

    namespace = _run_page(ROOT / "examples" / "model_variations.md")
    original = namespace["glm_then_tree"]
    fixed = namespace["fixed_suffix"]
    refitted = namespace["refitted_suffix"]
    data = namespace["data"]

    label = namespace["main_rate_row"]["label"]
    original_rate = (
        original.component("GLM main effects")
        .show()
        .table.filter(pl.col("label") == label)["relativity"]
        .item()
    )
    for revised in (fixed, refitted):
        revised_rate = (
            revised.component("GLM main effects")
            .show()
            .table.filter(pl.col("label") == label)["relativity"]
            .item()
        )
        assert revised_rate == 1.05
    assert original_rate != 1.05
    assert not np.allclose(fixed.predict(data), refitted.predict(data))


def test_project_workflow_page_runs_recipe_and_frozen_scorer_parity(
    tmp_path, monkeypatch
):
    """The recipe replay and frozen scorer are separate, usable exports."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(easy_glm, "load_motor_catalogue_example", _small_catalogue_book)
    _quiet_notebook_output(monkeypatch)

    namespace = _run_page(ROOT / "examples" / "project_workflow.md")
    data = namespace["data"]
    saved = namespace["saved"]
    rebuilt = namespace["rebuilt"]
    scorer = namespace["scorer"]

    np.testing.assert_allclose(
        saved.predict(data), scorer["predict"](data), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        saved.predict(data), rebuilt.predict(data), rtol=1e-6, atol=1e-10
    )
    for name in (
        "motor_project.json",
        "stack_project.json",
        "accepted_model.easyglm",
        "accepted_tables.xlsx",
        "rebuild_model.py",
        "score_model.py",
        "rebuilt_model.easyglm",
        "rebuilt_model_rate_tables.xlsx",
    ):
        assert (tmp_path / name).is_file()


def test_model_families_page_runs_without_network_and_keeps_units_separate(
    tmp_path, monkeypatch
):
    """Run all three public family examples against the local Swedish fixture."""
    monkeypatch.chdir(tmp_path)
    swedish = example_frames()["swedish_motorcycle"]
    monkeypatch.setattr(
        easy_glm, "load_swedish_motorcycle_data", lambda: swedish.clone()
    )
    _quiet_notebook_output(monkeypatch)

    namespace = _run_page(ROOT / "examples" / "model_families.md")
    pure_premium = namespace["pure_premium"]
    severity = namespace["severity"]
    occurrence = namespace["occurrence"]
    positive_exposure = namespace["positive_exposure"]
    severity_data = namespace["severity_data"]
    occurrence_data = namespace["occurrence_data"]

    assert pure_premium.summary()["family"] == "tweedie"
    assert severity.summary()["family"] == "gamma"
    assert occurrence.summary()["family"] == "binomial"
    assert severity_data["ClaimNb"].min() > 0
    assert severity_data["ClaimAmount"].min() > 0
    occurrence_prediction = occurrence.predict(occurrence_data)
    assert (occurrence_prediction >= 0).all()
    assert (occurrence_prediction <= 1).all()
    np.testing.assert_allclose(
        occurrence.predict(occurrence_data, expected=True), occurrence_prediction
    )
    assert positive_exposure["Exposure"].min() > 0
    for name in (
        "swedish_tweedie_pure_premium.xlsx",
        "swedish_tweedie_pure_premium.easyglm",
        "swedish_gamma_severity.xlsx",
        "swedish_gamma_severity.easyglm",
        "swedish_binomial_occurrence.xlsx",
        "swedish_binomial_occurrence.easyglm",
    ):
        assert (tmp_path / name).is_file()
