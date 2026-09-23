from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from easy_glm.pricing import PricingModel, PricingSession


def _split_book(n: int = 120) -> pl.DataFrame:
    rng = np.random.default_rng(22)
    x = rng.normal(size=n)
    exposure = rng.uniform(0.3, 1.0, n)
    claims = rng.poisson(exposure * np.exp(-1.2 + 0.2 * x))
    return pl.DataFrame(
        {
            "id": np.arange(n),
            "claims": claims,
            "exposure": exposure,
            "x": x,
            "ignored": rng.normal(size=n),
            "traintest": np.where(np.arange(n) % 4 == 0, 0, 1),
        }
    )


def test_setup_protects_control_columns_and_holdout_is_explicit() -> None:
    data = _split_book()
    work = PricingSession(
        data,
        claims="claims",
        exposure="exposure",
        id="id",
        split="traintest",
        ignored=["ignored"],
    )
    work.bands("x", cuts=[-1, 0, 1])
    model = work.fit_glm("Frequency", factors=["x"], alpha=0.001)

    assert set(model._run.metrics) == {"train"}
    holdout = model.validate_holdout()
    assert holdout["subset"].to_list() == ["holdout"]
    assert holdout["rows"].item() == 30
    with pytest.raises(ValueError, match="protected"):
        model.refit("Bad", add=["claims"])
    with pytest.raises(ValueError, match="protected"):
        model.fit_interaction("x", "exposure", name="Bad pair")


def test_compare_defaults_to_training_and_cv_refits_folds() -> None:
    work = PricingSession(
        _split_book(),
        claims="claims",
        exposure="exposure",
        id="id",
        split="traintest",
    )
    work.bands("x", cuts=[-1, 0, 1])
    first = work.fit_glm("First", factors=["x"], alpha=0.001)
    second = first.refit("Second", factors=["x"])

    ordinary = second.compare(first)
    assert ordinary["subset"].to_list() == ["train", "train"]
    assert "mean_deviance" in ordinary.columns
    assert ordinary.columns.index("mean_deviance") < ordinary.columns.index("gini")

    cross_validated = second.compare(first, cv=True)
    assert cross_validated["subset"].to_list() == [
        "cross_validation",
        "cross_validation",
    ]
    assert cross_validated["folds"].to_list() == [5, 5]
    assert second.evidence["validation"]["cross_validation"]["folds"] == 5
    assert first.evidence["validation"]["cross_validation"]["folds"] == 5


def test_explicit_holdout_evidence_is_recorded_for_each_model(tmp_path) -> None:
    work = PricingSession(
        _split_book(),
        claims="claims",
        exposure="exposure",
        id="id",
        split="traintest",
    )
    work.bands("x", cuts=[-1, 0, 1])
    first = work.fit_glm("First", factors=["x"], alpha=0.001)
    second = first.refit("Second", factors=["x"])

    result = second.validate_holdout(compare_with=first)
    assert set(second._run.metrics) == {"train"}
    assert set(first._run.metrics) == {"train"}
    assert second.evidence["validation"]["holdout"]["model"] == "Second"
    assert first.evidence["validation"]["holdout"]["model"] == "First"
    assert result["rows"].to_list() == [30, 30]

    for model in (first, second):
        saved = json.loads(model.save(tmp_path / f"{model.name}.json").read_text())
        assert saved["evidence"]["validation"]["holdout"]["rows"] == 30


def test_setup_rejects_ambiguous_or_missing_columns() -> None:
    data = _split_book()
    with pytest.raises(ValueError, match="different columns"):
        PricingSession(
            data,
            target="claims",
            weight="exposure",
            id="claims",
            split="traintest",
        )
    with pytest.raises(KeyError, match="not in the data"):
        PricingSession(data, target="missing")


def test_compare_rejects_different_policy_rows() -> None:
    first_work = PricingSession(
        _split_book(), claims="claims", exposure="exposure", id="id", split="traintest"
    )
    second_work = PricingSession(
        _split_book().reverse(),
        claims="claims",
        exposure="exposure",
        id="id",
        split="traintest",
    )
    for work in (first_work, second_work):
        work.bands("x", cuts=[-1, 0, 1])
    first = first_work.fit_glm("First", factors=["x"], alpha=0.001)
    second = second_work.fit_glm("Second", factors=["x"], alpha=0.001)
    with pytest.raises(ValueError, match="same ordered policy rows"):
        first.compare(second)
    with pytest.raises(ValueError, match="same ordered policy rows"):
        first.validate_holdout(compare_with=[second])


def test_loaded_model_without_data_has_clear_analysis_error(tmp_path) -> None:
    work = PricingSession(
        _split_book(), claims="claims", exposure="exposure", id="id", split="traintest"
    )
    work.bands("x", cuts=[-1, 0, 1])
    model = work.fit_glm("Frequency", factors=["x"], alpha=0.001)
    loaded = PricingModel.load(model.save(tmp_path / "model.json"))

    assert loaded.summary()["training"]
    with pytest.raises(ValueError, match=r"load\(path, data="):
        loaded.predict()
    with pytest.raises(ValueError, match=r"load\(path, data="):
        loaded.refit("Refit")
    with pytest.raises(ValueError, match=r"load\(path, data="):
        loaded.edit_rates("Edited").preview()
