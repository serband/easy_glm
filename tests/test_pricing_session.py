from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from easy_glm.pricing import PricingModel, PricingSession


def _book(n: int = 180) -> pl.DataFrame:
    rng = np.random.default_rng(17)
    age = rng.integers(18, 81, n)
    area = rng.integers(1, 5, n)
    exposure = rng.uniform(0.2, 1.0, n)
    rate = np.exp(-2.6 + 0.012 * (age - 45) + 0.08 * (area == 4))
    claims = rng.poisson(exposure * rate)
    return pl.DataFrame(
        {
            "policy": np.arange(n),
            "claims": claims,
            "exposure": exposure,
            "age": age,
            "area": area,
        }
    )


def test_session_bands_categories_and_checkpoint_are_detached(tmp_path) -> None:
    data = _book()
    work = PricingSession(
        data,
        claims="claims",
        exposure="exposure",
        id="policy",
        train_fraction=0.75,
        seed=9,
    )
    assert (
        work.summary().filter(pl.col("column") == "area")["role"].item() == "unassigned"
    )
    assert work.bands(default=8).table["value"].item() == 8
    age_preview = work.bands("age", cuts=[25, 35, 50, 65]).table
    assert age_preview["exposure"].sum() > 0
    area_preview = work.categories("area", levels=["1", "2", "3", "4"]).table
    assert area_preview.height == 5  # four levels plus Other

    basic = work.fit_glm("Age", factors=["age"], alpha=0.001)
    assert isinstance(basic, PricingModel)
    assert set(basic._run.metrics) == {"train"}
    assert basic._run.config is basic._project.models["Age"]
    original_knots = list(basic._run.spec["age"].knots)

    work.bands("age", cuts=[30, 45, 60])
    assert list(basic._run.spec["age"].knots) == original_knots
    revised = basic.refit("Revised age", factors=["age"])
    assert list(revised._run.spec["age"].knots) == [30.0, 45.0, 60.0]
    assert list(basic._run.spec["age"].knots) == original_knots

    np.testing.assert_allclose(
        basic.predict(expected=True),
        basic.predict() * basic._frame("train")["exposure"].to_numpy(),
    )
    assert "holdout" not in basic.summary()

    settings_path = work.save_settings(tmp_path / "pricing.json")
    restored = PricingSession.from_settings(data, settings_path)
    assert restored.settings() == work.settings()

    gamma = PricingSession(data, family="gamma", target="exposure", seed=3)
    gamma_path = gamma.save_settings(tmp_path / "gamma-settings.json")
    restored_gamma = PricingSession.from_settings(data, gamma_path)
    assert restored_gamma._family == "gamma"
    assert restored_gamma._link is None


def test_refit_adds_main_factor_without_mutating_parent() -> None:
    work = PricingSession(
        _book(), claims="claims", exposure="exposure", id="policy", seed=4
    )
    work.bands("age", number=5)
    work.categories("area")
    basic = work.fit_glm("Basic", factors=["age"], alpha=0.001)
    main = basic.refit("Main", add=["area"])

    assert basic._run.config.predictors == ["age"]
    assert main._run.config.predictors == ["age", "area"]
    assert basic.history[-1]["action"] == "fit_glm"
    assert main.history[-1]["action"] == "refit"


def test_interaction_uses_unassigned_parent_and_reports_validation_evidence() -> None:
    rng = np.random.default_rng(31)
    n = 400
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    q = rng.normal(size=n)
    exposure = rng.uniform(0.25, 1.0, n)
    claims = rng.poisson(exposure * np.exp(-1.0 + 0.25 * x + 0.2 * z))
    data = pl.DataFrame(
        {
            "id": np.arange(n),
            "claims": claims,
            "exposure": exposure,
            "x": x,
            "z": z,
            "q": q,
        }
    )
    work = PricingSession(data, claims="claims", exposure="exposure", id="id", seed=3)
    work.bands("x", cuts=[-1, 0, 1])
    work.bands("z", cuts=[-1, 0, 1])
    work.bands("q", cuts=[-1, 0, 1])
    main = work.fit_glm("Main", factors=["x"], alpha=0.001)
    pair = main.fit_interaction(
        "x",
        "z",
        name="Pair",
        trials=1,
        prefix_trials=1,
        time_limit_minutes=0.5,
    )

    assert pair._run.config.predictors == ["x"]
    assert pair._run.config.pair_time_limit_minutes == 0.5
    assert main._run.config.pair_time_limit_minutes == 15.0
    assert pair._project.data.roles.get("z") is None
    evidence = pair.summary()["pair_stages"][0]
    assert evidence["parents"] == ("x", "z")
    assert evidence["cv_evidence"] == "current"
    assert "table_validation_loss" in evidence
    assert "teacher_validation_loss" in evidence
    assert "table_approximation_loss" in evidence
    assert evidence["cells"] > 0

    first_table = pair._run.rate_model.to_dict()["pair_tables"][0]
    second = pair.fit_interaction(
        "z", "q", name="Second pair", trials=1, prefix_trials=1
    )
    frozen_table = second._run.rate_model.to_dict()["pair_tables"][0]
    assert second._run.config.pair_time_limit_minutes == 0.5
    assert frozen_table["axes"] == first_table["axes"]
    assert frozen_table["cells"] == first_table["cells"]
    assert second._frozen_stages == []
    assert second._run.pair_stages[0].reused is True
    assert second._run.pair_stages[0].status != "pricing_adjustment"


def test_random_split_is_stable_by_id_and_settings_validate_data(tmp_path) -> None:
    data = _book(40)
    work = PricingSession(
        data,
        claims="claims",
        exposure="exposure",
        id="policy",
        train_fraction=0.7,
        seed=8,
    )
    split_column = work._project.data.split.column
    membership = work._data.select("policy", split_column).sort("policy")
    path = work.save_settings(tmp_path / "settings.json")

    restored = PricingSession.from_settings(data.reverse(), path)
    restored_membership = restored._data.select("policy", split_column).sort("policy")
    assert restored_membership.equals(membership)

    changed = data.with_columns((pl.col("claims") + 1).alias("claims"))
    with pytest.raises(ValueError, match="does not match"):
        PricingSession.from_settings(changed, path)


def test_setup_rejects_empty_split_and_missing_random_split_ids() -> None:
    data = _book(20)
    with pytest.raises(ValueError, match="missing values"):
        PricingSession(
            data.with_columns(
                pl.when(pl.col("policy") == 0)
                .then(None)
                .otherwise(pl.col("policy"))
                .alias("policy")
            ),
            claims="claims",
            exposure="exposure",
            id="policy",
        )
    with pytest.raises(ValueError, match="holdout value matches no rows"):
        PricingSession(
            data.with_columns(pl.lit(1).alias("fixed_split")),
            claims="claims",
            exposure="exposure",
            split="fixed_split",
        )
    with pytest.raises(ValueError, match="training row and one holdout row"):
        PricingSession(
            data.drop("policy"),
            claims="claims",
            exposure="exposure",
            train_fraction=0.99,
            seed=0,
        )
