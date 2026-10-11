"""Public pricing checkpoints exercise the accepted complete stack scorer."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from easy_glm import PricingModel, PricingSession, load_motor_catalogue_example


def _session():
    data, vehicles, occupations, _ = load_motor_catalogue_example(rows=600, seed=19)
    session = PricingSession(
        data,
        family="poisson",
        claims="Claims",
        exposure="Exposure",
        id="PolicyID",
        split="traintest",
    )
    return session, data, vehicles, occupations


def test_legacy_glm_then_catboost_then_glm_keeps_named_checkpoints() -> None:
    session, data, _, _ = _session()
    main = session.fit_glm("Main", factors=["DriverAge"], alpha=0.01)
    first = main.predict(data)
    trees = main.fit_catboost(
        "Trees", factors=["BonusMalus"], iterations=12, depth=2, trials=0
    )
    corrected = trees.fit_glm("Correction", factors=["Region"], alpha=0.01)
    assert [c["kind"] for c in corrected.summary()["components"]] == [
        "glm",
        "catboost",
        "glm",
    ]
    np.testing.assert_allclose(main.predict(data), first, rtol=0, atol=0)
    assert np.all(np.isfinite(corrected.predict(data)))
    np.testing.assert_allclose(
        corrected.predict(data, expected=True),
        corrected.predict(data) * data["Exposure"].to_numpy(),
    )


def test_catalogue_review_export_and_bundle_round_trip(tmp_path) -> None:
    session, data, vehicles, _ = _session()
    main = session.fit_glm("Main", factors=["DriverAge"], alpha=0.01)
    vehicle = main.fit_catalogue(
        "Vehicle",
        catalogue=vehicles,
        key="VehicleCode",
        factors=["Make", "Power"],
        iterations=12,
        depth=2,
        trials=0,
    )
    original = vehicle.predict(data)
    assert vehicle.component("Vehicle").show().table.height == vehicles.height
    assert vehicle.ae("Power").table.height > 1
    reviewed = (
        vehicle.edit_rates("Reviewed vehicle")
        .cap_component("Vehicle", lower=0.7, upper=1.5)
        .round_component("Vehicle", decimals=3)
        .apply(refit_later_components=False)
    )
    np.testing.assert_allclose(vehicle.predict(data), original, rtol=0, atol=0)
    lookup = reviewed.export_lookup("Vehicle", tmp_path / "vehicle.csv")
    assert lookup.is_file()
    bundle = reviewed.save(tmp_path / "reviewed.easyglm")
    loaded = PricingModel.load(bundle)
    np.testing.assert_allclose(loaded.predict(data), reviewed.predict(data))
    validation = reviewed.compare(main, cv=True)
    assert validation.height == 2
    appended = reviewed.fit_glm("Region correction", factors=["Region"], alpha=0.01)
    assert appended._run.config.components[1].amendments
    assert appended.compare(main, cv=True).height == 2


def test_following_glm_can_use_pinned_catalogue_attribute() -> None:
    session, data, vehicles, _ = _session()
    main = session.fit_glm("Main", factors=["DriverAge"], alpha=0.01)
    vehicle = main.fit_catalogue(
        "Vehicle",
        catalogue=vehicles,
        key="VehicleCode",
        factors=["Make", "Power"],
        iterations=12,
        depth=2,
        trials=0,
    )
    adjusted = vehicle.fit_glm("Power correction", factors=["Power"], alpha=0.01)
    assert np.all(np.isfinite(adjusted.predict(data)))
    assert adjusted.ae("Power").table.height > 1


def test_catboost_can_start_before_glm() -> None:
    session, data, _, _ = _session()
    base = session.fit_catboost(
        "Tree base", factors=["DriverAge"], iterations=12, depth=2, trials=0
    )
    corrected = base.fit_glm("Region correction", factors=["Region"], alpha=0.01)
    assert [item["kind"] for item in corrected.summary()["components"]] == [
        "baseline",
        "catboost",
        "glm",
    ]
    assert np.all(np.isfinite(corrected.predict(data)))


def test_pricing_session_catboost_never_fits_holdout_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both Optuna folds and a manual full fit receive training rows only."""
    from easy_glm.workflow import stack

    class FakeModel:
        def predict(self, raw, *, prediction_type):
            assert prediction_type == "RawFormulaVal"
            return np.zeros(len(raw))

    calls: list[int] = []

    def spy(raw, target, baseline, **_kwargs):
        values = np.asarray(raw[:, 0], dtype=float)
        assert np.all(values < 1_000)
        assert np.all(np.asarray(target, dtype=float) < 1_000)
        assert len(values) == len(target) == len(baseline)
        calls.append(len(values))
        return type("Teacher", (), {"model": FakeModel()})()

    monkeypatch.setattr(stack, "fit_catboost_raw", spy)
    data = pl.DataFrame(
        {
            "PolicyID": list(range(12)),
            "Claims": [1.0] * 8 + [9_999.0] * 4,
            "Exposure": [1.0] * 12,
            "x": list(range(8)) + [9_999] * 4,
            "traintest": [1] * 8 + [0] * 4,
        }
    )
    session = PricingSession(
        data,
        family="poisson",
        claims="Claims",
        exposure="Exposure",
        id="PolicyID",
        split="traintest",
    )
    session.fit_catboost(
        "Automatic",
        factors=["x"],
        tune=True,
        tuning_budget="trials",
        trials=1,
        cv=2,
    )
    session.fit_catboost("Manual", factors=["x"], tune=False)
    assert calls.count(4) == 2  # two training-only Optuna folds
    assert calls[-1] == 8  # the manual full fit is the whole training partition


def test_standalone_catalogue_file_contains_complete_unit_prediction(tmp_path) -> None:
    session, data, vehicles, _ = _session()
    vehicle = session.fit_catalogue(
        "Vehicle",
        catalogue=vehicles,
        key="VehicleCode",
        factors=["Make", "Power"],
        iterations=12,
        depth=2,
        trials=0,
    )
    assert [item["kind"] for item in vehicle.summary()["components"]] == ["catalogue"]
    lookup = vehicle.component("Vehicle").show().table
    scored = vehicle.predict(data.head(10))
    expected = (
        data.head(10)
        .join(lookup.select("VehicleCode", "multiplier"), on="VehicleCode", how="left")[
            "multiplier"
        ]
        .to_numpy()
    )
    np.testing.assert_allclose(scored, expected)
    assert vehicle.export_lookup("Vehicle", tmp_path / "vehicle.parquet").is_file()
    assert vehicle.component("Vehicle").unit_label == "unit prediction"


def test_stack_rate_edit_is_component_qualified_and_refit_factor_change_refused() -> (
    None
):
    session, data, vehicles, _ = _session()
    main = session.fit_glm("Main", factors=["DriverAge"], alpha=0.01)
    vehicle = main.fit_catalogue(
        "Vehicle",
        catalogue=vehicles,
        key="VehicleCode",
        factors=["Make", "Power"],
        iterations=12,
        depth=2,
        trials=0,
    )
    first = vehicle.component("Main").show().table.row(0, named=True)
    review = vehicle.edit_rates("Edited main").set_relativity(
        "DriverAge",
        lower=first["from"],
        upper=first["to"],
        value=1.1,
        component="Main",
    )
    assert review.preview().changes.height == 1
    edited = review.apply(refit_later_components=False)
    assert np.all(np.isfinite(edited.predict(data)))
    try:
        edited.refit("Ambiguous", factors=["Region"])
    except ValueError as exc:
        assert "several named components" in str(exc)
    else:
        raise AssertionError("A factor change must select a component")


def test_legacy_main_and_two_pair_stages_stay_visible_after_catalogue() -> None:
    session, data, vehicles, _ = _session()
    main = session.fit_glm("Main", factors=["DriverAge", "Region"], alpha=0.01)
    first = main.fit_interaction(
        "DriverAge", "Region", name="First interaction", trials=1, prefix_trials=1
    )
    second = first.fit_interaction(
        "BonusMalus", "Region", name="Two interactions", trials=1, prefix_trials=1
    )
    vehicle = second.fit_catalogue(
        "Vehicle",
        catalogue=vehicles,
        key="VehicleCode",
        factors=["Make", "Power"],
        iterations=12,
        depth=2,
        trials=0,
    )
    assert [item["name"] for item in vehicle.summary()["components"]] == [
        "Main",
        "First interaction",
        "Two interactions",
        "Vehicle",
    ]
    assert (
        vehicle.component("Main")
        .show()
        .table.equals(main.component("Main").show().table)
    )
    assert (
        vehicle.component("First interaction")
        .show()
        .table.equals(first.component("First interaction").show().table)
    )
    assert (
        vehicle.component("Two interactions")
        .show()
        .table.equals(second.component("Two interactions").show().table)
    )
    assert vehicle.component("Vehicle").coverage(data)["rows"] == data.height


def test_comparison_uses_shared_folds_across_session_seeds() -> None:
    data, _, _, _ = load_motor_catalogue_example(rows=400, seed=13)
    first_session = PricingSession(
        data,
        claims="Claims",
        exposure="Exposure",
        id="PolicyID",
        split="traintest",
        seed=2,
    )
    second_session = PricingSession(
        data,
        claims="Claims",
        exposure="Exposure",
        id="PolicyID",
        split="traintest",
        seed=99,
    )
    first = first_session.fit_glm("First", factors=["DriverAge"], alpha=0.01)
    second = second_session.fit_glm("Second", factors=["DriverAge"], alpha=0.01)
    compared = first.compare(second, cv=True)
    np.testing.assert_allclose(
        compared["mean_deviance"].to_numpy()[0],
        compared["mean_deviance"].to_numpy()[1],
        rtol=0,
        atol=1e-12,
    )


def test_stack_rebalance_fails_clearly() -> None:
    session, _, _, _ = _session()
    stacked = session.fit_catboost(
        "Trees", factors=["DriverAge"], iterations=12, depth=2, trials=0
    )
    with pytest.raises(ValueError, match="Stack-level rebalancing is not supported"):
        stacked.edit_rates("Reviewed").rebalance()
