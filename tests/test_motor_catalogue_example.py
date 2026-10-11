from __future__ import annotations

import polars as pl
import pytest

from easy_glm.datasets import load_motor_catalogue_example


def test_catalogue_example_has_complete_unique_keys_and_unseen_training_codes() -> None:
    data, vehicles, occupations, postcodes = load_motor_catalogue_example(rows=400)
    train = data.filter(pl.col("traintest") == 1)
    assert train.height == 300
    assert data["Exposure"].min() > 0
    assert data["Claims"].min() >= 0
    assert data["PolicyID"].n_unique() == data.height
    for catalogue, key in (
        (vehicles, "VehicleCode"),
        (occupations, "OccupationCode"),
        (postcodes, "Postcode"),
    ):
        assert catalogue[key].dtype == pl.String
        assert catalogue[key].n_unique() == catalogue.height
        assert data.join(catalogue, on=key, how="left").height == data.height
        assert set(data[key]) <= set(catalogue[key])
        assert set(catalogue[key]) - set(train[key])
    assert vehicles["VehicleCode"][0] == "0001"
    assert occupations["OccupationCode"][0] == "001"
    assert "Power" not in data.columns
    assert "OccupationGroup" not in data.columns
    assert "AreaType" not in data.columns


def test_catalogue_example_is_reproducible_and_detached() -> None:
    first = load_motor_catalogue_example(rows=80, seed=2)
    second = load_motor_catalogue_example(rows=80, seed=2)
    assert all(a.equals(b) for a, b in zip(first, second, strict=True))
    assert not first[0].equals(load_motor_catalogue_example(rows=80, seed=3)[0])
    first[1].drop_in_place("Power")
    assert "Power" in second[1].columns


@pytest.mark.parametrize("rows", [0, 39, 4.5, True, "100"])
def test_catalogue_example_rejects_invalid_size(rows) -> None:
    with pytest.raises(ValueError, match="integer of at least 40"):
        load_motor_catalogue_example(rows=rows)
