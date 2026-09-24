"""Small, deterministic datasets for the interactive modelling examples."""

from __future__ import annotations

import numbers

import numpy as np
import polars as pl


def load_motor_catalogue_example(
    rows: int = 6_000, *, seed: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return synthetic motor experience and complete specialist catalogues.

    Returns ``(data, vehicles, occupations, postcodes)``. These are invented
    examples, not market data or recommended rating factors. Nothing is read
    from the network or written to disk.

    ``Claims`` contains claim counts and ``Exposure`` contains years. The
    ``traintest`` column is 1 for training and 0 for holdout. The experience
    contains lookup keys; specialist attributes live in their catalogues.
    Each catalogue includes keys with no training experience. Identifiers
    retain their leading zeros and example postcodes are deliberately fictional.
    """
    if isinstance(rows, bool) or not isinstance(rows, numbers.Integral) or rows < 40:
        raise ValueError("rows must be an integer of at least 40")
    rng = np.random.default_rng(seed)
    rows = int(rows)
    vehicle_index = np.arange(80)
    occupation_index = np.arange(30)
    postcode_index = np.arange(120)
    vehicle_power = 45 + (vehicle_index * 19) % 190
    vehicle_bodies = np.array(["Hatchback", "Saloon", "Estate", "SUV"])[
        vehicle_index % 4
    ]
    vehicles = pl.DataFrame(
        {
            "VehicleCode": [f"{value + 1:04d}" for value in vehicle_index],
            "Make": [f"Make {value % 8 + 1}" for value in vehicle_index],
            "Model": [f"Model {value % 20 + 1}" for value in vehicle_index],
            "Power": vehicle_power,
            "BodyType": vehicle_bodies,
        }
    )
    physical_work = (occupation_index % 3 == 0).astype(int)
    driving_for_work = (occupation_index % 5 == 0).astype(int)
    occupations = pl.DataFrame(
        {
            "OccupationCode": [f"{value + 1:03d}" for value in occupation_index],
            "OccupationGroup": [f"Group {value % 6 + 1}" for value in occupation_index],
            "PhysicalWork": physical_work,
            "DrivingForWork": driving_for_work,
        }
    )
    area_type = np.array(["Rural", "Town", "City"])[postcode_index % 3]
    density = 30 + (postcode_index * 137) % 8_000
    postcodes = pl.DataFrame(
        {
            "Postcode": [f"SYN{value + 1:04d}" for value in postcode_index],
            "AreaType": area_type,
            "Density": density,
            "DistanceToCentre": 1 + (postcode_index * 7) % 40,
        }
    )
    training_rows = rows * 3 // 4
    split = np.zeros(rows, dtype=np.int8)
    split[:training_rows] = 1
    vehicle = np.concatenate(
        [rng.integers(0, 60, training_rows), rng.integers(0, 80, rows - training_rows)]
    )
    occupation = np.concatenate(
        [rng.integers(0, 24, training_rows), rng.integers(0, 30, rows - training_rows)]
    )
    postcode = np.concatenate(
        [rng.integers(0, 90, training_rows), rng.integers(0, 120, rows - training_rows)]
    )
    age = rng.integers(18, 81, rows)
    bonus = rng.choice([50, 60, 75, 90, 100, 125], rows)
    region = rng.choice(["North", "South", "Urban"], rows)
    exposure = rng.uniform(0.2, 1.5, rows)
    exposure[0:2] = [0.5, 2.0]
    log_frequency = (
        -1.35
        + 0.38 * (age < 25)
        + 0.18 * (age >= 70)
        + 0.005 * (bonus - 75)
        + 0.19 * (region == "Urban")
        + 0.32 * ((age < 30) & (region == "Urban"))
        + 0.22 * ((bonus >= 100) & (region == "South"))
        + 0.003 * (vehicle_power[vehicle] - 130)
        + 0.14 * (vehicle_bodies[vehicle] == "SUV")
        + 0.13 * physical_work[occupation]
        + 0.22 * driving_for_work[occupation]
        + 0.12 * (area_type[postcode] == "City")
        + 0.000015 * density[postcode]
    )
    data = pl.DataFrame(
        {
            "PolicyID": np.arange(1, rows + 1),
            "Claims": rng.poisson(exposure * np.exp(log_frequency)),
            "Exposure": exposure,
            "traintest": split,
            "DriverAge": age,
            "BonusMalus": bonus,
            "Region": region,
            "VehicleCode": vehicles["VehicleCode"].gather(vehicle),
            "OccupationCode": occupations["OccupationCode"].gather(occupation),
            "Postcode": postcodes["Postcode"].gather(postcode),
        }
    )
    return data, vehicles, occupations, postcodes
