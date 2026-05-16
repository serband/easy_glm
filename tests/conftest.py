import numpy as np
import polars as pl
import pytest


@pytest.fixture
def synthetic_insurance_data() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    row_count = 500

    veh_age = rng.integers(0, 16, size=row_count)
    driver_age = rng.integers(18, 80, size=row_count)
    region = rng.choice(
        ["North", "South", "Urban"], size=row_count, p=[0.35, 0.35, 0.30]
    )
    exposure = rng.uniform(0.25, 1.0, size=row_count)

    region_effect = np.select(
        [region == "South", region == "Urban"],
        [0.25, 0.65],
        default=0.0,
    )
    log_frequency = (
        -2.4 + (veh_age * 0.055) + ((80 - driver_age) * 0.01) + region_effect
    )
    claim_count = rng.poisson(exposure * np.exp(log_frequency))
    train_test = (rng.random(row_count) < 0.75).astype(np.int8)

    return pl.DataFrame(
        {
            "VehAge": veh_age,
            "DrivAge": driver_age,
            "Region": region,
            "Exposure": exposure,
            "ClaimNb": claim_count,
            "traintest": train_test,
        }
    )
