"""Offline public-example loaders for isolated onboarding API/browser tests."""

from __future__ import annotations

import numpy as np
import polars as pl


def example_frames() -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(42)
    n = 240
    claims = rng.poisson(0.3, n)
    french = pl.DataFrame(
        {
            "IDpol": np.arange(n),
            "ClaimNb": claims,
            "Exposure": rng.uniform(0.2, 1.0, n),
            "DrivAge": rng.integers(18, 85, n),
            "Region": rng.choice(["North", "South", "West"], n),
            "BonusMalus": rng.integers(50, 110, n),
            "Density": rng.integers(20, 500, n),
        }
    )
    exposure = rng.uniform(0.2, 1.0, n)
    exposure[:2] = 0.0
    swedish = pl.DataFrame(
        {
            "ClaimAmount": claims * rng.gamma(2.0, 400.0, n),
            "ClaimNb": claims,
            "Exposure": exposure,
            "OwnerAge": rng.integers(18, 85, n),
            "Gender": rng.choice(["M", "F"], n),
            "Area": rng.integers(1, 4, n),
            "RiskClass": rng.integers(1, 5, n),
            "VehAge": rng.integers(0, 15, n),
            "BonusClass": rng.integers(1, 8, n),
        }
    )
    return {"french_motor": french, "swedish_motorcycle": swedish}


def install_example_loaders() -> None:
    """Call before starting a private desktop server; never use in production."""
    import easy_glm

    frames = example_frames()
    easy_glm.load_external_dataframe = lambda: frames["french_motor"].clone()
    easy_glm.load_swedish_motorcycle_data = lambda: frames["swedish_motorcycle"].clone()
