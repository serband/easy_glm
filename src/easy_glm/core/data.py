import hashlib
import urllib.request
from pathlib import Path

import pandas as pd
import polars as pl

_CACHE_DIR = Path.home() / ".cache" / "easy_glm"
_FRENCH_MOTOR_URL = (
    "https://github.com/dutangc/CASdatasets/raw/master/data/freMTPL2freq.rda"
)
_SWEDISH_MOTORCYCLE_URL = (
    "https://github.com/dutangc/CASdatasets/raw/master/data/swmotorcycle.rda"
)


def _cache_path(url: str) -> Path:
    """Return the cache file path for a given URL."""
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    return _CACHE_DIR / f"{url_hash}.parquet"


def _load_rda_dataframe(
    url: str,
    object_name: str,
    *,
    cache: bool,
) -> pl.DataFrame:
    """Download one dataframe from an R data file, with a Parquet cache."""
    cache_file = _cache_path(url)
    if cache and cache_file.exists():
        return pl.read_parquet(str(cache_file))

    import rdata

    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read()
    parsed_data = rdata.parser.parse_data(data)
    converted_data = rdata.conversion.convert(parsed_data)
    try:
        df: pd.DataFrame | pl.DataFrame = converted_data[object_name]
    except KeyError as exc:
        available = ", ".join(sorted(converted_data)) or "none"
        raise ValueError(
            f"The downloaded data did not contain {object_name!r}; "
            f"available objects: {available}"
        ) from exc
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    if cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(str(cache_file))

    return df


def load_external_dataframe(
    url: str | None = None,
    cache: bool = True,
) -> pl.DataFrame:
    """Download the French Motor dataset and return as a Polars DataFrame.

    By default, the dataset is cached to ``~/.cache/easy_glm/`` so
    subsequent calls are instant. Pass ``cache=False`` to force a
    fresh download.

    Parameters
    ----------
    url : str or None
        URL of the ``.rda`` dataset. Defaults to the French Motor
        third-party liability dataset from the CASdatasets repository.
    cache : bool
        If True (default), cache the dataset on disk after the first
        download.

    Returns
    -------
    pl.DataFrame
    """
    if url is None:
        url = _FRENCH_MOTOR_URL
    return _load_rda_dataframe(url, "freMTPL2freq", cache=cache)


def load_swedish_motorcycle_data(cache: bool = True) -> pl.DataFrame:
    """Download the Swedish motorcycle insurance portfolio.

    The public CASdatasets table contains policy-year exposure, claim count,
    total claim payments and six rating factors. Claim-free policies are kept,
    making ``ClaimAmount`` suitable for a Tweedie pure-premium example after
    rows with zero exposure are removed.

    By default, the data is cached in ``~/.cache/easy_glm/`` so later calls are
    immediate.
    """
    return _load_rda_dataframe(
        _SWEDISH_MOTORCYCLE_URL,
        "swmotorcycle",
        cache=cache,
    )
