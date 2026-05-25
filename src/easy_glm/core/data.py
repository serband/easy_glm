import hashlib
import urllib.request
from pathlib import Path

import pandas as pd
import polars as pl
import rdata

_CACHE_DIR = Path.home() / ".cache" / "easy_glm"


def _cache_path(url: str) -> Path:
    """Return the cache file path for a given URL."""
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    return _CACHE_DIR / f"{url_hash}.parquet"


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
        url = "https://github.com/dutangc/CASdatasets/raw/master/data/freMTPL2freq.rda"

    cache_file = _cache_path(url)
    if cache and cache_file.exists():
        return pl.read_parquet(str(cache_file))

    with urllib.request.urlopen(url) as response:
        data = response.read()
    parsed_data = rdata.parser.parse_data(data)
    converted_data = rdata.conversion.convert(parsed_data)
    df: pd.DataFrame | pl.DataFrame = converted_data["freMTPL2freq"]
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    if cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(str(cache_file))

    return df
