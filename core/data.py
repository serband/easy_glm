from typing import Union
import urllib.request
import polars as pl
import pandas as pd
import rdata

def load_external_dataframe() -> pl.DataFrame:
    """
    Load the French Motor dataset (freMTPL2freq) as a Polars DataFrame.

    Downloads and parses the dataset from GitHub using rdata, returning a Polars DataFrame.

    Returns:
        pl.DataFrame: The loaded French Motor dataset.

    Example:
        >>> df = load_external_dataframe()
    """
    url = "https://github.com/dutangc/CASdatasets/raw/master/data/freMTPL2freq.rda"
    with urllib.request.urlopen(url) as response:
        data = response.read()
    parsed_data = rdata.parser.parse_data(data)
    converted_data = rdata.conversion.convert(parsed_data)
    df: Union[pd.DataFrame, pl.DataFrame] = converted_data['freMTPL2freq']
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    return df
