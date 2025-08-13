from typing import Union
import urllib.request
import polars as pl
import pandas as pd
import rdata

def load_external_dataframe() -> pl.DataFrame:
    """Download and return the French Motor dataset as a Polars DataFrame."""
    url = "https://github.com/dutangc/CASdatasets/raw/master/data/freMTPL2freq.rda"
    with urllib.request.urlopen(url) as response:
        data = response.read()
    parsed_data = rdata.parser.parse_data(data)
    converted_data = rdata.conversion.convert(parsed_data)
    df: Union[pd.DataFrame, pl.DataFrame] = converted_data["freMTPL2freq"]
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    return df
