import numpy as np
import polars as pl
import pytest

from easy_glm.core.blueprint import generate_blueprint
from easy_glm.core.prepare import prepare_data


def test_generate_blueprint_all_nulls():
    """
    Test generate_blueprint with a DataFrame containing all null values.
    """
    data = {
        'numeric_col': [None, None, None, None, None],
        'categorical_col': [None, None, None, None, None]
    }
    df = pl.DataFrame(data)
    blueprint = generate_blueprint(df)

    assert isinstance(blueprint, dict)
    assert 'numeric_col' in blueprint
    assert 'categorical_col' in blueprint

    # Check that the blueprint for the numeric column is an empty list
    assert blueprint['numeric_col'] == []

    # Check that the blueprint for the categorical column is an empty list
    assert blueprint['categorical_col'] == []


def test_generate_blueprint_majority_nulls():
    """
    Test generate_blueprint with a DataFrame containing a majority of null values.
    """
    data = {
        'numeric_col': [1.0, None, 3.0, None, None],
        'categorical_col': ['A', None, 'C', None, None]
    }
    df = pl.DataFrame(data)
    blueprint = generate_blueprint(df)

    assert isinstance(blueprint, dict)
    assert 'numeric_col' in blueprint
    assert 'categorical_col' in blueprint

    # Check the blueprint for the numeric column
    assert blueprint['numeric_col'] == [1.0, 3.0]

    # Check the blueprint for the categorical column
    assert blueprint['categorical_col'] == ['A', 'C']


def test_prepare_data_all_nulls():
    """
    Test prepare_data with a DataFrame containing all null values.
    """
    data = {
        'numeric_col': [None, None, None, None, None],
        'categorical_col': [None, None, None, None, None]
    }
    df = pl.DataFrame(data)
    blueprint = generate_blueprint(df)
    prepared_df = prepare_data(
        modelling_variables=['numeric_col', 'categorical_col'],
        df=df,
        formats=blueprint
    )
    assert prepared_df.is_empty()


def test_prepare_data_majority_nulls():
    """
    Test prepare_data with a DataFrame containing a majority of null values.
    """
    data = {
        'numeric_col': [1.0, None, 3.0, None, None],
        'categorical_col': ['A', None, 'C', None, None]
    }
    df = pl.DataFrame(data)
    blueprint = generate_blueprint(df)
    prepared_df = prepare_data(
        modelling_variables=['numeric_col', 'categorical_col'],
        df=df,
        formats=blueprint
    )
    assert prepared_df.shape == (5, 3)
    assert 'numeric_col1.0' in prepared_df.columns
    assert 'numeric_col3.0' in prepared_df.columns
    assert 'categorical_col_lumped' in prepared_df.columns
    assert prepared_df['categorical_col_lumped'].to_list() == ['A', 'Other', 'C', 'Other', 'Other']
