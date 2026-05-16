import polars as pl

from easy_glm.core.blueprint import generate_blueprint
from easy_glm.core.prepare import prepare_data


def test_generate_blueprint_all_nulls():
    """
    Test generate_blueprint with a DataFrame containing all null values.
    """
    data = {
        "numeric_col": [None, None, None, None, None],
        "categorical_col": [None, None, None, None, None],
    }
    df = pl.DataFrame(data)
    blueprint = generate_blueprint(df)

    assert isinstance(blueprint, dict)
    assert "numeric_col" in blueprint
    assert "categorical_col" in blueprint

    # Check that the blueprint for the numeric column is an empty list
    assert blueprint["numeric_col"] == []

    # Check that the blueprint for the categorical column is an empty list
    assert blueprint["categorical_col"] == []


def test_generate_blueprint_majority_nulls():
    """
    Test generate_blueprint with a DataFrame containing a majority of null values.
    """
    data = {
        "numeric_col": [1.0, None, 3.0, None, None],
        "categorical_col": ["A", None, "C", None, None],
    }
    df = pl.DataFrame(data)
    blueprint = generate_blueprint(df)

    assert isinstance(blueprint, dict)
    assert "numeric_col" in blueprint
    assert "categorical_col" in blueprint

    # Check the blueprint for the numeric column
    assert blueprint["numeric_col"] == [1.0, 3.0]

    # Check the blueprint for the categorical column
    assert blueprint["categorical_col"] == ["A", "C"]


def test_prepare_data_all_nulls():
    """
    Test prepare_data with a DataFrame containing all null values.
    """
    data = {
        "numeric_col": [None, None, None, None, None],
        "categorical_col": [None, None, None, None, None],
    }
    df = pl.DataFrame(data)
    blueprint = generate_blueprint(df)
    prepared_df = prepare_data(
        modelling_variables=["numeric_col", "categorical_col"], df=df, formats=blueprint
    )
    assert prepared_df.is_empty()


def test_prepare_data_majority_nulls():
    """
    Test prepare_data with a DataFrame containing a majority of null values.
    """
    data = {
        "numeric_col": [1.0, None, 3.0, None, None],
        "categorical_col": ["A", None, "C", None, None],
    }
    df = pl.DataFrame(data)
    blueprint = generate_blueprint(df)
    prepared_df = prepare_data(
        modelling_variables=["numeric_col", "categorical_col"], df=df, formats=blueprint
    )
    assert prepared_df.shape == (5, 3)
    assert "numeric_col1.0" in prepared_df.columns
    assert "numeric_col3.0" in prepared_df.columns
    assert "categorical_col_lumped" in prepared_df.columns
    assert prepared_df["categorical_col_lumped"].to_list() == [
        "A",
        "Other",
        "C",
        "Other",
        "Other",
    ]


def test_prepare_data_handles_identifiers_with_spaces_and_reserved_words():
    df = pl.DataFrame(
        {
            "vehicle age": [1.0, 3.0, None],
            "select": ["A", "B", "C"],
        }
    )
    blueprint = {
        "vehicle age": [2.0],
        "select": ["A", "B"],
    }

    prepared_df = prepare_data(
        modelling_variables=["vehicle age", "select"],
        df=df,
        formats=blueprint,
        table_name="policy data",
    )

    assert prepared_df.columns == ["vehicle age2.0", "select_lumped"]
    assert prepared_df["select_lumped"].to_list() == ["A", "B", "Other"]


def test_prepare_data_preserves_unformatted_columns():
    df = pl.DataFrame({"raw score": [10, 20, 30]})

    prepared_df = prepare_data(
        modelling_variables=["raw score"],
        df=df,
        formats={},
    )

    assert prepared_df.columns == ["raw score"]
    assert prepared_df["raw score"].to_list() == [10, 20, 30]
