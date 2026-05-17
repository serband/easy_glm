from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from easy_glm import (
    fit_lasso_glm,
    generate_all_ratetables,
    generate_blueprint,
    prepare_data,
)
from easy_glm.engine import FromToRow, ModelMetadata, RateModel, VariableConfig


def test_from_rate_tables_numeric():
    rate_table = pl.DataFrame(
        {"VehAge": [0, 5, 10, 15], "relativity": [0.85, 0.90, 1.00, 1.10]}
    )
    blueprint = {"VehAge": [0, 5, 10, 15]}

    rm = RateModel.from_rate_tables({"VehAge": rate_table}, blueprint, base_rate=0.1)

    config = rm.variables["VehAge"]
    assert config.type == "numeric"
    assert len(config.table) == 5

    assert config.table[0].from_ is None
    assert config.table[0].to_ == 0
    assert config.table[0].relativity == 0.85

    assert config.table[1].from_ == 0
    assert config.table[1].to_ == 5
    assert config.table[1].relativity == 0.85

    assert config.table[2].from_ == 5
    assert config.table[2].to_ == 10
    assert config.table[2].relativity == 0.90

    assert config.table[4].from_ == 15
    assert config.table[4].to_ is None
    assert config.table[4].relativity == 1.10


def test_from_rate_tables_categorical():
    rate_table = pl.DataFrame(
        {"Region": ["North", "South", "Urban"], "relativity": [0.95, 1.05, 1.00]}
    )
    blueprint = {"Region": ["North", "South", "Urban"]}

    rm = RateModel.from_rate_tables({"Region": rate_table}, blueprint, base_rate=0.1)

    config = rm.variables["Region"]
    assert config.type == "categorical"
    assert len(config.table) == 4

    assert config.table[0].from_ == "North"
    assert config.table[0].to_ == "North"
    assert config.table[0].relativity == 0.95

    assert config.table[3].from_ is None
    assert config.table[3].to_ is None
    assert config.table[3].relativity == 1.0


def test_from_rate_tables_skips_missing_blueprint():
    rate_table = pl.DataFrame({"VarA": [1, 2], "relativity": [1.0, 1.0]})
    rm = RateModel.from_rate_tables({"VarA": rate_table}, {}, base_rate=0.1)
    assert len(rm.variables) == 0


def test_from_rate_tables_creates_initial_snapshot():
    rate_table = pl.DataFrame({"VehAge": [0, 5], "relativity": [0.85, 0.90]})
    blueprint = {"VehAge": [0, 5]}

    rm = RateModel.from_rate_tables({"VehAge": rate_table}, blueprint, base_rate=0.1)

    assert len(rm.snapshots) == 1
    assert rm.current_version == 1
    assert rm.snapshots[0].description == "Base model"
    assert rm.snapshots[0].version == 1


def test_predict_numeric_exact_levels():
    rm = _make_numeric_rm()
    data = pl.DataFrame({"DrivAge": [18.0, 23.0, 28.0, 33.0, 38.0]})
    preds = rm.predict(data)
    expected = 0.1 * np.array([1.45, 1.30, 1.15, 1.00, 0.90])
    np.testing.assert_array_almost_equal(preds, expected)


def test_predict_numeric_between_levels():
    rm = _make_numeric_rm()
    data = pl.DataFrame({"DrivAge": [20.0, 25.0, 30.0]})
    preds = rm.predict(data)
    expected = 0.1 * np.array([1.45, 1.30, 1.15])
    np.testing.assert_array_almost_equal(preds, expected)


def test_predict_numeric_below_first():
    rm = _make_numeric_rm()
    data = pl.DataFrame({"DrivAge": [10.0, 17.0]})
    preds = rm.predict(data)
    expected = 0.1 * np.array([1.45, 1.45])
    np.testing.assert_array_almost_equal(preds, expected)


def test_predict_numeric_above_last():
    rm = _make_numeric_rm()
    data = pl.DataFrame({"DrivAge": [40.0, 50.0]})
    preds = rm.predict(data)
    expected = 0.1 * np.array([0.90, 0.90])
    np.testing.assert_array_almost_equal(preds, expected)


def test_predict_numeric_boundary_edge():
    rm = _make_numeric_rm()
    data = pl.DataFrame({"DrivAge": [18.0, 22.999, 23.0]})
    preds = rm.predict(data)
    np.testing.assert_array_almost_equal(preds, 0.1 * np.array([1.45, 1.45, 1.30]))


def test_predict_categorical_exact():
    rm = _make_categorical_rm()
    data = pl.DataFrame({"Region": ["North", "South", "Urban"]})
    preds = rm.predict(data)
    expected = 0.1 * np.array([0.95, 1.05, 1.00])
    np.testing.assert_array_almost_equal(preds, expected)


def test_predict_categorical_unknown():
    rm = _make_categorical_rm()
    data = pl.DataFrame({"Region": ["Rural", "West"]})
    preds = rm.predict(data)
    expected = 0.1 * np.array([1.0, 1.0])
    np.testing.assert_array_almost_equal(preds, expected)


def test_predict_categorical_null():
    rm = _make_categorical_rm()
    data = pl.DataFrame({"Region": ["North", None]})
    preds = rm.predict(data)
    expected = 0.1 * np.array([0.95, 1.0])
    np.testing.assert_array_almost_equal(preds, expected)


def test_predict_multiple_variables():
    rm = _make_multi_rm()
    data = pl.DataFrame({"DrivAge": [23.0], "Region": ["North"]})
    preds = rm.predict(data)
    expected = 0.1 * 1.15 * 0.95
    np.testing.assert_array_almost_equal(preds, [expected])


def test_predict_empty_data():
    rm = _make_numeric_rm()
    data = pl.DataFrame({"DrivAge": pl.Series([], dtype=pl.Float64)})
    preds = rm.predict(data)
    assert len(preds) == 0


def test_predict_missing_column():
    rm = _make_numeric_rm()
    data = pl.DataFrame({"WrongCol": [1.0]})
    with pytest.raises(ValueError, match="Column 'DrivAge' not found"):
        rm.predict(data)


def test_predict_with_version():
    rm = _make_numeric_rm()
    original = rm.predict(pl.DataFrame({"DrivAge": [20.0]}))

    rm.update_relativity("DrivAge", from_=18, to_=23, new_value=2.0)
    rm.create_snapshot("Version 2")

    v2 = rm.predict(pl.DataFrame({"DrivAge": [20.0]}))
    assert v2[0] == 0.1 * 2.0
    assert v2[0] != original[0]

    v1 = rm.predict(pl.DataFrame({"DrivAge": [20.0]}), version=1)
    assert v1[0] == original[0]


def test_update_relativity():
    rm = _make_numeric_rm()

    data = pl.DataFrame({"DrivAge": [20.0]})
    before = rm.predict(data)

    rm.update_relativity("DrivAge", from_=18, to_=23, new_value=2.0)
    after = rm.predict(data)

    assert before[0] == 0.1 * 1.45
    assert after[0] == 0.1 * 2.0


def test_update_relativity_non_existent_variable():
    rm = _make_numeric_rm()
    with pytest.raises(KeyError, match="Variable 'FakeVar' not found"):
        rm.update_relativity("FakeVar", from_=1, to_=2, new_value=1.0)


def test_update_relativity_non_existent_row():
    rm = _make_numeric_rm()
    with pytest.raises(ValueError, match="No row found"):
        rm.update_relativity("DrivAge", from_=99, to_=100, new_value=1.0)


def test_create_snapshot():
    rm = _make_numeric_rm()
    rm.update_relativity("DrivAge", from_=18, to_=23, new_value=2.0)
    version = rm.create_snapshot("Test edit")

    assert version == 2
    assert rm.current_version == 2
    assert len(rm.snapshots) == 2

    s2 = rm.snapshots[1]
    assert s2.description == "Test edit"
    assert s2.parent_version == 1
    assert len(s2.changes) == 1
    assert s2.changes[0].variable == "DrivAge"
    assert s2.changes[0].new_relativity == 2.0


def test_create_snapshot_clears_pending_changes():
    rm = _make_numeric_rm()
    rm.update_relativity("DrivAge", from_=18, to_=23, new_value=2.0)
    rm.create_snapshot("Edit 1")

    rm.create_snapshot("No edits")
    s3 = rm.snapshots[2]
    assert len(s3.changes) == 0


def test_switch_to():
    rm = _make_numeric_rm()

    data = pl.DataFrame({"DrivAge": [20.0]})
    original = rm.predict(data)

    rm.update_relativity("DrivAge", from_=18, to_=23, new_value=2.0)
    rm.create_snapshot("v2")

    rm.switch_to(1)
    assert rm.current_version == 1
    assert rm.predict(data)[0] == original[0]


def test_switch_to_invalid():
    rm = _make_numeric_rm()
    with pytest.raises(ValueError, match="Invalid version"):
        rm.switch_to(0)
    with pytest.raises(ValueError, match="Invalid version"):
        rm.switch_to(99)


def test_list_snapshots():
    rm = _make_numeric_rm()
    rm.update_relativity("DrivAge", from_=18, to_=23, new_value=2.0)
    rm.create_snapshot("Edit")

    snapshots = rm.list_snapshots()
    assert len(snapshots) == 2
    assert snapshots[0]["version"] == 1
    assert snapshots[1]["version"] == 2
    assert snapshots[0]["changes_count"] == 0
    assert snapshots[1]["changes_count"] == 1


def test_diff():
    rm = _make_numeric_rm()
    rm.update_relativity("DrivAge", from_=18, to_=23, new_value=2.0)
    rm.create_snapshot("Edit")

    changes = rm.diff(1, 2)
    assert len(changes) == 1
    assert changes[0].variable == "DrivAge"
    assert changes[0].old_relativity == 1.45
    assert changes[0].new_relativity == 2.0


def test_to_json_from_json_roundtrip(tmp_path):
    rm = _make_multi_rm()
    rm.update_relativity("DrivAge", from_=18, to_=23, new_value=2.0)
    rm.create_snapshot("Edit")

    path = tmp_path / "model.easyglm"
    rm.to_json(path)

    loaded = RateModel.from_json(path)

    data = pl.DataFrame({"DrivAge": [20.0], "Region": ["North"]})
    assert loaded.predict(data)[0] == pytest.approx(rm.predict(data)[0])
    assert loaded.base_rate == rm.base_rate
    assert loaded.current_version == rm.current_version
    assert len(loaded.snapshots) == len(rm.snapshots)
    assert len(loaded.variables) == len(rm.variables)


def test_to_json_from_json_preserves_snapshot_relativities(tmp_path):
    rm = _make_numeric_rm()
    rm.update_relativity("DrivAge", from_=18, to_=23, new_value=2.0)
    rm.create_snapshot("Edit")

    path = tmp_path / "model.easyglm"
    rm.to_json(path)

    loaded = RateModel.from_json(path)
    loaded.switch_to(1)
    v1_data = pl.DataFrame({"DrivAge": [20.0]})
    assert loaded.predict(v1_data)[0] == pytest.approx(0.1 * 1.45)

    loaded.switch_to(2)
    assert loaded.predict(v1_data)[0] == pytest.approx(0.1 * 2.0)


class TestIntegrationWithPipeline:
    def test_full_pipeline(self, synthetic_insurance_data):
        df = synthetic_insurance_data
        predictors = ["VehAge", "Region", "DrivAge"]
        bp = generate_blueprint(df)
        prepped = prepare_data(
            df=df,
            modelling_variables=predictors,
            additional_columns=["Exposure", "ClaimNb", "traintest"],
            formats=bp,
            traintest_column="traintest",
            table_name="cars",
        )
        model = fit_lasso_glm(
            dataframe=prepped,
            target="ClaimNb",
            model_type="Poisson",
            weight_col="Exposure",
            train_test_col="traintest",
            divide_target_by_weight=True,
        )
        all_tables = generate_all_ratetables(
            model=model,
            dataset=df,
            predictor_variables=predictors,
            blueprint=bp,
            random_seed=42,
        )

        rm = RateModel.from_rate_tables(all_tables, bp, base_rate=0.05)

        assert set(rm.variables.keys()) == {"VehAge", "DrivAge", "Region"}
        assert rm.variables["VehAge"].type == "numeric"
        assert rm.variables["DrivAge"].type == "numeric"
        assert rm.variables["Region"].type == "categorical"
        assert rm.current_version == 1

        data = pl.DataFrame({"VehAge": [5], "DrivAge": [30], "Region": ["North"]})
        preds = rm.predict(data)
        assert len(preds) == 1
        assert np.isfinite(preds[0])

    def test_roundtrip_after_full_pipeline(self, synthetic_insurance_data, tmp_path):
        df = synthetic_insurance_data
        predictors = ["VehAge", "Region", "DrivAge"]
        bp = generate_blueprint(df)
        prepped = prepare_data(
            df=df,
            modelling_variables=predictors,
            additional_columns=["Exposure", "ClaimNb", "traintest"],
            formats=bp,
            traintest_column="traintest",
            table_name="cars",
        )
        model = fit_lasso_glm(
            dataframe=prepped,
            target="ClaimNb",
            model_type="Poisson",
            weight_col="Exposure",
            train_test_col="traintest",
            divide_target_by_weight=True,
        )
        all_tables = generate_all_ratetables(
            model=model,
            dataset=df,
            predictor_variables=predictors,
            blueprint=bp,
            random_seed=42,
        )

        rm = RateModel.from_rate_tables(all_tables, bp, base_rate=0.05)
        data = pl.DataFrame({"VehAge": [5], "DrivAge": [30], "Region": ["North"]})
        before = rm.predict(data)

        path = tmp_path / "model.easyglm"
        rm.to_json(path)
        loaded = RateModel.from_json(path)
        after = loaded.predict(data)

        np.testing.assert_array_almost_equal(before, after)

    def test_from_glm_model(self, synthetic_insurance_data):
        df = synthetic_insurance_data
        predictors = ["VehAge", "Region", "DrivAge"]
        bp = generate_blueprint(df)
        prepped = prepare_data(
            df=df,
            modelling_variables=predictors,
            additional_columns=["Exposure", "ClaimNb", "traintest"],
            formats=bp,
            traintest_column="traintest",
            table_name="cars",
        )
        model = fit_lasso_glm(
            dataframe=prepped,
            target="ClaimNb",
            model_type="Poisson",
            weight_col="Exposure",
            train_test_col="traintest",
            divide_target_by_weight=True,
        )

        rm = RateModel.from_glm_model(
            model=model,
            dataset=df,
            blueprint=bp,
            base_rate=0.05,
            model_type="poisson",
            target="ClaimNb",
            weight_col="Exposure",
            train_test_col="traintest",
            predictor_variables=predictors,
        )

        assert set(rm.variables.keys()) == {"VehAge", "DrivAge", "Region"}
        assert rm.metadata.model_type == "poisson"
        assert rm.metadata.target == "ClaimNb"
        assert rm.current_version == 1

        data = pl.DataFrame({"VehAge": [5], "DrivAge": [30], "Region": ["North"]})
        preds = rm.predict(data)
        assert len(preds) == 1
        assert np.isfinite(preds[0])


def _make_numeric_rm() -> RateModel:
    table = [
        FromToRow(from_=None, to_=18, relativity=1.45),
        FromToRow(from_=18, to_=23, relativity=1.45),
        FromToRow(from_=23, to_=28, relativity=1.30),
        FromToRow(from_=28, to_=33, relativity=1.15),
        FromToRow(from_=33, to_=38, relativity=1.00),
        FromToRow(from_=38, to_=None, relativity=0.90),
    ]
    variables = {"DrivAge": VariableConfig(type="numeric", table=table)}
    rm = RateModel(base_rate=0.1, variables=variables)
    rm.create_snapshot("Base")
    return rm


def _make_categorical_rm() -> RateModel:
    table = [
        FromToRow(from_="North", to_="North", relativity=0.95),
        FromToRow(from_="South", to_="South", relativity=1.05),
        FromToRow(from_="Urban", to_="Urban", relativity=1.00),
        FromToRow(from_=None, to_=None, relativity=1.0),
    ]
    variables = {"Region": VariableConfig(type="categorical", table=table)}
    rm = RateModel(base_rate=0.1, variables=variables)
    rm.create_snapshot("Base")
    return rm


def _make_multi_rm() -> RateModel:
    num_table = [
        FromToRow(from_=None, to_=18, relativity=1.45),
        FromToRow(from_=18, to_=23, relativity=1.30),
        FromToRow(from_=23, to_=28, relativity=1.15),
        FromToRow(from_=28, to_=33, relativity=1.00),
        FromToRow(from_=33, to_=38, relativity=0.90),
        FromToRow(from_=38, to_=None, relativity=0.90),
    ]
    cat_table = [
        FromToRow(from_="North", to_="North", relativity=0.95),
        FromToRow(from_="South", to_="South", relativity=1.05),
        FromToRow(from_="Urban", to_="Urban", relativity=1.00),
        FromToRow(from_=None, to_=None, relativity=1.0),
    ]
    variables = {
        "DrivAge": VariableConfig(type="numeric", table=num_table),
        "Region": VariableConfig(type="categorical", table=cat_table),
    }
    rm = RateModel(base_rate=0.1, variables=variables)
    rm.create_snapshot("Base")
    return rm


class TestMetadata:
    def test_from_rate_tables_stores_metadata(self):
        rate_table = pl.DataFrame(
            {"VehAge": [0, 5, 10], "relativity": [0.85, 0.90, 1.00]}
        )
        blueprint = {"VehAge": [0, 5, 10]}

        rm = RateModel.from_rate_tables(
            {"VehAge": rate_table},
            blueprint,
            base_rate=0.1,
            model_type="poisson",
            target="ClaimNb",
            weight_col="Exposure",
            train_test_col="traintest",
        )

        assert rm.metadata.model_type == "poisson"
        assert rm.metadata.target == "ClaimNb"
        assert rm.metadata.weight_col == "Exposure"
        assert rm.metadata.train_test_col == "traintest"
        assert rm.metadata.predictor_variables == ["VehAge"]

    def test_metadata_roundtrip_json(self, tmp_path):
        rm = _make_numeric_rm()
        rm.metadata.model_type = "poisson"
        rm.metadata.target = "ClaimNb"
        rm.metadata.weight_col = "Exposure"

        path = tmp_path / "model.easyglm"
        rm.to_json(path)

        loaded = RateModel.from_json(path)
        assert loaded.metadata.model_type == "poisson"
        assert loaded.metadata.target == "ClaimNb"
        assert loaded.metadata.weight_col == "Exposure"

    def test_metadata_in_snapshot(self):
        rm = _make_numeric_rm()
        rm.metadata.model_type = "poisson"
        rm.metadata.target = "ClaimNb"
        rm.create_snapshot("Added metadata")

        s = rm.snapshots[-1]
        assert s.metadata["model_type"] == "poisson"
        assert s.metadata["target"] == "ClaimNb"

    def test_switch_to_restores_metadata(self):
        rm = _make_numeric_rm()
        rm.metadata.target = "ClaimNb"
        rm.create_snapshot("Has target")

        rm.metadata.target = "DifferentTarget"
        rm.create_snapshot("Different target")

        rm.switch_to(2)
        assert rm.metadata.target == "ClaimNb"

        rm.switch_to(3)
        assert rm.metadata.target == "DifferentTarget"


class TestColumnMapping:
    def test_predict_with_column_map(self):
        rm = _make_numeric_rm()
        data = pl.DataFrame({"driver_age": [20.0], "extra": [1.0]})

        preds = rm.predict(data, column_map={"driver_age": "DrivAge"})
        assert preds[0] == pytest.approx(0.1 * 1.45)

    def test_predict_uses_model_column_mapping(self):
        rm = _make_numeric_rm()
        rm.column_mapping = {"driver_age": "DrivAge"}
        data = pl.DataFrame({"driver_age": [20.0]})

        preds = rm.predict(data)
        assert preds[0] == pytest.approx(0.1 * 1.45)

    def test_column_mapping_persists_in_snapshot(self):
        rm = _make_numeric_rm()
        rm.column_mapping = {"driver_age": "DrivAge"}
        rm.create_snapshot("With mapping")

        rm.column_mapping = {}
        rm.create_snapshot("Without mapping")

        rm.switch_to(2)
        assert rm.column_mapping == {"driver_age": "DrivAge"}

        rm.switch_to(3)
        assert rm.column_mapping == {}

    def test_column_mapping_roundtrip_json(self, tmp_path):
        rm = _make_numeric_rm()
        rm.column_mapping = {"a": "DrivAge", "b": "VehAge"}
        rm.create_snapshot("With mapping")

        path = tmp_path / "model.easyglm"
        rm.to_json(path)

        loaded = RateModel.from_json(path)
        assert loaded.column_mapping == {"a": "DrivAge", "b": "VehAge"}
        assert loaded.snapshots[1].column_mapping == {"a": "DrivAge", "b": "VehAge"}


class TestExposure:
    def _make_rm_with_exposure(self) -> RateModel:
        table = [
            FromToRow(from_=None, to_=18, relativity=1.45),
            FromToRow(from_=18, to_=23, relativity=1.30),
            FromToRow(from_=23, to_=28, relativity=1.15),
            FromToRow(from_=28, to_=None, relativity=1.00),
        ]
        variables = {"DrivAge": VariableConfig(type="numeric", table=table)}
        metadata = ModelMetadata(exposure_col="Exposure")
        rm = RateModel(base_rate=0.1, variables=variables, metadata=metadata)
        rm.create_snapshot("Base")
        return rm

    def test_predict_with_exposure(self):
        rm = self._make_rm_with_exposure()
        data = pl.DataFrame({"DrivAge": [20.0, 25.0], "Exposure": [1.0, 0.5]})
        preds = rm.predict(data)
        expected = 0.1 * np.array([1.30, 1.15]) * np.array([1.0, 0.5])
        np.testing.assert_array_almost_equal(preds, expected)

    def test_predict_exposure_col_not_found_warns(self):
        rm = self._make_rm_with_exposure()
        data = pl.DataFrame({"DrivAge": [20.0]})
        with pytest.warns(UserWarning, match="Exposure column 'Exposure' not found"):
            preds = rm.predict(data)
        expected = 0.1 * 1.30
        np.testing.assert_array_almost_equal(preds, [expected])

    def test_predict_exposure_override(self):
        rm = self._make_rm_with_exposure()
        data = pl.DataFrame({"DrivAge": [20.0], "Exp2": [2.0]})
        preds = rm.predict(data, exposure_col="Exp2")
        expected = 0.1 * 1.30 * 2.0
        np.testing.assert_array_almost_equal(preds, [expected])

    def test_predict_exposure_none_override(self):
        rm = self._make_rm_with_exposure()
        data = pl.DataFrame({"DrivAge": [20.0], "Exposure": [3.0]})
        preds = rm.predict(data, exposure_col=None)
        expected = 0.1 * 1.30
        np.testing.assert_array_almost_equal(preds, [expected])

    def test_exposure_roundtrip_json(self, tmp_path):
        rm = self._make_rm_with_exposure()
        path = tmp_path / "model.easyglm"
        rm.to_json(path)

        loaded = RateModel.from_json(path)
        assert loaded.metadata.exposure_col == "Exposure"

        data = pl.DataFrame({"DrivAge": [20.0], "Exposure": [2.0]})
        preds = loaded.predict(data)
        expected = 0.1 * 1.30 * 2.0
        np.testing.assert_array_almost_equal(preds, [expected])

    def test_from_rate_tables_stores_exposure(self):
        rate_table = pl.DataFrame(
            {"VehAge": [0, 5, 10], "relativity": [0.85, 0.90, 1.00]}
        )
        blueprint = {"VehAge": [0, 5, 10]}
        rm = RateModel.from_rate_tables(
            {"VehAge": rate_table},
            blueprint,
            base_rate=0.1,
            exposure_col="Exposure",
        )
        assert rm.metadata.exposure_col == "Exposure"

    def test_predict_no_exposure_stored(self):
        rm = _make_numeric_rm()
        data = pl.DataFrame({"DrivAge": [20.0], "Exposure": [2.0]})
        preds = rm.predict(data)
        expected = 0.1 * 1.45
        np.testing.assert_array_almost_equal(preds, [expected])
