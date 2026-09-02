from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from easy_glm import DesignSpec, base_rate, fit_glm, rate_tables, to_rate_model
from easy_glm.engine import FromToRow, ModelMetadata, RateModel, VariableConfig


def _numeric_table(
    edges: list[float], rels: list[float], null_rel: float | None = None
):
    """Bands (None, e0), [e0, e1), ..., [e_last, None) (+ null row) in the 0.3 table format."""
    froms: list[float | None] = [None, *edges]
    tos: list[float | None] = [*edges, None]
    if null_rel is not None:
        froms.append(None)
        tos.append(None)
        rels = [*rels, null_rel]
    return pl.DataFrame(
        {
            "from": pl.Series(froms, dtype=pl.Float64),
            "to": pl.Series(tos, dtype=pl.Float64),
            "relativity": rels,
        }
    )


def _categorical_table(
    levels: list[str], rels: list[float], other: float | None = None
):
    froms: list[str | None] = list(levels)
    tos: list[str | None] = list(levels)
    if other is not None:
        froms.append(None)
        tos.append(None)
        rels = [*rels, other]
    return pl.DataFrame(
        {
            "from": pl.Series(froms, dtype=pl.Utf8),
            "to": pl.Series(tos, dtype=pl.Utf8),
            "relativity": rels,
        }
    )


def test_from_rate_tables_numeric():
    table = _numeric_table([0, 5, 10, 15], [0.85, 0.85, 0.90, 1.00, 1.10])
    rm = RateModel.from_rate_tables({"VehAge": table}, base_rate=0.1)

    config = rm.variables["VehAge"]
    assert config.type == "numeric"
    assert len(config.table) == 5
    assert (config.table[0].from_, config.table[0].to_) == (None, 0)
    assert config.table[0].relativity == 0.85
    assert (config.table[2].from_, config.table[2].to_) == (5, 10)
    assert config.table[2].relativity == 0.90
    assert (config.table[4].from_, config.table[4].to_) == (15, None)
    assert config.null_relativity is None
    np.testing.assert_array_equal(
        rm.predict(pl.DataFrame({"VehAge": [-1.0, 0.0, 7.0, 15.0, 99.0]})),
        0.1 * np.array([0.85, 0.85, 0.90, 1.10, 1.10]),
    )


def test_from_rate_tables_numeric_with_null_row():
    table = _numeric_table([0, 5], [0.8, 0.9, 1.0], null_rel=1.2)
    rm = RateModel.from_rate_tables({"VehAge": table}, base_rate=1.0)
    assert rm.variables["VehAge"].null_relativity == 1.2
    np.testing.assert_array_equal(
        rm.predict(pl.DataFrame({"VehAge": [None, 3.0]})), [1.2, 0.9]
    )


def test_from_rate_tables_categorical():
    table = _categorical_table(["North", "South", "Urban"], [0.95, 1.05, 1.00])
    rm = RateModel.from_rate_tables({"Region": table}, base_rate=0.1)

    config = rm.variables["Region"]
    assert config.type == "categorical"
    assert len(config.table) == 4
    assert (config.table[0].from_, config.table[0].to_) == ("North", "North")
    assert config.table[0].relativity == 0.95
    assert (config.table[3].from_, config.table[3].to_) == (None, None)
    assert config.table[3].relativity == 1.0  # Other row added when absent

    with_other = _categorical_table(["A", "B"], [0.9, 1.1], other=1.3)
    rm2 = RateModel.from_rate_tables({"R": with_other}, base_rate=1.0)
    np.testing.assert_array_equal(
        rm2.predict(pl.DataFrame({"R": ["A", "B", "zzz", None]})), [0.9, 1.1, 1.3, 1.3]
    )


def test_from_rate_tables_rejects_bad_tables():
    with pytest.raises(KeyError):
        RateModel.from_rate_tables({}, base_rate=0.1, predictor_variables=["X"])
    with pytest.raises(ValueError, match="lacks columns"):
        RateModel.from_rate_tables(
            {"X": pl.DataFrame({"from": [1.0], "relativity": [1.0]})}, base_rate=0.1
        )
    gap = pl.DataFrame(
        {
            "from": pl.Series([None, 0.0, 6.0], dtype=pl.Float64),
            "to": pl.Series([0.0, 5.0, None], dtype=pl.Float64),
            "relativity": [1.0, 1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="gap or overlap"):
        RateModel.from_rate_tables({"X": gap}, base_rate=0.1)


def test_from_rate_tables_creates_initial_snapshot():
    table = _numeric_table([0, 5], [0.85, 0.90, 0.95])
    rm = RateModel.from_rate_tables({"VehAge": table}, base_rate=0.1)

    assert len(rm.snapshots) == 1
    assert rm.current_version == 1
    assert rm.snapshots[0].description == "Base model"
    assert rm.snapshots[0].metrics is None


def test_snapshot_metrics_are_stored_and_round_trip(tmp_path):
    rm = _make_numeric_rm()
    rm.create_snapshot("with metrics", metrics={"holdout": {"ae": 1.02}})
    assert rm.snapshots[-1].metrics == {"holdout": {"ae": 1.02}}
    rm.set_snapshot_metrics({"holdout": {"ae": 0.99}})
    assert rm.snapshots[-1].metrics == {"holdout": {"ae": 0.99}}
    with pytest.raises(ValueError):
        rm.set_snapshot_metrics({}, version=9)
    rm.to_json(tmp_path / "m.easyglm")
    back = RateModel.from_json(tmp_path / "m.easyglm")
    assert back.snapshots[-1].metrics == {"holdout": {"ae": 0.99}}


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
    @staticmethod
    def _fit(df):
        train = df.filter(pl.col("traintest") == 1)
        spec = DesignSpec.from_data(train, ["VehAge", "Region", "DrivAge"])
        return fit_glm(
            train,
            spec,
            "ClaimNb",
            family="poisson",
            weight_col="Exposure",
            divide_target_by_weight=True,
            alpha=0.01,
        )

    def test_from_rate_tables_matches_to_rate_model(self, synthetic_insurance_data):
        df = synthetic_insurance_data
        fit = self._fit(df)
        rm_tables = RateModel.from_rate_tables(rate_tables(fit), base_rate(fit))
        rm_exact = to_rate_model(fit)

        assert set(rm_tables.variables) == {"VehAge", "DrivAge", "Region"}
        assert rm_tables.variables["VehAge"].type == "numeric"
        assert rm_tables.variables["Region"].type == "categorical"
        assert rm_tables.current_version == 1
        scoring = df.with_columns(
            pl.when(pl.arange(0, df.height) % 7 == 0)
            .then(None)
            .otherwise(pl.col("VehAge"))
            .alias("VehAge"),
            pl.when(pl.arange(0, df.height) % 11 == 0)
            .then(pl.lit("Mars"))
            .otherwise(pl.col("Region"))
            .alias("Region"),
        )
        np.testing.assert_allclose(
            rm_tables.predict(scoring, exposure_col=None),
            rm_exact.predict(scoring, exposure_col=None),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            rm_tables.predict(scoring, exposure_col=None),
            fit.predict(scoring),
            rtol=1e-10,
        )

    def test_roundtrip_after_full_pipeline(self, synthetic_insurance_data, tmp_path):
        df = synthetic_insurance_data
        rm = RateModel.from_rate_tables(rate_tables(self._fit(df)), 0.05)
        data = pl.DataFrame({"VehAge": [5], "DrivAge": [30], "Region": ["North"]})
        before = rm.predict(data)
        rm.to_json(tmp_path / "model.easyglm")
        after = RateModel.from_json(tmp_path / "model.easyglm").predict(data)
        np.testing.assert_array_almost_equal(before, after)

    def test_from_glm_model(self, synthetic_insurance_data):
        df = synthetic_insurance_data
        fit = self._fit(df)
        rm = RateModel.from_glm_model(
            fit, exposure_col="Exposure", train_test_col="traintest"
        )
        assert set(rm.variables) == {"VehAge", "DrivAge", "Region"}
        assert rm.metadata.model_type == "poisson"
        assert rm.metadata.target == "ClaimNb"
        assert rm.metadata.exposure_col == "Exposure"
        assert rm.current_version == 1
        np.testing.assert_allclose(
            rm.predict(df.head(20), exposure_col=None),
            fit.predict(df.head(20)),
            rtol=1e-10,
        )


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
        rate_table = _numeric_table([0, 5, 10], [0.85, 0.85, 0.90, 1.00])

        rm = RateModel.from_rate_tables(
            {"VehAge": rate_table},
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
        rate_table = _numeric_table([0, 5, 10], [0.85, 0.85, 0.90, 1.00])
        rm = RateModel.from_rate_tables(
            {"VehAge": rate_table},
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
