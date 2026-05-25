from __future__ import annotations

import polars as pl
import pytest

from easy_glm.engine import FromToRow, RateModel, VariableConfig


@pytest.fixture
def rm_with_metadata() -> RateModel:
    num_table = [
        FromToRow(from_=None, to_=18, relativity=1.45),
        FromToRow(from_=18, to_=23, relativity=1.30),
        FromToRow(from_=23, to_=28, relativity=1.15),
        FromToRow(from_=28, to_=33, relativity=1.00),
        FromToRow(from_=33, to_=None, relativity=0.90),
    ]
    variables = {"DrivAge": VariableConfig(type="numeric", table=num_table)}
    rm = RateModel(base_rate=0.1, variables=variables)
    rm.metadata.model_type = "poisson"
    rm.metadata.target = "ClaimNb"
    rm.metadata.weight_col = "Exposure"
    rm.metadata.train_test_col = "traintest"
    rm.create_snapshot("Base")
    return rm


@pytest.fixture
def insurance_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "DrivAge": [20, 25, 30, 25, 20, 35, 30, 30],
            "Exposure": [1.0, 0.5, 1.0, 0.8, 1.2, 0.9, 1.0, 0.7],
            "ClaimNb": [0, 1, 0, 0, 2, 1, 1, 0],
            "traintest": [1, 1, 1, 1, 0, 0, 0, 0],
        }
    )


class TestMetricsModule:
    def test_compute_actual_expected_basic(self, rm_with_metadata, insurance_data):
        from easy_glm.ui.metrics import compute_actual_expected

        result = compute_actual_expected(
            rm_with_metadata, insurance_data, "DrivAge", formula="sum_weighted"
        )

        assert "subsets" in result
        assert "train" in result["subsets"]
        assert "test" in result["subsets"]
        assert result["variable"] == "DrivAge"

        train = result["subsets"]["train"]
        assert len(train) > 0
        for bucket in train:
            assert "level" in bucket
            assert "actual" in bucket
            assert "expected" in bucket
            assert "exposure" in bucket
            assert isinstance(bucket["exposure"], float)

    def test_compute_actual_expected_all_data(self, rm_with_metadata):
        data = pl.DataFrame(
            {
                "DrivAge": [20, 25, 30],
                "Exposure": [1.0, 1.0, 1.0],
                "ClaimNb": [1, 2, 3],
            }
        )
        rm_no_split = RateModel(base_rate=0.1, variables=rm_with_metadata.variables)
        rm_no_split.metadata.target = "ClaimNb"
        rm_no_split.metadata.weight_col = "Exposure"
        rm_no_split.create_snapshot("Base")

        from easy_glm.ui.metrics import compute_actual_expected

        result = compute_actual_expected(
            rm_no_split, data, "DrivAge", formula="sum_weighted"
        )

        assert "all" in result["subsets"]
        assert "train" not in result["subsets"]

    def test_compute_actual_expected_missing_target(self, rm_with_metadata):
        data = pl.DataFrame({"DrivAge": [20, 25], "Exposure": [1.0, 1.0]})
        rm_no_target = RateModel(base_rate=0.1, variables=rm_with_metadata.variables)
        rm_no_target.create_snapshot("Base")

        from easy_glm.ui.metrics import compute_actual_expected

        with pytest.raises(ValueError, match="missing 'target'"):
            compute_actual_expected(rm_no_target, data, "DrivAge")

    def test_compute_actual_expected_target_not_in_data(self, rm_with_metadata):
        data = pl.DataFrame({"DrivAge": [20, 25]})

        from easy_glm.ui.metrics import compute_actual_expected

        with pytest.raises(ValueError, match="not found in data"):
            compute_actual_expected(rm_with_metadata, data, "DrivAge")

    def test_unweighted_formula(self, rm_with_metadata):
        data = pl.DataFrame(
            {
                "DrivAge": [20, 25, 30],
                "Exposure": [1.0, 1.0, 1.0],
                "ClaimNb": [1, 2, 3],
            }
        )

        from easy_glm.ui.metrics import compute_actual_expected

        result = compute_actual_expected(
            rm_with_metadata, data, "DrivAge", formula="sum_unweighted"
        )
        buckets = result["subsets"]["all"]
        for b in buckets:
            assert b["actual"] >= 0
            assert "exposure" in b

    def test_sum_over_weight_formula(self, rm_with_metadata):
        data = pl.DataFrame(
            {
                "DrivAge": [20, 25, 30],
                "Exposure": [1.0, 2.0, 3.0],
                "ClaimNb": [1, 2, 3],
            }
        )

        from easy_glm.ui.metrics import compute_actual_expected

        result = compute_actual_expected(
            rm_with_metadata, data, "DrivAge", formula="sum_over_weight"
        )
        buckets = result["subsets"]["all"]
        assert len(buckets) > 0
        for b in buckets:
            assert "exposure" in b


class TestChartsModule:
    def test_build_histogram_numeric(self, insurance_data):
        from easy_glm.ui.charts import build_histogram

        fig = build_histogram(insurance_data, "DrivAge")
        assert fig is not None
        assert len(fig.data) >= 1

    def test_build_histogram_categorical(self):
        data = pl.DataFrame({"Region": ["North", "South", "North", "Urban", "Urban"]})
        from easy_glm.ui.charts import build_histogram

        fig = build_histogram(data, "Region")
        assert fig is not None

    def test_build_relativity_chart_numeric(self, rm_with_metadata):
        from easy_glm.ui.charts import build_relativity_chart

        config = rm_with_metadata.variables["DrivAge"]
        fig = build_relativity_chart(config, "DrivAge")
        assert fig is not None
        assert len(fig.data) >= 1

    def test_build_actual_vs_expected(self, rm_with_metadata, insurance_data):
        from easy_glm.ui.charts import build_actual_vs_expected
        from easy_glm.ui.metrics import compute_actual_expected

        metrics = compute_actual_expected(
            rm_with_metadata, insurance_data, "DrivAge", formula="sum_weighted"
        )
        fig = build_actual_vs_expected(metrics, "DrivAge")
        assert fig is not None
        assert len(fig.data) > 0
