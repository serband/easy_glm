"""Piece C1 — foundations: format versions, strict dispatch, offsets in the
RateModel, snapshot diff, editor argument order, adjusted-table exports and
tolerant project loading."""

from __future__ import annotations

import json
import math
import warnings
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from easy_glm import DesignSpec, fit_glm, to_rate_model
from easy_glm.engine import FromToRow, RateModel, VariableConfig
from easy_glm.engine.rate_model import FORMAT_VERSION
from easy_glm.workflow import Project, to_script
from easy_glm.workflow.project import PROJECT_VERSION


def _frame(n: int = 3000, seed: int = 9) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n).astype(float)
    power = rng.integers(4, 12, n)
    region = rng.choice(["R1", "R2", "R3"], n).astype(object)
    expo = rng.uniform(0.2, 1.0, n)
    mu = np.exp(-2.0 + 0.01 * (age - 40) + np.where(region == "R1", 0.0, 0.2))
    return pl.DataFrame(
        {
            "ClaimNb": rng.poisson(mu * expo).astype(float),
            "Exposure": expo,
            "DrivAge": age,
            "VehPower": power,
            "Region": region,
            "logprem": np.log(rng.uniform(100, 900, n)),
        }
    )


@pytest.fixture(scope="module")
def fit_and_rm():
    df = _frame()
    spec = DesignSpec.from_data(
        df, ["DrivAge", "VehPower", "Region"], categorical=["VehPower"]
    )
    fit = fit_glm(
        df,
        spec,
        "ClaimNb",
        family="poisson",
        weight_col="Exposure",
        divide_target_by_weight=True,
        offset_col="logprem",
        alpha=0.003,
    )
    return df, fit, to_rate_model(fit, exposure_col="Exposure")


# --------------------------------------------------------------------------
# format version and dispatch
# --------------------------------------------------------------------------
class TestFormatVersion:
    def test_writes_current_version_and_metadata(self, fit_and_rm, tmp_path):
        _, fit, rm = fit_and_rm
        rm.to_json(tmp_path / "m.easyglm")
        raw = json.loads((tmp_path / "m.easyglm").read_text())
        assert raw["format_version"] == FORMAT_VERSION == 2
        meta = raw["metadata"]
        assert meta["offset_col"] == "logprem" and meta["offset_is_log"] is True
        assert meta["link"] == "log" and meta["divide_target_by_weight"] is True

    def test_newer_version_is_rejected(self, fit_and_rm):
        _, _, rm = fit_and_rm
        raw = rm._to_dict()
        raw["format_version"] = FORMAT_VERSION + 1
        with pytest.raises(ValueError, match="format version 3"):
            RateModel._from_dict(raw)

    def test_versionless_0_3_file_loads_and_scores_identically(self, fit_and_rm):
        df, fit, rm = fit_and_rm
        # a 0.3 file: no format_version, no offset/link/target flags
        rm_no_offset = to_rate_model(
            fit_glm(
                df,
                fit.spec,
                "ClaimNb",
                family="poisson",
                weight_col="Exposure",
                divide_target_by_weight=True,
                alpha=0.003,
            )
        )
        raw = rm_no_offset._to_dict()
        raw.pop("format_version")
        for k in ("offset_col", "offset_is_log", "link", "divide_target_by_weight"):
            raw["metadata"].pop(k)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # migration must be silent
            legacy = RateModel._from_dict(raw)
        assert legacy.metadata.offset_col is None
        assert legacy.metadata.divide_target_by_weight is None  # unknown for old files
        np.testing.assert_array_equal(legacy.predict(df), rm_no_offset.predict(df))
        assert legacy._to_dict()["format_version"] == FORMAT_VERSION

    def test_unknown_metadata_keys_are_ignored(self, fit_and_rm):
        _, _, rm = fit_and_rm
        raw = rm._to_dict()
        raw["metadata"]["from_the_future"] = 42
        back = RateModel._from_dict(raw)
        assert back.metadata.target == rm.metadata.target

    def test_unknown_table_type_raises_on_predict(self):
        cfg = VariableConfig(type="spline", table=[FromToRow(None, None, 1.0)])
        rm = RateModel(base_rate=1.0, variables={"x": cfg})
        with pytest.raises(ValueError, match="spline"):
            rm.predict(pl.DataFrame({"x": [1.0, 2.0]}))


# --------------------------------------------------------------------------
# offsets
# --------------------------------------------------------------------------
class TestOffsets:
    def test_metadata_copied_and_applied(self, fit_and_rm):
        df, fit, rm = fit_and_rm
        assert rm.metadata.offset_col == "logprem"
        np.testing.assert_allclose(
            rm.predict(df, exposure_col=None), fit.predict(df), rtol=1e-10
        )
        # exposure and offset compose
        np.testing.assert_allclose(
            rm.predict(df), fit.predict(df) * df["Exposure"].to_numpy(), rtol=1e-10
        )

    def test_missing_offset_column_warns(self, fit_and_rm):
        df, fit, rm = fit_and_rm
        with pytest.warns(UserWarning, match="Offset column"):
            out = rm.predict(df.drop("logprem"), exposure_col=None)
        np.testing.assert_allclose(
            out, fit.predict(df) / np.exp(df["logprem"].to_numpy()), rtol=1e-10
        )

    def test_raw_offset_scale(self, fit_and_rm):
        df, fit, rm = fit_and_rm
        raw = rm._to_dict()
        raw["metadata"]["offset_is_log"] = False
        raw["metadata"]["offset_col"] = "Exposure"
        rm2 = RateModel._from_dict(raw)
        base = rm.predict(
            df.with_columns(pl.lit(0.0).alias("logprem")), exposure_col=None
        )
        np.testing.assert_allclose(
            rm2.predict(df, exposure_col=None),
            base * df["Exposure"].to_numpy(),
            rtol=1e-10,
        )


# --------------------------------------------------------------------------
# snapshot diff, exports
# --------------------------------------------------------------------------
class TestDiffAndExports:
    def test_diff_between_snapshots(self, fit_and_rm):
        _, _, rm = fit_and_rm
        rm = rm.clone()
        row = rm.variables["Region"].table[0]
        rm.update_relativity("Region", row.from_, row.to_, row.relativity * 2)
        v2 = rm.create_snapshot("doubled")
        changes = rm.diff(1, v2)
        assert len(changes) == 1
        c = changes[0]
        assert c.variable == "Region" and c.from_ == row.from_
        assert c.new_relativity == pytest.approx(2 * c.old_relativity)
        assert rm.diff(v2, v2) == []
        with pytest.raises(ValueError):
            rm.diff(0, 1)

    def test_excel_bytes_reflect_adjustments(self, fit_and_rm, tmp_path):
        pytest.importorskip("streamlit")
        from easy_glm.app.ui import excel_bytes

        _, _, rm = fit_and_rm
        rm = rm.clone()
        row = rm.variables["Region"].table[0]
        rm.update_relativity("Region", row.from_, row.to_, 3.0)
        run = SimpleNamespace(name="freq", rate_model=rm)
        (tmp_path / "x.xlsx").write_bytes(excel_bytes(run))
        sheets = pl.read_excel(tmp_path / "x.xlsx", sheet_id=0)
        region = sheets["Region"]
        assert region["relativity"][0] == pytest.approx(3.0)
        assert "fitted" in region.columns and region["fitted"][0] != pytest.approx(3.0)

    def test_exported_script_writes_adjusted_tables(self):
        p = Project(name="x")
        p.data.source.path = "data.parquet"
        p.data.roles = {
            "ClaimNb": "target",
            "Exposure": "weight",
            "DrivAge": "predictor",
        }
        p.new_model("m", divide_target_by_weight=True)
        p.models["m"].penalty.alpha = 0.01
        src = to_script(p, "m")
        assert "rm.to_excel(" in src and "EasyGLM(" not in src


# --------------------------------------------------------------------------
# editor arguments
# --------------------------------------------------------------------------
def test_editor_args_put_streamlit_options_before_separator():
    from easy_glm.ui import editor_args

    args = editor_args(
        "m.easyglm", port=8765, data_path="d.parquet", formula="sum_over_weight"
    )
    assert args.index("--server.port") < args.index("--")
    assert args[args.index("--server.port") + 1] == "8765"
    after = args[args.index("--") + 1 :]
    assert "--model-path=m.easyglm" in after and "--data-path=d.parquet" in after
    assert "--formula=sum_over_weight" in after


# --------------------------------------------------------------------------
# project format
# --------------------------------------------------------------------------
class TestProjectFormat:
    def _v1(self) -> dict:
        p = Project(name="old")
        p.data.roles = {"y": "target", "x": "predictor"}
        p.new_model("m")
        raw = p.to_dict()
        raw["version"] = 1
        return raw

    def test_v1_project_loads_and_is_migrated(self):
        back = Project.from_dict(self._v1())
        assert back.version == PROJECT_VERSION == 2
        assert back.models["m"].target == "y"

    def test_unknown_keys_warn_but_load(self):
        raw = self._v1()
        raw["models"]["m"]["shiny_new_option"] = True
        raw["design"]["variables"]["x"] = {"kind": "step", "future_knob": 3}
        raw["data"]["split"]["something"] = 1
        with pytest.warns(UserWarning, match="unknown project keys"):
            back = Project.from_dict(raw)
        assert back.design.variables["x"].kind == "step"

    def test_newer_project_is_rejected(self):
        with pytest.raises(ValueError, match="newer"):
            Project.from_dict({"version": PROJECT_VERSION + 1})

    def test_versionless_dict_is_treated_as_v1(self):
        raw = self._v1()
        raw.pop("version")
        assert Project.from_dict(raw).version == PROJECT_VERSION


def test_nan_old_relativity_for_new_rows(fit_and_rm):
    _, _, rm = fit_and_rm
    rm = rm.clone()
    rm.snapshots[0].relativities["Region"] = rm.snapshots[0].relativities["Region"][:-1]
    v2 = rm.create_snapshot("with extra row")
    extra = [c for c in rm.diff(1, v2) if math.isnan(c.old_relativity)]
    assert len(extra) == 1


# --------------------------------------------------------------------------
# review follow-ups (docs/reviews/c1-foundations.md §3 and §5)
# --------------------------------------------------------------------------
FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


class TestReal030Fixture:
    """A .easyglm written by the v0.3.0 tag (two snapshots, one adjustment)."""

    def test_loads_silently_and_scores_bitwise_identically(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rm = RateModel.from_json(FIXTURES / "v0_3_0_model.easyglm")
        score = pl.read_parquet(FIXTURES / "v0_3_0_scoring.parquet")
        saved = pl.read_parquet(FIXTURES / "v0_3_0_predictions.parquet")
        np.testing.assert_array_equal(
            rm.predict(score, exposure_col=None), saved["v2"].to_numpy()
        )
        np.testing.assert_array_equal(
            rm.predict(score, version=1, exposure_col=None), saved["v1"].to_numpy()
        )
        np.testing.assert_array_equal(
            rm.predict(score), saved["with_exposure"].to_numpy()
        )
        assert len(rm.snapshots) == 2 and rm.metadata.divide_target_by_weight is None
        assert rm._to_dict()["format_version"] == FORMAT_VERSION


def test_unknown_table_type_rejected_at_load(fit_and_rm):
    _, _, rm = fit_and_rm
    raw = rm._to_dict()
    raw["variables"]["Region"]["type"] = "spline"
    with pytest.raises(ValueError, match="spline"):
        RateModel._from_dict(raw)


def test_null_format_version_is_treated_as_v1(fit_and_rm):
    _, _, rm = fit_and_rm
    raw = rm._to_dict()
    raw["format_version"] = None
    assert RateModel._from_dict(raw)._to_dict()["format_version"] == FORMAT_VERSION


def test_offset_nulls_propagate_identically(fit_and_rm):
    df, fit, rm = fit_and_rm
    frame = df.head(40).with_columns(
        pl.when(pl.arange(0, 40) % 5 == 0)
        .then(None)
        .otherwise(pl.col("logprem"))
        .alias("logprem")
    )
    a = rm.predict(frame, exposure_col=None)
    b = fit.predict(frame)
    assert np.isnan(a).sum() == 8
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=1e-10)


def test_glm_fit_predict_warns_when_offset_missing(fit_and_rm):
    df, fit, _ = fit_and_rm
    with pytest.warns(UserWarning, match="Offset column"):
        fit.predict(df.drop("logprem").head(5))


def test_categorical_dtype_mismatch_warns(fit_and_rm):
    df, _, rm = fit_and_rm
    as_float = df.head(200).with_columns(pl.col("VehPower").cast(pl.Float64))
    with pytest.warns(UserWarning, match="VehPower"):
        rm.predict(as_float, exposure_col=None)


def test_rate_model_tables_without_snapshots():
    from easy_glm.core.excel import rate_model_tables

    cfg = VariableConfig(
        type="numeric",
        table=[
            FromToRow(None, 30.0, 1.1),
            FromToRow(30.0, None, 0.9),
            FromToRow(None, None, 1.0),
        ],
    )
    rm = RateModel(base_rate=1.0, variables={"x": cfg})
    t = rate_model_tables(rm)["x"]
    assert "fitted" not in t.columns and t.height == 3


class TestProjectUnknownKeysEverywhere:
    def _raw(self) -> dict:
        p = Project(name="p")
        p.data.roles = {"y": "target", "x": "predictor"}
        p.new_model("m")
        return p.to_dict()

    @pytest.mark.parametrize("where", ["top", "data", "design", "adjustment"])
    def test_warns(self, where):
        raw = self._raw()
        if where == "top":
            raw["runs"] = {}
        elif where == "data":
            raw["data"]["future_data_key"] = 1
        elif where == "design":
            raw["design"]["future_design_key"] = 1
        else:
            raw["models"]["m"]["adjustments"] = [
                {
                    "variable": "x",
                    "from": None,
                    "to": 1.0,
                    "relativity": 1.1,
                    "note": "hi",
                }
            ]
        with pytest.warns(UserWarning, match="unknown project keys"):
            Project.from_dict(raw)


class TestEditorDefaultFormula:
    """D7: the A/E default derived from metadata gives overall train A/E = 1."""

    @staticmethod
    def _count_model(exposure_col):
        df = _frame(seed=21)
        spec = DesignSpec.from_data(df, ["DrivAge", "Region"])
        fit = fit_glm(
            df,
            spec,
            "ClaimNb",
            family="poisson",
            weight_col="Exposure",
            divide_target_by_weight=True,
            alpha=0.001,
        )
        return df, to_rate_model(fit, exposure_col=exposure_col)

    @pytest.mark.parametrize("exposure_col", ["Exposure", None])
    def test_overall_train_ae_is_one(self, exposure_col):
        from easy_glm.ui.metrics import compute_actual_expected, default_formula

        df, rm = self._count_model(exposure_col)
        formula = default_formula(rm.metadata)
        assert formula == "sum_over_weight"
        res = compute_actual_expected(rm, df, "Region", formula=formula)["subsets"][
            "all"
        ]
        actual = sum(r["actual"] * r["exposure"] for r in res)
        expected = sum(r["expected"] * r["exposure"] for r in res)
        # A Poisson GLM with intercept reproduces total claims on its training
        # data only up to the solver's stopping rule: the reviewer measured 3e-5
        # at glum's default gradient_tol, so 1e-4 is the honest bound here.
        assert actual / expected == pytest.approx(1.0, abs=1e-4)

    def test_rate_target_and_unknown_flag_use_weighted_mean(self):
        from easy_glm.engine.models import ModelMetadata
        from easy_glm.ui.metrics import default_formula

        assert (
            default_formula(
                ModelMetadata(divide_target_by_weight=False, weight_col="w")
            )
            == "sum_weighted"
        )
        assert (
            default_formula(ModelMetadata(divide_target_by_weight=None, weight_col="w"))
            == "sum_weighted"
        )
        assert default_formula(None) == "sum_weighted"
