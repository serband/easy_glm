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
        cfg = VariableConfig(type="interaction", table=[FromToRow(None, None, 1.0)])
        rm = RateModel(base_rate=1.0, variables={"x": cfg})
        with pytest.raises(ValueError, match="interaction"):
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
