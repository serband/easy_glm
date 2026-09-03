"""D5 — relativity tooling: smooth, cap / floor, round, undo / redo, snapshots.

Three layers:

* the engine (:mod:`easy_glm.engine.tooling`) on hand-built tables, where every
  number can be checked by hand: the weighted log mean a smoothing must
  preserve, the idempotence of cap / floor and round, the monotone result of the
  isotonic fit, the null row nothing touches, and the node rule of a
  piecewise-linear table;
* the plumbing that gives the tools their weights — training exposure per band,
  from the fit through the tables into the RateModel and back out of JSON;
* the workbench: applying a tool writes ordinary adjustments, the tables stay
  exact afterwards, and undo / redo / snapshot / snapshot-diff work through
  AppTest.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from easy_glm.core.design import DesignSpec
from easy_glm.core.excel import rate_model_tables
from easy_glm.core.fit import fit_glm
from easy_glm.core.tables import to_rate_model
from easy_glm.engine import tooling
from easy_glm.engine.models import BandRow, FromToRow, VariableConfig
from easy_glm.engine.rate_model import RateModel
from easy_glm.workflow import (
    Adjustment,
    AdjustmentError,
    Project,
    TableSnapshot,
    VariableDesign,
    apply_adjustments,
    expected_claims,
    missing_variables,
    rate_model_diff,
    rate_model_for,
    rebalance_override,
    rebuild_rate_model,
    run_model,
)

TOL = 1e-12


# --------------------------------------------------------------------------
# hand-built tables
# --------------------------------------------------------------------------
def step_table() -> VariableConfig:
    """Five bins with very different exposure and a null row that is out of
    line with every band, so anything that touched it would show."""
    rows = [
        FromToRow(None, 25.0, 1.60, 200.0),
        FromToRow(25.0, 35.0, 0.80, 1200.0),
        FromToRow(35.0, 45.0, 1.15, 900.0),
        FromToRow(45.0, 60.0, 0.95, 700.0),
        FromToRow(60.0, None, 1.30, 100.0),
        FromToRow(None, None, 2.50, 40.0),
    ]
    return VariableConfig(type="numeric", table=rows)


def categorical_table() -> VariableConfig:
    rows = [
        FromToRow("small", "small", 0.90, 500.0),
        FromToRow("medium", "medium", 1.30, 300.0),
        FromToRow("large", "large", 1.05, 200.0),
        FromToRow(None, None, 1.75, 25.0),
    ]
    return VariableConfig(type="categorical", table=rows)


def linear_table() -> VariableConfig:
    """A clamped curve on [10, 40] with knots at 20 and 30: rows are
    ``(None, 10)``, three bands and ``(40, None)``, plus the null row."""
    rows = [
        BandRow(None, 10.0, 1.20, 0.0, 50.0),
        BandRow(10.0, 20.0, 1.20, 0.0, 600.0),
        BandRow(20.0, 30.0, 0.80, 0.0, 900.0),
        BandRow(30.0, 40.0, 1.10, 0.0, 300.0),
        BandRow(40.0, None, 0.95, 0.0, 80.0),
        BandRow(None, None, 1.40, 0.0, 30.0),
    ]
    cfg = VariableConfig(type="linear", table=rows, x_base=20.0)
    return tooling.apply_values(cfg, [r.relativity for r in rows])  # derives the slopes


def hand_log_mean(cfg: VariableConfig) -> float:
    """The weighted mean of the log relativities, written out longhand."""
    pairs = [
        (float(cfg.table[g[0]].relativity), sum(cfg.table[i].exposure for i in g))
        for g in tooling.groups(cfg)
    ]
    total = sum(w for _v, w in pairs)
    return sum(w * np.log(v) for v, w in pairs) / total


class TestWeightsAndGroups:
    def test_groups_leave_out_the_null_row_and_pair_the_linear_node(self):
        assert tooling.groups(step_table()) == [[0], [1], [2], [3], [4]]
        assert tooling.groups(categorical_table()) == [[0], [1], [2]]
        assert tooling.groups(linear_table()) == [[0, 1], [2], [3], [4]]

    def test_group_weights_are_the_band_exposure(self):
        w, uniform = tooling.group_weights(step_table())
        assert not uniform
        assert list(w) == [200.0, 1200.0, 900.0, 700.0, 100.0]
        # the linear node at the lower clamp owns both rows that show it
        wl, _ = tooling.group_weights(linear_table())
        assert list(wl) == [650.0, 900.0, 300.0, 80.0]

    def test_no_exposure_falls_back_to_equal_weights(self):
        cfg = step_table()
        for row in cfg.table:
            row.exposure = 0.0
        w, uniform = tooling.group_weights(cfg)
        assert uniform and list(w) == [1.0] * 5
        result = tooling.smooth_moving_average(cfg, "x")
        assert result.uniform_weights
        assert abs(result.log_mean_after - result.log_mean_before) < TOL

    def test_weighted_log_mean_matches_the_longhand_sum(self):
        cfg = step_table()
        assert tooling.weighted_log_mean(cfg) == pytest.approx(
            hand_log_mean(cfg), abs=TOL
        )


# --------------------------------------------------------------------------
# smoothing
# --------------------------------------------------------------------------
class TestSmoothing:
    @pytest.mark.parametrize("window", [3, 5, 7])
    def test_moving_average_preserves_the_weighted_log_mean(self, window):
        cfg = step_table()
        r = tooling.smooth_moving_average(cfg, "DrivAge", window=window)
        assert abs(r.log_mean_after - r.log_mean_before) < TOL
        assert abs(r.level_shift) < TOL
        # and it really did smooth: the biggest jump between bands shrinks
        before = np.diff(np.log([1.60, 0.80, 1.15, 0.95, 1.30]))
        after = np.diff(np.log(r.values[:5]))
        assert np.abs(after).max() < np.abs(before).max()

    def test_moving_average_never_touches_the_null_row(self):
        cfg = step_table()
        r = tooling.smooth_moving_average(cfg, "DrivAge")
        assert r.values[-1] == 2.50
        assert "Other / Unknown" not in r.changed_labels

    def test_moving_average_of_a_flat_table_changes_nothing(self):
        cfg = step_table()
        flat = tooling.apply_values(cfg, [1.1, 1.1, 1.1, 1.1, 1.1, 2.5])
        r = tooling.smooth_moving_average(flat, "DrivAge")
        assert r.changed == 0
        assert r.values == pytest.approx([1.1, 1.1, 1.1, 1.1, 1.1, 2.5], abs=TOL)

    def test_an_even_or_tiny_window_is_refused(self):
        cfg = step_table()
        for window in (2, 4, 1, 0):
            with pytest.raises(tooling.ToolingError, match="odd window"):
                tooling.smooth_moving_average(cfg, "DrivAge", window=window)

    def test_a_window_wider_than_the_table_still_works(self):
        cfg = step_table()
        r = tooling.smooth_moving_average(cfg, "DrivAge", window=25)
        assert abs(r.log_mean_after - r.log_mean_before) < TOL
        assert len(set(np.round(r.values[:5], 9))) == 1  # every band the same

    @pytest.mark.parametrize("direction", ["increasing", "decreasing"])
    def test_isotonic_is_monotone_and_keeps_the_level(self, direction):
        cfg = step_table()
        r = tooling.smooth_isotonic(cfg, "DrivAge", direction=direction)
        values = np.array(r.values[:5])
        steps = np.diff(np.log(values))
        if direction == "increasing":
            assert (steps >= -TOL).all()
        else:
            assert (steps <= TOL).all()
        assert abs(r.log_mean_after - r.log_mean_before) < TOL
        assert r.values[-1] == 2.50  # the null row again

    def test_isotonic_pools_the_bands_that_break_the_direction(self):
        cfg = tooling.apply_values(step_table(), [1.0, 1.0, 0.5, 1.0, 1.0, 2.5])
        r = tooling.smooth_isotonic(cfg, "DrivAge", direction="increasing")
        # the dip is pooled with its neighbours, the two ends stay put
        assert r.values[0] == pytest.approx(r.values[1], abs=TOL)
        assert r.values[1] < r.values[3]
        assert abs(r.log_mean_after - r.log_mean_before) < TOL

    def test_isotonic_handles_a_band_with_no_exposure(self):
        cfg = step_table()
        cfg.table[2].exposure = 0.0  # a band no policy reached
        r = tooling.smooth_isotonic(cfg, "DrivAge", direction="increasing")
        assert (np.diff(np.log(r.values[:5])) >= -TOL).all()
        assert abs(r.log_mean_after - r.log_mean_before) < TOL

    def test_a_categorical_is_refused_until_the_order_is_confirmed(self):
        cfg = categorical_table()
        for tool in (tooling.smooth_moving_average, tooling.smooth_isotonic):
            with pytest.raises(tooling.ToolingError, match="not an order of the risk"):
                tool(cfg, "Size")
        r = tooling.smooth_moving_average(cfg, "Size", ordered=True)
        assert abs(r.log_mean_after - r.log_mean_before) < TOL
        assert r.values[-1] == 1.75  # the Other row is never smoothed

    def test_a_relativity_of_zero_is_refused_rather_than_logged(self):
        cfg = tooling.apply_values(step_table(), [1.0, 0.0, 1.0, 1.0, 1.0, 2.5])
        with pytest.raises(tooling.ToolingError, match="has no logarithm"):
            tooling.smooth_moving_average(cfg, "DrivAge")

    def test_smoothing_needs_more_than_one_band(self):
        cfg = VariableConfig(
            type="categorical",
            table=[FromToRow("a", "a", 1.0, 5.0), FromToRow(None, None, 1.0, 1.0)],
        )
        with pytest.raises(tooling.ToolingError, match="nothing to smooth"):
            tooling.smooth_moving_average(cfg, "one", ordered=True)


# --------------------------------------------------------------------------
# cap / floor and round
# --------------------------------------------------------------------------
class TestCapFloorAndRound:
    def test_cap_and_floor_clamp_and_are_idempotent(self):
        cfg = step_table()
        r = tooling.cap_floor(cfg, "DrivAge", floor=0.9, cap=1.2)
        assert r.values[:5] == pytest.approx([1.2, 0.9, 1.15, 0.95, 1.2], abs=TOL)
        assert r.values[-1] == 2.50  # the null row is not clamped
        again = tooling.cap_floor(
            tooling.apply_values(cfg, r.values), "DrivAge", floor=0.9, cap=1.2
        )
        assert again.values == pytest.approx(r.values, abs=TOL)
        assert again.changed == 0

    def test_a_cap_moves_the_level_and_says_so(self):
        r = tooling.cap_floor(step_table(), "DrivAge", cap=1.0)
        assert r.level_shift < 0
        assert r.log_mean_after < r.log_mean_before

    def test_cap_floor_arguments_are_checked(self):
        cfg = step_table()
        with pytest.raises(tooling.ToolingError, match="floor, a cap, or both"):
            tooling.cap_floor(cfg, "DrivAge")
        with pytest.raises(tooling.ToolingError, match="above the cap"):
            tooling.cap_floor(cfg, "DrivAge", floor=1.5, cap=1.0)
        with pytest.raises(tooling.ToolingError, match="floor must be above 0"):
            tooling.cap_floor(cfg, "DrivAge", floor=0.0)
        with pytest.raises(tooling.ToolingError, match="cap must be above 0"):
            tooling.cap_floor(cfg, "DrivAge", cap=-1.0)

    @pytest.mark.parametrize(
        "kwargs", [{"decimals": 1}, {"decimals": 2}, {"step": 0.05}]
    )
    def test_rounding_is_idempotent(self, kwargs):
        cfg = step_table()
        r = tooling.round_relativities(cfg, "DrivAge", **kwargs)
        again = tooling.round_relativities(
            tooling.apply_values(cfg, r.values), "DrivAge", **kwargs
        )
        assert again.values == pytest.approx(r.values, abs=TOL)
        assert again.changed == 0
        assert r.values[-1] == 2.50

    def test_rounding_to_a_step_lands_on_the_step(self):
        cfg = tooling.apply_values(step_table(), [1.083, 1.02, 0.976, 1.0, 1.3, 2.5])
        r = tooling.round_relativities(cfg, "DrivAge", step=0.05)
        assert r.values[:5] == pytest.approx([1.1, 1.0, 1.0, 1.0, 1.3], abs=1e-9)

    def test_rounding_that_would_zero_a_relativity_is_refused(self):
        cfg = tooling.apply_values(step_table(), [0.04, 1.0, 1.0, 1.0, 1.0, 2.5])
        with pytest.raises(tooling.ToolingError, match="zero or less"):
            tooling.round_relativities(cfg, "DrivAge", decimals=1)

    def test_round_takes_decimals_or_a_step_but_not_both(self):
        cfg = step_table()
        with pytest.raises(tooling.ToolingError, match="not both"):
            tooling.round_relativities(cfg, "DrivAge", decimals=2, step=0.05)
        with pytest.raises(tooling.ToolingError, match="not both"):
            tooling.round_relativities(cfg, "DrivAge")


# --------------------------------------------------------------------------
# piecewise-linear tables
# --------------------------------------------------------------------------
class TestLinearTables:
    def test_the_two_rows_of_the_lower_node_move_together(self):
        cfg = linear_table()
        r = tooling.smooth_moving_average(cfg, "Density")
        assert r.values[0] == r.values[1]
        assert r.values[-1] == 1.40  # the null row

    def test_smoothing_keeps_the_curve_continuous(self):
        cfg = linear_table()
        r = tooling.smooth_moving_average(cfg, "Density")
        after = tooling.apply_values(cfg, r.values)
        bands = [b for b in after.table if b.from_ is not None and b.to_ is not None]
        for band, nxt in zip(bands, bands[1:], strict=False):
            assert band.relativity_to == pytest.approx(nxt.relativity, rel=1e-12)
        # the flat end rows keep slope 0 and the curve is log-linear inside a band
        assert after.table[0].slope == 0.0
        mid = bands[0]
        assert mid.relativity_at(15.0) == pytest.approx(
            mid.relativity * (mid.relativity_to / mid.relativity) ** 0.5, rel=1e-12
        )

    def test_the_level_check_uses_the_nodes(self):
        cfg = linear_table()
        r = tooling.smooth_isotonic(cfg, "Density", direction="increasing")
        assert abs(r.log_mean_after - r.log_mean_before) < TOL
        assert (np.diff(np.log(r.values[1:5])) >= -TOL).all()

    def test_a_cap_on_a_linear_table_clamps_the_nodes(self):
        cfg = linear_table()
        r = tooling.cap_floor(cfg, "Density", cap=1.0)
        assert r.values[:5] == pytest.approx([1.0, 1.0, 0.8, 1.0, 0.95], abs=TOL)


def test_an_interaction_has_no_tools():
    rows = [FromToRow(None, None, 1.0)]
    cfg = VariableConfig(type="interaction", table=rows, parents=("a", "b"))
    with pytest.raises(tooling.ToolingError, match="Edit the cells in the grid"):
        tooling.cap_floor(cfg, "a×b", cap=1.2)


def test_apply_values_does_not_touch_the_original():
    cfg = step_table()
    before = [r.relativity for r in cfg.table]
    tooling.apply_values(cfg, [2.0] * len(cfg.table))
    assert [r.relativity for r in cfg.table] == before


# --------------------------------------------------------------------------
# the exposure the tools weight by
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    rng = np.random.default_rng(5)
    n = 4000
    age = rng.integers(18, 80, n).astype(float)
    density = np.exp(rng.uniform(0, 8, n))
    region = rng.choice(["R1", "R2", "R3"], n, p=[0.6, 0.3, 0.1]).astype(object)
    exposure = rng.uniform(0.2, 1.0, n)
    mu = np.exp(
        -2.0
        - 0.02 * np.maximum(45 - age, 0)
        + 0.05 * np.log(density)
        + np.where(region == "R1", 0.0, 0.2)
    )
    age[rng.random(n) < 0.04] = np.nan
    return pl.DataFrame(
        {
            "IDpol": np.arange(n),
            "ClaimNb": rng.poisson(mu * exposure).astype(float),
            "Exposure": exposure,
            "DrivAge": age,
            "Density": density,
            "Region": region,
            "traintest": (rng.random(n) < 0.7).astype(int),
        }
    )


@pytest.fixture(scope="module")
def fitted(data):
    train = data.filter(pl.col("traintest") == 1)
    spec = DesignSpec.from_data(
        train, ["DrivAge", "Density", "Region"], linear=["Density"]
    )
    fit = fit_glm(
        train,
        spec,
        "ClaimNb",
        family="poisson",
        weight_col="Exposure",
        divide_target_by_weight=True,
        alpha=0.002,
    )
    return fit, train


class TestExposurePlumbing:
    def test_every_table_row_carries_its_training_exposure(self, fitted):
        fit, train = fitted
        rm = to_rate_model(fit)
        total = train["Exposure"].sum()
        for var in ("DrivAge", "Density", "Region"):
            rows = rm.variables[var].table
            assert sum(r.exposure for r in rows) == pytest.approx(total)
        # the null row of DrivAge holds exactly the missing ages' exposure
        missing = pl.col("DrivAge").is_null() | pl.col("DrivAge").is_nan()
        null_exposure = train.filter(missing)["Exposure"].sum()
        assert rm.variables["DrivAge"].table[-1].exposure == pytest.approx(
            null_exposure
        )
        assert null_exposure > 0

    def test_exposure_survives_json_snapshots_and_excel(self, fitted, tmp_path):
        fit, _train = fitted
        rm = to_rate_model(fit)
        before = [r.exposure for r in rm.variables["DrivAge"].table]
        rm.to_json(tmp_path / "m.easyglm")
        back = RateModel.from_json(tmp_path / "m.easyglm")
        assert [r.exposure for r in back.variables["DrivAge"].table] == before
        # a snapshot keeps it, and so does switching back to one
        rm.update_relativity("DrivAge", *_first_band(rm, "DrivAge"), 1.23)
        rm.create_snapshot("edited")
        rm.switch_to(1)
        assert [r.exposure for r in rm.variables["DrivAge"].table] == before
        path = rm.to_excel(tmp_path / "m.xlsx")
        tables = {
            name: pl.read_excel(path, sheet_name=name)
            for name in ("DrivAge", "Density")
        }
        rebuilt = RateModel.from_rate_tables(tables, rm.base_rate)
        assert [
            r.exposure for r in rebuilt.variables["DrivAge"].table
        ] == pytest.approx(before)

    def test_a_file_without_exposure_still_loads(self, fitted, tmp_path):
        fit, _train = fitted
        rm = to_rate_model(fit)
        raw = rm._to_dict()
        for var in raw["variables"].values():
            for row in var["table"]:
                row.pop("exposure", None)
        back = RateModel._from_dict(raw)
        assert all(r.exposure == 0.0 for r in back.variables["DrivAge"].table)
        result = tooling.smooth_moving_average(back.variables["DrivAge"], "DrivAge")
        assert result.uniform_weights


def _first_band(rm: RateModel, var: str):
    row = rm.variables[var].table[0]
    return row.from_, row.to_


# --------------------------------------------------------------------------
# the workbench: adjustments, exactness, snapshots
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def workspace(tmp_path_factory, data):
    folder = tmp_path_factory.mktemp("d5")
    path = folder / "policies.parquet"
    data.write_parquet(path)
    p = Project(name="d5test")
    p.data.source.type = "parquet"
    p.data.source.path = str(path)
    p.data.roles = {
        "IDpol": "id",
        "ClaimNb": "target",
        "Exposure": "weight",
        "DrivAge": "predictor",
        "Density": "predictor",
        "Region": "predictor",
        "traintest": "split",
    }
    p.data.split.mode = "column"
    p.data.split.column = "traintest"
    p.design.variables["Density"] = VariableDesign(kind="linear")
    p.new_model("freq", divide_target_by_weight=True)
    p.models["freq"].penalty.alpha = 0.002
    p.models["freq"].penalty.cv = None
    project_path = folder / "d5.easyglm-project.json"
    p.to_json(project_path)
    return {"folder": folder, "project": str(project_path), "data": str(path)}


def _lookup(table: pl.DataFrame, kind: str, value) -> float:
    """The relativity a rate table gives one value, read off the frame itself —
    the tables as a human would read them, not through the scorer."""
    rows = list(table.iter_rows(named=True))
    null_row = next((r for r in rows if r["from"] is None and r["to"] is None), None)
    body = [r for r in rows if r is not null_row]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float(null_row["relativity"])
    if kind == "categorical":
        for r in body:
            if str(r["from"]) == str(value):
                return float(r["relativity"])
        return float(null_row["relativity"])
    for r in body:
        lo, hi = r["from"], r["to"]
        if (lo is None or value >= lo) and (hi is None or value < hi):
            if kind == "linear" and lo is not None and hi is not None:
                span = (float(value) - lo) / (hi - lo)
                return (
                    float(r["relativity"])
                    * (float(r["relativity_to"]) / float(r["relativity"])) ** span
                )
            return float(r["relativity"])
    raise AssertionError(f"no band for {value!r}")


def test_a_smoothed_table_is_applied_as_adjustments_and_stays_exact(workspace, data):
    """The whole D5 path: smooth a curve, write it as ordinary band adjustments,
    rebuild the rate model without refitting, and check the model scores exactly
    what the tables say — including the smoothed piecewise-linear curve."""
    p = Project.from_json(workspace["project"])
    run = run_model(p, data, "freq")
    cfg = p.models["freq"]

    wanted: dict[str, list[float]] = {}
    for var in ("DrivAge", "Density"):
        var_cfg = run.rate_model.variables[var]
        result = tooling.smooth_moving_average(var_cfg, var, window=3)
        assert abs(result.log_mean_after - result.log_mean_before) < TOL
        wanted[var] = list(result.values)
        rows = var_cfg.table
        fitted = run.tables[var]["relativity"].to_list()
        for row, fit_value, new in zip(rows, fitted, result.values, strict=True):
            if abs(new - row.relativity) > 1e-12 and abs(new - fit_value) > 1e-12:
                assert new > 0
                cfg.adjustments.append(Adjustment(var, row.from_, row.to_, new))
    assert cfg.adjustments
    rebuild_rate_model(p, run, data)

    rm = run.rate_model
    tables = rate_model_tables(rm)
    # the smoothing landed in the tables, band for band
    for var, values in wanted.items():
        assert [r.relativity for r in rm.variables[var].table] == pytest.approx(
            values, abs=1e-12
        )
    sample = data.head(400)
    expected = np.full(sample.height, rm.base_rate)
    for var in ("DrivAge", "Density", "Region"):
        kind = rm.variables[var].type
        expected *= np.array(
            [_lookup(tables[var], kind, v) for v in sample[var].to_list()]
        )
    np.testing.assert_allclose(
        rm.predict(sample, exposure_col=None), expected, rtol=1e-12
    )


def test_a_snapshot_is_a_named_set_of_adjustments_that_survives_a_rebuild(
    workspace, data
):
    p = Project.from_json(workspace["project"])
    run = run_model(p, data, "freq")
    cfg = p.models["freq"]
    row = run.rate_model.variables["DrivAge"].table[0]

    cfg.snapshots.append(TableSnapshot("fitted", "now", []))
    cfg.adjustments = [Adjustment("DrivAge", row.from_, row.to_, 3.0)]
    rebuild_rate_model(p, run, data)
    cfg.snapshots.append(TableSnapshot("capped", "now", list(cfg.adjustments)))

    # the snapshot still means the same tables after a rebuild
    a = rate_model_for(p, run, cfg.snapshots[0].adjustments)
    b = rate_model_for(p, run, cfg.snapshots[1].adjustments)
    diff = rate_model_diff(a, b, tol=0.01)
    assert diff.height == 1
    assert diff["variable"][0] == "DrivAge"
    assert diff["relativity_b"][0] == pytest.approx(3.0)
    assert rate_model_diff(b, run.rate_model).is_empty()  # b is where we are
    # and the project file carries it
    path = workspace["folder"] / "with_snapshots.json"
    p.to_json(path)
    reloaded = Project.from_json(path)
    assert [s.name for s in reloaded.models["freq"].snapshots] == ["fitted", "capped"]
    assert reloaded.models["freq"].snapshots[1].adjustments[0].relativity == 3.0


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def wk(at, name: str) -> str:
    return f"{name}_{at.session_state['project_token']}"


def _script(project_path: str) -> str:
    return f"""
import importlib
import streamlit as st
from easy_glm.app import state as S
from easy_glm.workflow import Project

S.init_state()
if not st.session_state.get("_loaded"):
    S.set_project(Project.from_json({project_path!r}), None)
    st.session_state._loaded = True
if S.get_run("freq") is None:
    S.fit_model("freq")
importlib.import_module("easy_glm.app.pages_tables").render()
st.session_state["_project"] = S.project()
"""


@pytest.fixture(scope="module")
def page(workspace):
    at = AppTest.from_string(_script(workspace["project"]), default_timeout=240)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _button(at, label: str):
    found = [b for b in at.button if b.label == label]
    assert found, f"no {label!r} button among {[b.label for b in at.button]}"
    return found[0]


def _adjustments(at) -> list:
    return at.session_state["_project"].models["freq"].adjustments


class TestTablesPageTools:
    def test_apply_a_tool_then_undo_and_redo_it(self, workspace):
        at = AppTest.from_string(_script(workspace["project"]), default_timeout=240)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        assert not _adjustments(at)
        # the level check is on the page before anything is applied
        assert any("mean log relativity now" in m.label for m in at.metric)
        before = _relativities(at, "DrivAge")
        _button(at, "Apply to the table").click().run()
        assert not at.exception, [e.value for e in at.exception]
        applied = list(_adjustments(at))
        assert applied and all(a.variable == "DrivAge" for a in applied)
        assert any("applied to" in s.value for s in at.success)

        assert _relativities(at, "DrivAge") != before

        _button(at, "Undo").click().run()
        assert not at.exception
        assert not _adjustments(at)
        # undo restores the previous tables exactly, not approximately
        assert _relativities(at, "DrivAge") == before
        _button(at, "Redo").click().run()
        assert not at.exception
        assert [
            (a.variable, a.from_, a.to_, a.relativity) for a in _adjustments(at)
        ] == [(a.variable, a.from_, a.to_, a.relativity) for a in applied]
        # undo is bounded and never goes past the beginning
        _button(at, "Undo").click().run()
        assert not _adjustments(at)
        assert _button(at, "Undo").disabled

    def test_the_tool_selector_offers_each_operation_and_its_parameters(self, page):
        at = page
        tool = at.selectbox(key=wk(at, "tool_which_freq_DrivAge"))
        assert tool.options == [
            "Smooth (moving average)",
            "Smooth (isotonic)",
            "Cap / floor",
            "Round",
        ]
        assert at.number_input(key=wk(at, "tool_window_freq_DrivAge")).value == 3
        at.selectbox(key=wk(at, "tool_which_freq_DrivAge")).set_value(
            "Cap / floor"
        ).run()
        assert not at.exception
        at.number_input(key=wk(at, "tool_cap_freq_DrivAge")).set_value(1.05).run()
        assert not at.exception
        assert any("bands that would change" in m.label for m in at.metric)
        at.selectbox(key=wk(at, "tool_which_freq_DrivAge")).set_value("Round").run()
        assert not at.exception
        assert at.number_input(key=wk(at, "tool_decimals_freq_DrivAge")).value == 2

    def test_a_categorical_says_why_it_will_not_be_smoothed(self, workspace):
        at = AppTest.from_string(_script(workspace["project"]), default_timeout=240)
        at.run()
        at.selectbox(key=wk(at, "tables_var")).set_value("Region").run()
        assert not at.exception, [e.value for e in at.exception]
        assert any("not an order of the risk" in i.value for i in at.info)
        at.checkbox(key=wk(at, "tool_ordered_freq_Region")).set_value(True).run()
        assert not at.exception
        assert not any("not an order of the risk" in i.value for i in at.info)
        _button(at, "Apply to the table").click().run()
        assert not at.exception
        assert all(a.variable == "Region" for a in _adjustments(at))
        assert _adjustments(at)

    def test_snapshot_create_restore_and_diff(self, workspace):
        at = AppTest.from_string(_script(workspace["project"]), default_timeout=240)
        at.run()
        at.text_input(key=wk(at, "snap_name_freq")).set_value("as fitted").run()
        _button(at, "Snapshot as…").click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert [s.name for s in _snapshots(at)] == ["as fitted"]
        # a second snapshot with the same name is refused
        at.text_input(key=wk(at, "snap_name_freq")).set_value("as fitted").run()
        _button(at, "Snapshot as…").click().run()
        assert any("already has a snapshot" in e.value for e in at.error)

        _button(at, "Apply to the table").click().run()
        assert _adjustments(at)
        at.text_input(key=wk(at, "snap_name_freq")).set_value("smoothed").run()
        _button(at, "Snapshot as…").click().run()
        assert [s.name for s in _snapshots(at)] == ["as fitted", "smoothed"]

        # the diff between the two snapshots lists the bands the tool moved
        at.selectbox(key=wk(at, "snap_diff_a_freq")).set_value("as fitted").run()
        at.selectbox(key=wk(at, "snap_diff_b_freq")).set_value("smoothed").run()
        assert not at.exception, [e.value for e in at.exception]
        diffs = [
            d.value for d in at.dataframe if "status" in getattr(d.value, "columns", [])
        ]
        assert diffs and len(diffs[0]) > 0
        assert set(diffs[0]["variable"]) == {"DrivAge"}
        # the tables now *are* the "smoothed" snapshot, so that pair is empty
        at.selectbox(key=wk(at, "snap_diff_a_freq")).set_value("(the tables now)").run()
        assert any("exactly the same premium" in s.value for s in at.success)

        # restoring the first snapshot takes the adjustments away again
        at.selectbox(key=wk(at, "snap_chosen_freq")).set_value("as fitted").run()
        _button(at, "Restore").click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert not _adjustments(at)
        # deleting a snapshot asks twice: it is the one action undo cannot undo
        _button(at, "Delete").click().run()
        assert not at.exception
        assert [s.name for s in _snapshots(at)] == ["as fitted", "smoothed"]
        assert any("for good" in w.value for w in at.warning)
        _button(at, "Delete twice").click().run()
        assert not at.exception
        assert [s.name for s in _snapshots(at)] == ["smoothed"]

    def test_the_page_carries_the_training_exposure_into_the_editor(self, page):
        run = page.session_state["runs"]["freq"][1]
        rows = run.rate_model.variables["DrivAge"].table
        assert sum(r.exposure for r in rows) > 0
        assert "exposure" in rate_model_tables(run.rate_model)["DrivAge"].columns


def _snapshots(at) -> list:
    return at.session_state["_project"].models["freq"].snapshots


def _relativities(at, var: str) -> list[float]:
    """The relativities the session's rate model would score with."""
    run = at.session_state["runs"]["freq"][1]
    return [r.relativity for r in run.rate_model.variables[var].table]


# --------------------------------------------------------------------------
# review round 1: what a tool does to the book, and putting the level back
# --------------------------------------------------------------------------
class TestExpectedClaims:
    """B1 — the number the panel reports must be the money, not a geometric
    average of it."""

    def test_a_smoothing_moves_the_book_even_though_the_log_mean_is_kept(
        self, workspace, data
    ):
        p = Project.from_json(workspace["project"])
        run = run_model(p, data, "freq")
        cfg = p.models["freq"]
        train = data.filter(pl.col("traintest") == 1)

        result = tooling.smooth_moving_average(
            run.rate_model.variables["DrivAge"], "DrivAge", window=3
        )
        # the shape rule holds ...
        assert abs(result.log_mean_after - result.log_mean_before) < TOL
        # ... and it is *not* the same as leaving the premium alone
        before = expected_claims(run.rate_model, train, cfg)
        after = expected_claims(
            tooling.preview_model(run.rate_model, "DrivAge", result.values),
            train,
            cfg,
        )
        assert before > 0
        assert abs(after / before - 1.0) > 1e-6, (
            "a smoothing that happens to leave the book untouched would make "
            "this test vacuous; pick a wobblier factor"
        )

    def test_preview_model_changes_one_table_and_nothing_else(self, workspace, data):
        p = Project.from_json(workspace["project"])
        run = run_model(p, data, "freq")
        values = [r.relativity for r in run.rate_model.variables["DrivAge"].table]
        same = tooling.preview_model(run.rate_model, "DrivAge", values)
        np.testing.assert_allclose(
            same.predict(data.head(200), exposure_col=None),
            run.rate_model.predict(data.head(200), exposure_col=None),
            rtol=0,
            atol=0,
        )
        assert same.base_rate == run.rate_model.base_rate
        # the original is untouched by the preview
        capped = tooling.cap_floor(
            run.rate_model.variables["DrivAge"], "DrivAge", cap=0.5
        )
        tooling.preview_model(run.rate_model, "DrivAge", capped.values)
        assert [
            r.relativity for r in run.rate_model.variables["DrivAge"].table
        ] == values

    def test_rebalance_puts_total_expected_claims_back_exactly(self, workspace, data):
        p = Project.from_json(workspace["project"])
        run = run_model(p, data, "freq")
        cfg = p.models["freq"]
        train = data.filter(pl.col("traintest") == 1)
        target = expected_claims(run.rate_model, train, cfg)

        # cap the curve hard: the book loses money
        capped = tooling.cap_floor(
            run.rate_model.variables["DrivAge"], "DrivAge", cap=0.9
        )
        for row, value in zip(
            run.rate_model.variables["DrivAge"].table, capped.values, strict=True
        ):
            if abs(value - row.relativity) > 1e-12:
                cfg.adjustments.append(Adjustment("DrivAge", row.from_, row.to_, value))
        rebuild_rate_model(p, run, data)
        assert expected_claims(run.rate_model, train, cfg) < target * (1 - 1e-4)

        # rebalance: the level is back to the penny, the relativities untouched
        relativities = [r.relativity for r in run.rate_model.variables["DrivAge"].table]
        cfg.base_rate_override = rebalance_override(p, run, data)
        rebuild_rate_model(p, run, data)
        assert expected_claims(run.rate_model, train, cfg) == pytest.approx(
            target, rel=1e-10
        )
        assert [
            r.relativity for r in run.rate_model.variables["DrivAge"].table
        ] == pytest.approx(relativities, abs=1e-12)
        # and it is idempotent: rebalancing a balanced book changes nothing
        again = rebalance_override(p, run, data)
        assert again == pytest.approx(run.rate_model.base_rate, rel=1e-12)


class TestStaleAdjustments:
    """S1 — a set of adjustments that no longer fits the model is refused with
    a message, never applied half-way and never a traceback."""

    def test_missing_variables_names_what_cannot_be_applied(self, workspace, data):
        p = Project.from_json(workspace["project"])
        run = run_model(p, data, "freq")
        adjustments = [
            Adjustment("Region", "R1", "R1", 1.5),
            Adjustment("Gone", 1.0, 2.0, 1.5),
            Adjustment("AlsoGone", 1.0, 2.0, 1.5),
        ]
        assert missing_variables(run.rate_model, adjustments) == ["AlsoGone", "Gone"]
        assert missing_variables(run.rate_model, []) == []

    def test_an_adjustment_on_a_missing_variable_is_an_adjustment_error(
        self, workspace, data
    ):
        """Not a KeyError: the workbench drops an AdjustmentError and says so,
        so a stale snapshot or project can never traceback the page."""
        p = Project.from_json(workspace["project"])
        run = run_model(p, data, "freq")
        cfg = p.models["freq"]
        cfg.adjustments = [Adjustment("Gone", 1.0, 2.0, 1.5)]
        with pytest.raises(AdjustmentError) as exc:
            apply_adjustments(rate_model_for(p, run, []), cfg)
        assert "not a variable of the model" in str(exc.value)
        assert exc.value.adjustment is cfg.adjustments[0]

    def test_removing_an_interaction_cleans_its_snapshots_too(self):
        p = Project(name="x")
        p.new_model("m")
        cfg = p.models["m"]
        cfg.adjustments = [
            Adjustment("A×B", 1.0, 2.0, 1.5, from_b="R1", to_b="R1", cell=True),
            Adjustment("A", 1.0, 2.0, 1.1),
        ]
        cfg.snapshots = [TableSnapshot("before", "now", list(cfg.adjustments))]
        assert cfg.drop_adjustments_for("A×B") == 2
        assert [a.variable for a in cfg.adjustments] == ["A"]
        assert [a.variable for a in cfg.snapshots[0].adjustments] == ["A"]
        assert cfg.drop_adjustments_for("A×B") == 0


def test_a_table_of_only_the_null_row_has_nothing_to_work_on():
    """N1 — no band means no tool, rather than a nan level check."""
    cfg = VariableConfig(type="categorical", table=[FromToRow(None, None, 1.2, 3.0)])
    for call in (
        lambda: tooling.cap_floor(cfg, "x", cap=1.0),
        lambda: tooling.round_relativities(cfg, "x", decimals=2),
        lambda: tooling.smooth_moving_average(cfg, "x", ordered=True),
    ):
        with pytest.raises(tooling.ToolingError, match="no band"):
            call()


def test_the_change_threshold_is_the_editors():
    """N2 — a tool that called something changed while the editor called it
    unchanged would enable Apply and then write nothing."""
    from easy_glm.app import grids

    assert grids.TOL == tooling.TOL


# --------------------------------------------------------------------------
# review round 1: the same three things through the page
# --------------------------------------------------------------------------
def _script_for(project_path: str) -> str:
    """The page script, on a project file that already holds what the test
    needs (a snapshot, a removed predictor, ...) and autosaves to that file."""
    return f"""
import importlib
import streamlit as st
from easy_glm.app import state as S
from easy_glm.workflow import Project

S.init_state()
if not st.session_state.get("_loaded"):
    S.set_project(Project.from_json({project_path!r}), {project_path!r})
    st.session_state._loaded = True
if S.get_run("freq") is None:
    S.fit_model("freq")
importlib.import_module("easy_glm.app.pages_tables").render()
st.session_state["_project"] = S.project()
"""


def _metric(at, label: str) -> str:
    found = [m for m in at.metric if m.label == label]
    assert found, f"no {label!r} metric among {[m.label for m in at.metric]}"
    return found[0].value


def _base_rate(at) -> float:
    return at.session_state["runs"]["freq"][1].rate_model.base_rate


class TestLevelOnThePage:
    """B1 — the panel tells the truth about the money, and can put it back."""

    def test_the_panel_reports_the_real_change_in_expected_claims(self, workspace):
        at = AppTest.from_string(_script(workspace["project"]), default_timeout=240)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        shown = _metric(at, "expected claims (training)")
        assert shown not in ("no change", "—"), shown
        assert shown.endswith("%")
        # it agrees with the engine, to the digits the tile prints
        run = at.session_state["runs"]["freq"][1]
        p = at.session_state["_project"]
        cfg = p.models["freq"]
        train = _train_rows(at)
        result = tooling.smooth_moving_average(
            run.rate_model.variables["DrivAge"], "DrivAge", window=3
        )
        before = expected_claims(run.rate_model, train, cfg)
        after = expected_claims(
            tooling.preview_model(run.rate_model, "DrivAge", result.values), train, cfg
        )
        assert shown == f"{after / before - 1:+.3%}"
        # the log-mean tiles are still there, and are still equal for a smoothing
        assert _metric(at, "mean log relativity now") == _metric(at, "after this tool")

    def test_rebalance_puts_the_book_back_and_is_one_undo_step(self, workspace):
        at = AppTest.from_string(_script(workspace["project"]), default_timeout=240)
        at.run()
        # nothing edited yet: no off-balance block at all
        assert not [b for b in at.button if b.label == "Rebalance base rate"]
        p0 = at.session_state["_project"]
        run0 = at.session_state["runs"]["freq"][1]
        target = expected_claims(
            rate_model_for(p0, run0, [], base_rate_override=None),
            _train_rows(at),
            p0.models["freq"],
        )

        _button(at, "Apply to the table").click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert any("Off-balance" in c.value for c in at.caption)
        base_before = _base_rate(at)

        _button(at, "Rebalance base rate").click().run()
        assert not at.exception, [e.value for e in at.exception]
        p = at.session_state["_project"]
        cfg = p.models["freq"]
        run = at.session_state["runs"]["freq"][1]
        assert cfg.base_rate_override is not None
        assert expected_claims(run.rate_model, _train_rows(at), cfg) == pytest.approx(
            target, rel=1e-10
        )
        assert _base_rate(at) != base_before
        assert any("balanced" in c.value for c in at.caption)
        # and it is one undo step, base rate included
        _button(at, "Undo").click().run()
        assert at.session_state["_project"].models["freq"].base_rate_override is None
        assert _base_rate(at) == base_before


class TestUndoRestoresTheBaseRate:
    """B2 — a snapshot carries a base rate; undoing its restore must give the
    old one back, not leave the snapshot's in force."""

    def test_undo_after_a_restore_returns_the_base_rate_and_the_predictions(
        self, workspace, data
    ):
        p = Project.from_json(workspace["project"])
        p.models["freq"].snapshots = [
            TableSnapshot("levelled", "now", [], base_rate_override=0.5)
        ]
        path = workspace["folder"] / "with_override.easyglm-project.json"
        p.to_json(path)
        at = AppTest.from_string(_script_for(str(path)), default_timeout=240)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        fitted_base = _base_rate(at)
        sample = data.head(300)
        before = at.session_state["runs"]["freq"][1].predict(sample)

        _button(at, "Restore").click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert _base_rate(at) == 0.5
        assert at.session_state["_project"].models["freq"].base_rate_override == 0.5

        _button(at, "Undo").click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert _base_rate(at) == fitted_base
        assert at.session_state["_project"].models["freq"].base_rate_override is None
        np.testing.assert_array_equal(
            at.session_state["runs"]["freq"][1].predict(sample), before
        )


class TestRestoringAStaleSnapshot:
    """S1 — a snapshot older than the design says so and changes nothing."""

    def test_restore_refuses_and_saves_nothing_when_a_factor_is_gone(self, workspace):
        p = Project.from_json(workspace["project"])
        cfg = p.models["freq"]
        cfg.snapshots = [
            TableSnapshot("with Region", "now", [Adjustment("Region", "R1", "R1", 1.5)])
        ]
        cfg.predictors = [v for v in cfg.predictors if v != "Region"]
        path = workspace["folder"] / "stale.easyglm-project.json"
        p.to_json(path)
        at = AppTest.from_string(_script_for(str(path)), default_timeout=240)
        at.run()
        assert not at.exception, [e.value for e in at.exception]

        _button(at, "Restore").click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert any(
            "cannot be restored" in e.value and "Region" in e.value for e in at.error
        )
        # nothing changed, and nothing was written to the project file
        assert not at.session_state["_project"].models["freq"].adjustments
        assert not Project.from_json(path).models["freq"].adjustments
        # the page is still usable: the other buttons are there
        assert [b for b in at.button if b.label == "Snapshot as…"]


def test_the_fitted_reference_is_stamped_with_the_fit_it_belongs_to(workspace):
    """The rebalance target is cached; a cache that outlived a refit would
    rebalance the book to another model's level."""
    at = AppTest.from_string(_script(workspace["project"]), default_timeout=240)
    at.run()
    _button(at, "Apply to the table").click().run()
    assert not at.exception, [e.value for e in at.exception]
    stamp, total = at.session_state["_fitted_claims"]["freq"]
    assert total > 0
    from easy_glm.app.state import model_hash

    assert stamp == model_hash(at.session_state["_project"], "freq")


def _train_rows(at) -> pl.DataFrame:
    """The training rows of the AppTest session."""
    from easy_glm.workflow import train_holdout

    p = at.session_state["_project"]
    train, _holdout = train_holdout(at.session_state["prepared"][1], p.data.split)
    return train
