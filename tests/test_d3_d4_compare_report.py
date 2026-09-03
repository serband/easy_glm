"""D3 / D4 — champion vs challenger and the self-contained HTML report.

Three layers:

* the engine helper :func:`easy_glm.workflow.relativity_diff` on hand-made
  differences (identical runs, one known adjustment, a variable only one model
  has, moved knots, the base rate, the tolerance);
* :func:`easy_glm.workflow.to_report_html` — self-contained (no ``http(s)://``
  in a ``src=`` / ``href=``), one section per predictor, the comparison section
  only when a challenger is passed, under 5 MB, and (when Playwright is
  installed) no console errors in a headless browser;
* the Compare page, the sidebar's "compare with" and the Export page's report
  button through Streamlit's AppTest.
"""

from __future__ import annotations

import copy
import importlib.util
import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from easy_glm.workflow import (
    Interaction,
    Project,
    VariableDesign,
    describe_diff,
    load_source,
    prepare,
    relativity_diff,
    run_model,
    to_report_html,
)

MODEL_A = "freq_a"
MODEL_B = "freq_b"
PREDICTORS_A = ["DrivAge", "BonusMalus", "Region"]
PREDICTORS_B = [*PREDICTORS_A, "Density"]


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def _write_data(folder: Path) -> Path:
    rng = np.random.default_rng(19)
    n = 4000
    age = rng.integers(18, 80, n).astype(float)
    bm = rng.integers(50, 200, n).astype(float)
    dens = np.exp(rng.uniform(0, 8, n))
    region = rng.choice(["R1", "R2", "R3", "R4"], n, p=[0.5, 0.3, 0.15, 0.05]).astype(
        object
    )
    expo = rng.uniform(0.2, 1.0, n)
    mu = np.exp(
        -2.2
        - 0.02 * np.maximum(45 - age, 0)
        + 0.004 * (bm - 100)
        + 0.05 * np.log(dens)
        + np.where(region == "R1", 0, 0.2)
    )
    df = pl.DataFrame(
        {
            "IDpol": np.arange(n),
            "ClaimNb": rng.poisson(mu * expo).astype(float),
            "Exposure": expo,
            "DrivAge": age,
            "BonusMalus": bm,
            "Density": dens,
            "Region": region,
            "traintest": (rng.random(n) < 0.7).astype(int),
        }
    )
    path = folder / "policies.parquet"
    df.write_parquet(path)
    return path


def _project(data: Path) -> Project:
    p = Project(name="d3test")
    p.data.source.type = "parquet"
    p.data.source.path = str(data)
    p.data.roles = {
        "ClaimNb": "target",
        "Exposure": "weight",
        "IDpol": "id",
        "DrivAge": "predictor",
        "BonusMalus": "predictor",
        "Density": "predictor",
        "Region": "predictor",
        "traintest": "split",
    }
    p.data.split.mode = "column"
    p.data.split.column = "traintest"
    p.design.variables["Density"] = VariableDesign(kind="linear")
    for name, predictors in ((MODEL_A, PREDICTORS_A), (MODEL_B, PREDICTORS_B)):
        p.new_model(name, divide_target_by_weight=True)
        p.models[name].predictors = list(predictors)
        p.models[name].penalty.alpha = 0.002
        p.models[name].penalty.cv = None
    p.models[MODEL_B].interactions = [Interaction("DrivAge", "Region", 0.01)]
    p.champion = MODEL_A
    return p


@pytest.fixture(scope="module")
def workspace(tmp_path_factory) -> dict:
    folder = tmp_path_factory.mktemp("d3")
    data = _write_data(folder)
    p = _project(data)
    path = folder / "d3.easyglm-project.json"
    p.to_json(path)
    df = prepare(p, load_source(p.data.source))
    runs = {name: run_model(p, df, name) for name in p.models}
    return {
        "folder": folder,
        "project_path": str(path),
        "project": p,
        "frame": df,
        "runs": runs,
    }


# --------------------------------------------------------------------------
# D3 — relativity_diff
# --------------------------------------------------------------------------
class TestRelativityDiff:
    def test_identical_runs_give_an_empty_diff(self, workspace):
        run = workspace["runs"][MODEL_A]
        diff = relativity_diff(run, run)
        assert diff.is_empty()
        # even empty, the table has the documented columns
        assert diff.columns == [
            "variable",
            "kind",
            "band",
            "status",
            "relativity_a",
            "relativity_b",
            "log_diff",
            "abs_log_diff",
        ]

    def test_a_known_adjustment_is_exactly_one_row(self, workspace):
        run = workspace["runs"][MODEL_A]
        other = copy.deepcopy(run)
        rows = other.rate_model.variables["Region"].table
        target = rows[1]
        new_value = float(target.relativity) * 1.25
        other.rate_model.update_relativity(
            "Region", target.from_, target.to_, new_value
        )
        diff = relativity_diff(run, other)
        assert diff.height == 1, diff
        row = diff.row(0, named=True)
        assert row["variable"] == "Region"
        assert row["status"] == "changed"
        assert row["relativity_b"] == pytest.approx(new_value)
        assert row["log_diff"] == pytest.approx(np.log(1.25))

    def test_a_change_below_the_tolerance_is_not_listed(self, workspace):
        run = workspace["runs"][MODEL_A]
        other = copy.deepcopy(run)
        rows = other.rate_model.variables["Region"].table
        target = rows[1]
        other.rate_model.update_relativity(
            "Region", target.from_, target.to_, float(target.relativity) * 1.005
        )
        assert relativity_diff(run, other, 0.01).is_empty()  # 0.5 % < 1 %
        assert relativity_diff(run, other, 0.001).height == 1

    def test_a_variable_only_one_model_has_is_listed(self, workspace):
        a, b = workspace["runs"][MODEL_A], workspace["runs"][MODEL_B]
        diff = relativity_diff(a, b)
        only_b = diff.filter(pl.col("status") == "only_in_b")
        assert set(only_b["variable"]) == {"Density", "DrivAge×Region"}
        assert (only_b["band"] == "(all bands)").all()
        # and the other way round
        back = relativity_diff(b, a)
        assert set(back.filter(pl.col("status") == "only_in_a")["variable"]) == {
            "Density",
            "DrivAge×Region",
        }

    def test_moved_bands_are_only_in_rows_not_false_changes(self, workspace):
        """Bands are matched by label, so a knot that moved is reported as a
        band only one model has rather than as a changed relativity."""
        run = workspace["runs"][MODEL_A]
        other = copy.deepcopy(run)
        rows = other.rate_model.variables["DrivAge"].table
        rows[2].to_ = float(rows[2].to_) + 0.5
        rows[3].from_ = float(rows[3].from_) + 0.5
        diff = relativity_diff(run, other)
        assert set(diff["status"]) <= {"band_only_in_a", "band_only_in_b"}
        assert diff.filter(pl.col("status") == "band_only_in_a").height == 2
        assert diff.filter(pl.col("status") == "band_only_in_b").height == 2

    def test_the_base_rate_is_compared_too(self, workspace):
        run = workspace["runs"][MODEL_A]
        other = copy.deepcopy(run)
        other.rate_model.base_rate = run.rate_model.base_rate * 1.10
        diff = relativity_diff(run, other)
        assert diff.height == 1
        row = diff.row(0, named=True)
        assert row["variable"] == "(base rate)"
        assert row["log_diff"] == pytest.approx(np.log(1.10))

    def test_interaction_cells_are_compared(self, workspace):
        run = workspace["runs"][MODEL_B]
        other = copy.deepcopy(run)
        cells = other.rate_model.variables["DrivAge×Region"].table
        cell = next(c for c in cells if c.exposure > 0)
        other.rate_model.update_relativity(
            "DrivAge×Region",
            cell.from_a,
            cell.to_a,
            float(cell.relativity) * 1.5,
            from_b=cell.from_b,
            to_b=cell.to_b,
        )
        diff = relativity_diff(run, other)
        assert diff.height == 1
        assert diff.row(0, named=True)["kind"] == "interaction"
        assert " | " in diff.row(0, named=True)["band"]

    def test_describe_diff_puts_the_statuses_in_words(self, workspace):
        a, b = workspace["runs"][MODEL_A], workspace["runs"][MODEL_B]
        shown = describe_diff(relativity_diff(a, b), MODEL_A, MODEL_B)
        assert f"only in {MODEL_B}" in set(shown["status"])
        assert f"{MODEL_A} relativity" in shown.columns
        assert f"{MODEL_B} relativity" in shown.columns


# --------------------------------------------------------------------------
# D4 — the HTML report
# --------------------------------------------------------------------------
def _no_scripts(html: str) -> str:
    """The page with every script body removed, so an attribute check cannot be
    fooled by a URL inside embedded JavaScript."""
    return re.sub(r"(<script[^>]*>).*?(</script>)", r"\1\2", html, flags=re.S)


@pytest.fixture(scope="module")
def report(workspace) -> str:
    return to_report_html(
        workspace["project"],
        workspace["runs"],
        workspace["frame"],
        champion=MODEL_A,
        challenger=MODEL_B,
    )


class TestReport:
    def test_it_is_self_contained(self, report):
        external = re.findall(
            r'(?:src|href)\s*=\s*["\']https?://[^"\']*', _no_scripts(report)
        )
        assert external == []
        assert "cdn" not in report.lower().split("<style>")[0]

    def test_it_is_a_whole_html_document(self, report):
        assert report.startswith("<!doctype html>")
        assert report.rstrip().endswith("</html>")
        assert "<style>" in report

    def test_one_section_per_predictor(self, report, workspace):
        for var in PREDICTORS_A:
            assert f'id="var-{var.lower()}"' in report, var
        assert report.count('class="variable"') >= len(PREDICTORS_A)

    def test_the_champions_metrics_and_versions_are_there(self, report, workspace):
        run = workspace["runs"][MODEL_A]
        assert f"{run.metrics['holdout']['ae']:.4f}" in report
        assert "generated" in report
        assert "polars" in report

    def test_the_script_is_in_the_appendix(self, report):
        assert "fit_glm(" in report
        assert "StepEncoder(&#x27;DrivAge&#x27;" in report  # escaped, inside <pre>

    def test_the_compare_section_only_with_a_challenger(self, workspace, report):
        assert 'id="compare"' in report
        assert MODEL_B in report
        solo = to_report_html(
            workspace["project"],
            workspace["runs"],
            workspace["frame"],
            champion=MODEL_A,
        )
        assert 'id="compare"' not in solo
        assert "Relativities that differ" not in solo

    def test_the_challenger_can_be_the_champion_of_the_report(self, workspace):
        html = to_report_html(
            workspace["project"],
            workspace["runs"],
            workspace["frame"],
            champion=MODEL_B,
            challenger=MODEL_A,
        )
        assert 'id="var-drivage-region"' in html  # the interaction heatmap
        assert 'id="interactions"' in html

    def test_it_is_small_enough_to_email(self, report):
        assert len(report.encode()) < 5 * 1000 * 1000

    def test_an_unknown_model_is_a_clear_error(self, workspace):
        with pytest.raises(KeyError):
            to_report_html(
                workspace["project"],
                workspace["runs"],
                workspace["frame"],
                champion="nope",
            )

    def test_the_same_model_twice_is_not_a_comparison(self, workspace):
        html = to_report_html(
            workspace["project"],
            workspace["runs"],
            workspace["frame"],
            champion=MODEL_A,
            challenger=MODEL_A,
        )
        assert 'id="compare"' not in html

    def test_it_opens_in_a_headless_browser_without_console_errors(
        self, report, tmp_path
    ):
        sync_playwright = pytest.importorskip(
            "playwright.sync_api", reason="Playwright not installed"
        ).sync_playwright
        path = tmp_path / "report.html"
        path.write_text(report)
        problems: list[str] = []
        requests: list[str] = []
        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch()
            except Exception as exc:  # noqa: BLE001 - no browser downloaded
                pytest.skip(f"no chromium: {exc}")
            page = browser.new_page()
            page.on(
                "console",
                lambda m: (
                    problems.append(f"{m.type}: {m.text}")
                    if m.type == "error"
                    else None
                ),
            )
            page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
            page.on(
                "request",
                lambda r: (
                    requests.append(r.url) if not r.url.startswith("file:") else None
                ),
            )
            page.goto(path.resolve().as_uri(), wait_until="networkidle")
            page.wait_for_timeout(300)
            assert page.locator("section.variable").count() >= len(PREDICTORS_A)
            browser.close()
        assert problems == []
        assert requests == []  # nothing was fetched from anywhere


# --------------------------------------------------------------------------
# D3 / D4 — the pages
# --------------------------------------------------------------------------
# Streamlit guards only the page tests: the engine and report tests above must
# still run under an interpreter that has Playwright but no Streamlit.
if importlib.util.find_spec("streamlit") is not None:
    from streamlit.testing.v1 import AppTest
else:  # pragma: no cover - exercised in the Playwright-only environment
    AppTest = None

needs_streamlit = pytest.mark.skipif(
    AppTest is None, reason="Streamlit is not installed in this interpreter"
)


def wk(at, name: str) -> str:
    return f"{name}_{at.session_state['project_token']}"


def _script(page: str, project_path: str, *, fit: bool) -> str:
    return f"""
import importlib
import streamlit as st
from easy_glm.app import state as S
from easy_glm.workflow import Project

S.init_state()
if not st.session_state.get("_loaded"):
    S.set_project(Project.from_json({project_path!r}), None)
    st.session_state._loaded = True
if {fit!r}:
    for _m in ({MODEL_A!r}, {MODEL_B!r}):
        if S.get_run(_m) is None:
            S.fit_model(_m)
importlib.import_module("easy_glm.app." + {page!r}).render()
st.session_state["_project"] = S.project()
"""


def _run(script: str, timeout: int = 300) -> AppTest:
    at = AppTest.from_string(script, default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@needs_streamlit
class TestComparePage:
    def test_it_asks_for_a_second_fit_when_there_is_none(self, workspace):
        at = _run(_script("pages_compare", workspace["project_path"], fit=False))
        assert any("two fitted models" in m.value for m in at.info)

    def test_it_shows_both_models_and_the_diff(self, workspace):
        at = _run(_script("pages_compare", workspace["project_path"], fit=True))
        text = " ".join(
            [m.value for m in at.markdown]
            + [c.value for c in at.caption]
            + [s.value for s in at.subheader]
        )
        assert MODEL_A in text and MODEL_B in text
        assert at.selectbox(key=wk(at, "cmp_a")).value == MODEL_A
        assert at.selectbox(key=wk(at, "cmp_b_freq_a_None")).value == MODEL_B
        # the metrics table and, on the last tab, the relativity diff
        frames = [d.value for d in at.dataframe]  # AppTest hands back pandas
        assert frames, "no tables on the page"
        metrics = frames[0]
        assert "A/E" in list(metrics["metric"])
        assert MODEL_A in metrics.columns and MODEL_B in metrics.columns
        diff = frames[-1]
        assert "band" in diff.columns and "status" in diff.columns
        assert len(diff) > 0

    def test_the_tolerance_filters_the_diff(self, workspace):
        at = _run(_script("pages_compare", workspace["project_path"], fit=True))
        before = len(at.dataframe[-1].value)
        at.number_input(key=wk(at, "cmp_tol")).set_value(0.9).run()
        assert not at.exception, [e.value for e in at.exception]
        after = at.dataframe[-1].value
        assert len(after) < before
        # no relativity moves by a factor of e**0.9, so only the variables one
        # model has and the other does not are left
        assert all(str(v).startswith("only in") for v in after["status"])

    def test_make_champion_promotes_the_challenger(self, workspace):
        at = _run(_script("pages_compare", workspace["project_path"], fit=True))
        assert at.session_state["_project"].champion == MODEL_A
        button = [b for b in at.button if b.label == f"Make {MODEL_B} champion"]
        assert button
        button[0].click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert at.session_state["_project"].champion == MODEL_B

    def test_it_reports_a_model_that_can_no_longer_be_scored(self, workspace, tmp_path):
        """A predictor removed from the data is a message, not a traceback."""
        p = Project.from_json(workspace["project_path"])
        df = pl.read_parquet(p.data.source.path).drop("BonusMalus")
        data = tmp_path / "trimmed.parquet"
        df.write_parquet(data)
        script = _script("pages_compare", workspace["project_path"], fit=True)
        script = script.replace(
            "importlib.import_module",
            f"S.project().data.source.path = {str(data)!r}\n"
            "st.session_state.raw = None\n"
            "importlib.import_module",
        )
        at = AppTest.from_string(script, default_timeout=300)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        assert at.error or at.info or at.warning


@pytest.fixture(scope="module")
def saved_project(tmp_path_factory) -> str:
    """A project whose two fits are persisted next to it, so opening it (as
    ``main.py`` does) restores both without refitting."""
    folder = tmp_path_factory.mktemp("d3saved")
    data = _write_data(folder)
    p = _project(data)
    path = folder / "saved.easyglm-project.json"
    p.to_json(path)
    at = AppTest.from_string(
        f"""
import streamlit as st
from easy_glm.app import state as S
from easy_glm.workflow import Project

S.init_state()
if not st.session_state.get("_loaded"):
    S.set_project(Project.from_json({str(path)!r}), {str(path)!r})
    st.session_state._loaded = True
for _m in ({MODEL_A!r}, {MODEL_B!r}):
    if S.get_run(_m) is None:
        S.fit_model(_m)
st.session_state["_fitted"] = S.fitted_models()
""",
        default_timeout=300,
    )
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["_fitted"] == [MODEL_A, MODEL_B]
    return str(path)


@needs_streamlit
class TestSidebarChallenger:
    def test_diagnostics_defaults_to_the_sidebar_choice(self, workspace):
        script = _script("pages_diagnostics", workspace["project_path"], fit=True)
        script = script.replace(
            "importlib.import_module",
            f"S.set_challenger({MODEL_B!r})\nimportlib.import_module",
        )
        at = AppTest.from_string(script, default_timeout=300)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        assert at.selectbox(key=wk(at, f"diag_chal_{MODEL_B}")).value == MODEL_B

    def test_rate_tables_overlays_the_sidebar_challenger(self, workspace):
        script = _script("pages_tables", workspace["project_path"], fit=True)
        script = script.replace(
            "importlib.import_module",
            f"S.set_challenger({MODEL_B!r})\nimportlib.import_module",
        )
        at = AppTest.from_string(script, default_timeout=300)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        key = wk(at, f"tables_chal_{MODEL_A}_{MODEL_B}")
        assert at.selectbox(key=key).value == MODEL_B

    def test_the_main_page_offers_the_selector(self, saved_project):
        """The sidebar's "compare with" appears once two models are fitted (the
        runs are restored from the project's persisted folder)."""
        import sys

        from easy_glm import app as app_pkg

        argv = sys.argv
        sys.argv = ["main.py", f"--project={saved_project}"]
        try:
            at = AppTest.from_file(
                str(Path(app_pkg.__file__).with_name("main.py")), default_timeout=300
            )
            at.run()
        finally:
            sys.argv = argv
        assert not at.exception, [e.value for e in at.exception]
        boxes = [b for b in at.sidebar.selectbox if b.label == "Compare with"]
        assert boxes, [b.label for b in at.sidebar.selectbox]
        assert MODEL_B in boxes[0].options


@needs_streamlit
class TestExportPage:
    def test_the_report_button_is_offered_and_produces_html(self, workspace):
        at = _run(_script("pages_export", workspace["project_path"], fit=True))
        downloads = at.get("download_button")
        buttons = [b for b in downloads if "HTML report" in b.label]
        assert buttons, [b.label for b in downloads]
        assert any("HTML report" in s.value for s in at.subheader)
        # AppTest cannot read a download payload; the button's help carries the
        # size of the file it would hand over
        assert buttons[0].proto.help.endswith(" kB")

    def test_the_report_can_include_the_challenger(self, workspace):
        at = _run(_script("pages_export", workspace["project_path"], fit=True))

        def _size() -> int:
            button = [b for b in at.get("download_button") if "HTML report" in b.label][
                0
            ]
            return int(button.proto.help.split()[0].replace(",", ""))

        solo = _size()
        at.selectbox(key=wk(at, f"report_chal_{MODEL_A}_None")).set_value(MODEL_B).run()
        assert not at.exception, [e.value for e in at.exception]
        assert _size() > solo  # the comparison section made the file bigger
