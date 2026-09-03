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
    _svg,
    base_rate_change,
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


@pytest.fixture(scope="module")
def step_vs_linear(tmp_path_factory) -> tuple:
    """Two runs of the same predictors where Density is a **step** term in one
    and a **piecewise-linear** term in the other (the design lives on the
    project, so the second run is fitted after flipping the kind)."""
    folder = tmp_path_factory.mktemp("d3kinds")
    data = _write_data(folder)
    p = _project(data)
    p.design.variables.pop("Density", None)
    p.models[MODEL_A].predictors = list(PREDICTORS_B)
    df = prepare(p, load_source(p.data.source))
    stepped = run_model(p, df, MODEL_A)
    p.design.variables["Density"] = VariableDesign(kind="linear")
    straight = run_model(p, df, MODEL_A)
    return stepped, straight


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

    def test_a_moved_knot_is_compared_on_the_common_grid(self, workspace):
        """A numeric factor is compared on the union of both models' band
        edges, so moving one knot reports exactly the sliver of ages that would
        be charged differently — not four unmatched bands."""
        run = workspace["runs"][MODEL_A]
        other = copy.deepcopy(run)
        rows = other.rate_model.variables["DrivAge"].table
        moved = float(rows[3].from_)
        rows[2].to_ = moved + 0.5
        rows[3].from_ = moved + 0.5
        diff = relativity_diff(run, other)
        assert diff.height == 1, diff
        row = diff.row(0, named=True)
        assert row["variable"] == "DrivAge"
        assert row["status"] == "changed"
        assert row["band"] == f"[{moved}, {moved + 0.5})"
        # in that sliver A is already in the next band while B is still in the
        # previous one, so the two relativities are the two neighbouring ones
        assert row["relativity_a"] == pytest.approx(rows[3].relativity)
        assert row["relativity_b"] == pytest.approx(rows[2].relativity)

    def test_a_step_and_a_linear_term_are_compared_on_a_common_grid(
        self, step_vs_linear
    ):
        """The same factor banded in one model and a straight line in the other
        is never matched by band label: the rows say so in ``kind`` and the
        values are both curves evaluated at the same points."""
        stepped, straight = step_vs_linear
        assert stepped.rate_model.variables["Density"].type == "numeric"
        assert straight.rate_model.variables["Density"].type == "linear"
        diff = relativity_diff(stepped, straight).filter(
            pl.col("variable") == "Density"
        )
        assert diff.height > 0
        assert set(diff["kind"]) == {"numeric → linear"}
        assert set(diff["status"]) == {"changed"}
        # every listed band was evaluated on both curves, so both values are there
        assert diff["relativity_a"].null_count() == 0
        assert diff["relativity_b"].null_count() == 0

    def test_the_diff_is_symmetric(self, workspace):
        """Swapping the runs swaps the statuses and negates every log ratio."""
        a, b = workspace["runs"][MODEL_A], workspace["runs"][MODEL_B]
        forward = relativity_diff(a, b).sort(["variable", "band"])
        back = relativity_diff(b, a).sort(["variable", "band"])
        assert forward.height == back.height
        swap = {"only_in_a": "only_in_b", "only_in_b": "only_in_a"}
        assert [swap.get(s, s) for s in forward["status"]] == list(back["status"])
        for f, r in zip(
            forward["log_diff"].to_list(), back["log_diff"].to_list(), strict=True
        ):
            assert (f is None and r is None) or f == pytest.approx(-r, abs=1e-12)

    def test_one_edited_linear_band_is_exactly_one_row(self, workspace):
        """A piecewise-linear term is compared by its band-start values, so a
        single edited node is a single row."""
        run = workspace["runs"][MODEL_B]
        other = copy.deepcopy(run)
        rows = other.rate_model.variables["Density"].table
        # a node in the middle: editing the first sloped band would also move
        # the flat row below the clamp, which is a second (correct) row
        bands = [r for r in rows if r.from_ is not None and r.to_ is not None]
        node = bands[len(bands) // 2]
        other.rate_model.update_relativity(
            "Density", node.from_, node.to_, float(node.relativity) * 1.30
        )
        diff = relativity_diff(run, other).filter(pl.col("variable") == "Density")
        assert diff.height == 1, diff
        assert diff.row(0, named=True)["log_diff"] == pytest.approx(np.log(1.30))

    def test_the_tolerance_boundary_is_strict(self, workspace):
        """``|log diff| == tol`` is not a change; a negative tolerance is read
        as its size."""
        run = workspace["runs"][MODEL_A]
        other = copy.deepcopy(run)
        rows = other.rate_model.variables["Region"].table
        target = rows[1]
        other.rate_model.update_relativity(
            "Region", target.from_, target.to_, float(target.relativity) * 1.05
        )
        moved = relativity_diff(run, other, 0.0)
        assert moved.height == 1
        size = moved.row(0, named=True)["abs_log_diff"]
        assert relativity_diff(run, other, size).is_empty()  # strictly greater
        assert relativity_diff(run, other, size * 0.999).height == 1
        assert relativity_diff(run, other, -size * 0.999).height == 1

    def test_two_identical_relativities_are_never_a_change(self, workspace):
        """Including two zeros, whose log ratio does not exist: an actuary who
        floors the same band in both models must not get a phantom row."""
        run = workspace["runs"][MODEL_A]
        a, b = copy.deepcopy(run), copy.deepcopy(run)
        target = a.rate_model.variables["Region"].table[1]
        for model in (a, b):
            rows = model.rate_model.variables["Region"].table
            rows[1].relativity = 0.0
            model.rate_model._precompute_variables(model.rate_model.variables)
        assert target is not None
        assert relativity_diff(a, b).is_empty()
        # but a band that really crosses zero is always listed
        b.rate_model.variables["Region"].table[1].relativity = 0.5
        b.rate_model._precompute_variables(b.rate_model.variables)
        crossed = relativity_diff(a, b)
        assert crossed.height == 1
        assert crossed.row(0, named=True)["log_diff"] is None

    def test_base_rate_change_is_the_overall_level(self, workspace):
        run = workspace["runs"][MODEL_A]
        other = copy.deepcopy(run)
        other.rate_model.base_rate = run.rate_model.base_rate * 0.976
        assert base_rate_change(run, other) == pytest.approx(-0.024)
        assert base_rate_change(run, run) == pytest.approx(0.0)

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

    def test_it_carries_no_javascript_at_all(self, report):
        """The static half of D4's browser criterion, proved without a browser:
        a page with no script cannot log a console error."""
        assert "<script" not in report.lower()
        assert "javascript:" not in report.lower()
        assert not re.search(r"\son[a-z]+\s*=", report)  # no inline handlers

    def test_every_chart_has_an_accessible_name(self, report):
        """`role="img"` with no name is announced as "image" 25 times over."""
        first_children = re.findall(r"<svg\b[^>]*>\s*(<[a-z]+)", report)
        assert len(first_children) == report.count("<svg")
        assert set(first_children) == {"<title"}, set(first_children)
        assert "<title>DrivAge: fitted relativities by band</title>" in report
        assert "<title>DrivAge: actual vs expected by band (holdout)</title>" in report

    def test_the_challengers_line_is_dashed(self, report):
        """Solid over solid hides the champion wherever the two models agree —
        the workbench dashes the challenger and so must the report."""
        assert 'stroke-dasharray="7 4"' in report

    def test_the_overall_level_is_a_headline_above_the_diff(self, report, workspace):
        change = base_rate_change(
            workspace["runs"][MODEL_A], workspace["runs"][MODEL_B]
        )
        assert "Overall level (base rate)" in report
        assert f"{change:+.1%}" in report

    def test_a_challenger_that_cannot_be_scored_is_explained(self, workspace):
        """A report may never name a challenger and then silently omit the
        comparison — that is indistinguishable from a bug."""
        trimmed = workspace["frame"].drop("Density")  # only MODEL_B needs it
        html = to_report_html(
            workspace["project"],
            workspace["runs"],
            trimmed,
            champion=MODEL_A,
            challenger=MODEL_B,
        )
        assert f"No comparison with {MODEL_B}" in html
        assert "could not be scored here" in html
        assert "Density" in html.split("No comparison")[1][:600]
        assert 'id="compare"' in html  # the table of contents still resolves
        assert "Relativities that differ" not in html

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
        # side by side: one column per model per subset, so the four numbers a
        # reader compares are on one line
        for name in (MODEL_A, MODEL_B):
            for subset in ("train", "holdout"):
                assert f"{name} · {subset}" in metrics.columns, metrics.columns
        facts = frames[1]
        assert "alpha" in list(facts["metric"])
        assert MODEL_A in facts.columns and MODEL_B in facts.columns
        diff = frames[-1]
        assert "band" in diff.columns and "status" in diff.columns
        assert len(diff) > 0

    def test_the_metrics_table_is_exactly_the_runs_metrics(self, workspace):
        """Every cell of the side-by-side table is the model's own number."""
        from easy_glm.app.pages_compare import _ROWS, _fmt, _metrics_table

        runs = workspace["runs"]
        table = _metrics_table(runs[MODEL_A], runs[MODEL_B])
        for label, key, spec in _ROWS:
            row = table.filter(pl.col("metric") == label).row(0, named=True)
            for name in (MODEL_A, MODEL_B):
                for subset in ("train", "holdout"):
                    expected = _fmt(runs[name].metrics[subset][key], spec)
                    assert row[f"{name} · {subset}"] == expected, (label, name, subset)

    def test_the_overall_level_change_is_shown_above_the_diff(self, workspace):
        at = _run(_script("pages_compare", workspace["project_path"], fit=True))
        assert any("Overall level (base rate)" in m.value for m in at.info), [
            m.value for m in at.info
        ]

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


# --------------------------------------------------------------------------
# D4 — the report's chart writer
# --------------------------------------------------------------------------
class TestSvg:
    """`workflow._svg` is hand-rolled tick and axis arithmetic; these pin the
    degenerate inputs that a real book eventually produces."""

    def test_nice_ticks_on_degenerate_ranges(self):
        assert _svg.nice_ticks(1.0, 1.0)  # a constant series still gets ticks
        assert _svg.nice_ticks(float("nan"), 1.0) == [0.0, 1.0]
        assert _svg.nice_ticks(float("-inf"), float("inf")) == [0.0, 1.0]
        ticks = _svg.nice_ticks(0.0, 100.0)
        assert ticks[0] <= 0.0 and ticks[-1] >= 100.0
        assert len(ticks) < 12

    def test_a_chart_with_no_bands_is_an_empty_frame(self):
        out = _svg.category_chart([], bars=[], lines=[], title="nothing")
        assert out.startswith("<svg") and out.endswith("</svg>")
        assert "<title>nothing</title>" in out

    def test_a_single_band_and_all_null_values(self):
        assert "<svg" in _svg.category_chart(["only"], bars=[1.0], lines=[])
        out = _svg.category_chart(
            ["a", "b"], bars=[None, None], lines=[("x", [None, None], _svg.BLUE)]
        )
        assert "<svg" in out and "NaN" not in out

    def test_labels_and_titles_are_escaped(self):
        out = _svg.category_chart(
            ["<script>alert(1)</script>"],
            bars=[1.0],
            lines=[("a & b", [1.0], _svg.BLUE)],
            title="<b>t</b>",
        )
        assert "<script>" not in out
        assert "&lt;script&gt;" in out and "&amp;" in out
        assert "<title>&lt;b&gt;t&lt;/b&gt;</title>" in out

    def test_a_dashed_line_is_dashed_in_the_chart_and_the_legend(self):
        plain = _svg.category_chart(["a"], lines=[("x", [1.0], _svg.BLUE)])
        dashed = _svg.category_chart(["a"], lines=[("x", [1.0], _svg.BLUE, True)])
        assert 'stroke-dasharray="7 4"' not in plain
        assert dashed.count('stroke-dasharray="7 4"') == 2  # the line and its key

    def test_the_ratio_colour_is_centred_on_one(self):
        assert _svg._ratio_colour(1.0) == "rgb(255,255,255)"
        assert _svg._ratio_colour(None) == _svg._ratio_colour(-1.0)  # "no value"
        above, below = _svg._ratio_colour(2.0), _svg._ratio_colour(0.5)
        assert above.startswith("rgb(255,") and below.endswith(",153)")

    def test_a_heatmap_names_itself_and_labels_every_cell(self):
        out = _svg.heatmap(
            ["r1", "r2"],
            ["c1"],
            [[1.2], [None]],
            row_name="A",
            col_name="B",
            hover={"exposure": [[10.0], [0.0]]},
            title="A × B: cells",
        )
        assert "<title>A × B: cells</title>" in out
        assert out.count("<title>") == 3  # the chart plus one per cell
        assert "exposure: 10.0" in out

    def test_a_curve_is_drawn_and_named(self):
        out = _svg.curve_chart(
            [("relativity", [0.0, 1.0, 2.0], [1.0, 1.1, 1.2], _svg.BLUE)],
            marks=[(1.0, "base (1.00)")],
            title="Density: curve",
        )
        assert "<polyline" in out
        assert "<title>Density: curve</title>" in out
        assert "base (1.00)" in out
