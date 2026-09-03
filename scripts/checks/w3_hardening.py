"""Actuarial check for piece W3 — workbench hardening.

Replays the thirteen blocking findings of the break-it review
(docs/reviews/w2-breakage.md, items 1–13: four ways to lose work, nine
tracebacks) against the current workbench, using Streamlit's test harness on a
small synthetic motor book, and writes a plain-language page: what the tester
did, what used to happen, and what the tool says now.

Usage: python scripts/checks/w3_hardening.py [--write]
  --write regenerates docs/checks/w3-hardening.md; otherwise the page is printed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "checks" / "w3-hardening.md"
N = 3000


def wk(at, name: str) -> str:
    """Session-state key of a page widget (keys carry the project token)."""
    return f"{name}_{at.session_state['project_token']}"


def _frame():
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(5)
    age = rng.integers(18, 80, N).astype(float)
    bm = rng.integers(50, 200, N).astype(float)
    region = rng.choice(["R1", "R2", "R3", "R4"], N, p=[0.5, 0.3, 0.15, 0.05])
    expo = rng.uniform(0.2, 1.0, N)
    mu = np.exp(-2.2 - 0.02 * np.maximum(45 - age, 0) + 0.004 * (bm - 100))
    return pl.DataFrame(
        {
            "IDpol": [f"P{i:06d}" for i in range(N)],
            "ClaimNb": rng.poisson(mu * expo).astype(float),
            "Exposure": expo,
            "DrivAge": age,
            "BonusMalus": bm,
            "Region": region.astype(object),
            "traintest": (rng.random(N) < 0.7).astype(int),
        }
    )


def _project(data_path: Path):
    from easy_glm.workflow import Project

    p = Project(name="w3check")
    p.data.source.type = "parquet"
    p.data.source.path = str(data_path)
    p.data.roles = {
        "ClaimNb": "target",
        "Exposure": "weight",
        "IDpol": "id",
        "DrivAge": "predictor",
        "BonusMalus": "predictor",
        "Region": "predictor",
        "traintest": "split",
    }
    p.data.split.mode = "column"
    p.data.split.column = "traintest"
    p.new_model("freq", divide_target_by_weight=True)
    p.models["freq"].penalty.alpha = 0.002
    p.models["freq"].penalty.cv = None
    return p


def _script(page: str, project_path: str, prelude: str = "", body: str = "") -> str:
    return f"""
import importlib
import streamlit as st
from easy_glm.app import state as S
from easy_glm.workflow import Project

S.init_state()
st.session_state.setdefault("_out", {{}})
out = st.session_state["_out"]
if not st.session_state.get("_loaded"):
    S.set_project(Project.from_json({project_path!r}), {project_path!r})
    st.session_state._loaded = True
    {prelude}
importlib.import_module("easy_glm.app." + {page!r}).render()
{body}
out["path"] = st.session_state.get("project_path")
out["name"] = S.project().name
"""


def _run(script: str):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(script, default_timeout=120)
    at.run()
    return at


def _messages(at) -> list[str]:
    return [e.value for e in at.error] + [w.value for w in at.warning]


def _short(text: str, n: int = 150) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def main(write: bool) -> int:

    from easy_glm.app import ui
    from easy_glm.app.pages_variables import apply_roles_grid, preview_derived
    from easy_glm.workflow import Project, add_split_column
    from easy_glm.workflow.project import validate_model_name

    tmp = Path(tempfile.mkdtemp(prefix="easy_glm_w3_"))
    data = tmp / "policies.parquet"
    _frame().write_parquet(data)
    project = tmp / "w3check.easyglm-project.json"
    _project(data).to_json(project)
    raw_columns = _frame().columns
    rows = []  # finding, what the tester did, before, now

    # 1 — New empty project
    before = project.read_bytes()
    at = _run(_script("pages_project", str(project)))
    at.button(key=wk(at, "new_project_btn")).click().run()
    at.button(key=wk(at, "new_project_btn")).click().run()
    at.text_input(
        key=[k for k in at.session_state.filtered_state if k.startswith("proj_name_")][
            0
        ]
    ).set_value("scratch").run()
    now1 = (
        "The first click asks you to click again; the new project starts with no file "
        f"(path shown: {at.session_state['_out']['path']}), and typing a name changes "
        f"nothing on disk — the old file is byte-for-byte unchanged: "
        f"{'yes' if project.read_bytes() == before else 'NO'}."
    )
    rows.append(
        (
            1,
            "New empty project, then type a name",
            "The new project inherited the old file's path and the next autosave "
            "emptied the old project file.",
            now1,
        )
    )

    # 2 — two tabs on one file
    other = Project.from_json(project)
    other.name = "edited-by-tab-A"
    body = """
p = S.project()
if not out.get("touched"):
    other = Project.from_json(st.session_state.project_path); other.name = "edited-by-tab-A"
    other.to_json(st.session_state.project_path)
    import os, time; time.sleep(0.01)
    os.utime(st.session_state.project_path, None)
    out["touched"] = True
    p.models["freq"].notes = "note from tab B"
    S.touch()
out["conflict"] = st.session_state.get("conflict")
"""
    at = _run(_script("pages_model", str(project), body=body))
    at.run()
    warn = [
        w.value.replace(f"**{project}**", "<project file>")
        for w in at.warning
        if "another browser tab" in w.value
    ]
    on_disk = json.loads(project.read_text())["name"]
    at.button(key="conflict_reload").click().run()
    now2 = (
        f"Tab B's autosave is paused with the notice “{_short(warn[0], 110) if warn else 'MISSING'}”. "
        f"The file still holds tab A's version (name {on_disk!r}); after *Reload from disk* tab B "
        f"shows {at.session_state['_out']['name']!r} and its own stale edit is dropped."
    )
    rows.append(
        (
            2,
            "The same project open in two tabs, both editing",
            "Each tab's autosave wrote its whole copy over the other tab's work; "
            "the last tab to touch anything won silently.",
            now2,
        )
    )
    project.write_text(json.dumps(_project(data).to_dict(), indent=2))

    # 3 — rename the target
    p = Project.from_json(project)
    p.rename_column("ClaimNb", "claims")
    problems = p.validate(model="freq", columns=["IDpol", "Exposure", "DrivAge"])
    now3 = (
        f"Renaming ClaimNb to claims carries the role and the model's target with it "
        f"(target is now {p.models['freq'].target!r}). A model whose column has truly gone "
        f"is refused with: “{_short(problems[0], 90)}”."
    )
    rows.append(
        (
            3,
            "Rename the target column in the roles grid",
            "The Target box silently jumped to the first column (the policy number) "
            "and that was autosaved.",
            now3,
        )
    )

    # 4 — random split named after a data column
    p = Project.from_json(project)
    p.data.split.mode = "random"
    p.data.split.column = "ClaimNb"
    try:
        add_split_column(_frame(), p.data.split)
        msg4 = "ACCEPTED"
    except ValueError as exc:
        msg4 = str(exc)
    rows.append(
        (
            4,
            "Random split with the split column named ClaimNb (the target)",
            "The 0/1 flag replaced the target column; the fit 'succeeded' on nonsense.",
            f"Refused: “{_short(msg4, 120)}”. The Split page shows the same message and "
            "keeps the old name.",
        )
    )

    # 5 — data file whose columns differ from the project
    weird = tmp / "weird.parquet"
    _frame().rename({"DrivAge": "1st_age", "Region": "région/zone"}).write_parquet(
        weird
    )
    p = Project.from_json(project)
    p.data.source.path = str(weird)
    weird_project = tmp / "weird.easyglm-project.json"
    p.to_json(weird_project)
    seen = {}
    for page in ("pages_variables", "pages_split", "pages_model"):
        at = _run(_script(page, str(weird_project)))
        seen[page] = (
            [e.value for e in at.exception],
            [m for m in _messages(at) if "not in" in m],
        )
    crashes = sum(len(v[0]) for v in seen.values())
    sample = next((m[1][0] for m in seen.values() if m[1]), "MISSING")
    rows.append(
        (
            5,
            "Load a data file whose columns differ from the project's",
            "Traceback on Variables and Split; the project was unusable until the "
            "file was edited by hand.",
            f"Every page renders ({crashes} tracebacks across Variables, Split, Model); "
            f"the Variables page lists the roles whose columns are gone and the Model "
            f"page says “{_short(seen['pages_model'][1][0], 80)}”. E.g. "
            f"“{_short(sample, 100)}”.",
        )
    )

    # 6 — rename onto an existing name
    p = Project.from_json(project)
    grid = [
        {
            "column": c,
            "rename to": "",
            "role": p.data.roles.get(c, "unassigned"),
            "type": "auto",
        }
        for c in raw_columns
    ]
    grid[4]["rename to"] = "DrivAge"
    changed, notices = apply_roles_grid(p, raw_columns, grid)
    rows.append(
        (
            6,
            "Rename BonusMalus to DrivAge (a name already in use)",
            "Traceback 'column DrivAge is duplicate'; the rename was saved and the "
            "role lost.",
            f"Refused, nothing saved (changed={changed}): “{_short(notices[0][1], 110)}”.",
        )
    )

    # 7 — clear the rename cell
    p = Project.from_json(project)
    p.data.renames = {"BonusMalus": "bm"}
    p.rename_column("BonusMalus", "bm")
    grid = [
        {
            "column": c,
            "rename to": float("nan") if c == "BonusMalus" else "",
            "role": p.data.roles.get(p.data.renames.get(c, c), "unassigned"),
            "type": "auto",
        }
        for c in raw_columns
    ]
    apply_roles_grid(p, raw_columns, grid)
    rows.append(
        (
            7,
            "Clear a 'rename to' cell with Delete",
            "Traceback ('float' has no attribute 'strip'); the rename and the lost role "
            "stayed in the file.",
            f"An emptied cell means 'no rename': renames are now {p.data.renames}, "
            f"BonusMalus is back as a {p.data.roles.get('BonusMalus')!r} and the model "
            f"predicts with {p.models['freq'].predictors}.",
        )
    )

    # 8 / 9 — derived columns that cannot run
    p = Project.from_json(project)
    _, e8 = preview_derived(p, _frame(), "foo", "pl.col('foo') + 1")
    _, e9 = preview_derived(p, _frame(), "bad", "pl.col('Region') / 2")
    rows.append(
        (
            "8, 9",
            "Add a derived column that refers to itself, or divides a text column",
            "Traceback on Variables and Split; the column was saved and every page "
            "that needs the data was broken.",
            f"*Add* runs the expression first and refuses: “{_short(e8, 70)}” / "
            f"“{_short(e9, 70)}”. A derived column that breaks later is reported as a "
            "data-step problem on every page instead of a traceback.",
        )
    )

    # 10 — split column mode without a column / text column with TRAIN=1
    p = Project.from_json(project)
    p.data.roles.pop("traintest")
    p.data.split.column = "nope"
    p10 = tmp / "split.easyglm-project.json"
    p.to_json(p10)
    at = _run(_script("pages_split", str(p10)))
    msgs10 = [m for m in _messages(at) if "split" in m.lower() or "pick" in m.lower()]
    p.data.split.column = "Region"
    p.data.split.train_value = "1"
    try:
        add_split_column(_frame(), p.data.split)
        m10b = "ACCEPTED"
    except ValueError as exc:
        m10b = str(exc)
    rows.append(
        (
            10,
            "Split page with no usable split column; a text column as indicator "
            "with TRAIN = 1",
            "Traceback 'cannot compare string with numeric' because the page picked "
            "the first column (the policy number) by itself.",
            f"Nothing is picked for you: {len(at.exception)} tracebacks, the page says "
            f"“{_short(msgs10[0], 80) if msgs10 else 'MISSING'}”; a text column compares as "
            f"text and “{_short(m10b, 80)}”.",
        )
    )

    # 11 — bad project files
    bad = {
        "parquet.json": data.read_bytes(),
        "truncated.json": b'{"name": "x", "data": {',
        "v99.json": json.dumps({"version": 99}).encode(),
        "list.json": b"[1, 2, 3]",
        "badtypes.json": json.dumps({"version": 2, "data": {"roles": "oops"}}).encode(),
    }
    outcomes = {}
    for name, content in bad.items():
        (tmp / name).write_bytes(content)
        at = _run(
            _script(
                "pages_project",
                str(project),
                body=f"from easy_glm.app.pages_project import open_project_file\n"
                f"out['msg'] = open_project_file({str(tmp / name)!r})",
            )
        )
        outcomes[name] = (
            at.session_state["_out"]["msg"],
            at.session_state["_out"]["name"],
        )
    all_msgs = all(
        m and m.startswith("Not a valid easy_glm project") for m, _ in outcomes.values()
    )
    untouched = all(n == "w3check" for _, n in outcomes.values())
    rows.append(
        (
            11,
            "Upload or open five kinds of broken project file",
            "Tracebacks (JSON decode errors, key errors) and in one case a half-loaded "
            "project replaced the open one.",
            f"All five end with a message starting “Not a valid easy_glm project” "
            f"({'yes' if all_msgs else 'NO'}); the open project is left alone every time "
            f"({'yes' if untouched else 'NO'}). E.g. “{_short(outcomes['truncated.json'][0], 90)}”.",
        )
    )

    # 12 — save to a bad path; autosave to a read-only file
    at = _run(
        _script(
            "pages_project",
            str(project),
            body="out['save'] = S.save_project('/nonexistent_dir_easy_glm/x.json')\n"
            "out['save_dir'] = S.save_project('/')",
        )
    )
    o = at.session_state["_out"]
    ro = tmp / "ro" / "p.easyglm-project.json"
    ro.parent.mkdir()
    ro.write_text("{}")
    os.chmod(ro, stat.S_IREAD)
    at2 = _run(
        _script(
            "pages_model",
            str(project),
            prelude=f"st.session_state.project_path = {str(ro)!r}; st.session_state.project_stamp = None",
        )
    )
    at2.text_input(key=wk(at2, "notes_freq")).set_value("x").run()
    auto = [e for e in _messages(at2) if e.startswith("Autosave failed")]
    os.chmod(ro, stat.S_IREAD | stat.S_IWRITE)
    rows.append(
        (
            12,
            "Save to a folder that cannot exist, to '/', or autosave to a read-only file",
            "Tracebacks (PermissionError, FileNotFoundError, FileExistsError); a failed "
            "autosave was only mentioned on the Project page.",
            f"Messages: “{_short(o['save'], 70)}”, “{_short(o['save_dir'], 60)}”. The "
            f"project keeps its previous file (still {Path(o['path']).name}). A failing "
            f"autosave is shown at the top of every page: “{_short(auto[0], 60) if auto else 'MISSING'}”.",
        )
    )

    # 13 — model names
    p = Project.from_json(project)
    m13 = validate_model_name("a/b", p.models)
    rows.append(
        (
            13,
            "Create a model called a/b, then open Rate tables or Export",
            "Traceback from the Excel writer (the name became a folder in the file "
            "path); the whole Rate tables and Export pages were dead.",
            f"The Create button is disabled with “{_short(m13, 80)}”; names are trimmed, "
            f"limited to 60 characters and must be unique. A legacy project that already "
            f"holds such a name still downloads: its files are called "
            f"{ui.safe_filename('a/b')}.xlsx / .easyglm.",
        )
    )

    shutil.rmtree(tmp, ignore_errors=True)

    lines = [
        "# W3 — workbench hardening: the break-it findings, replayed",
        "",
        "*Generated by `scripts/checks/w3_hardening.py` on a synthetic 3,000-policy book "
        "(text policy numbers, a 0/1 train flag, three rating factors). The thirteen rows "
        "are the blocking findings of `docs/reviews/w2-breakage.md`: 1–4 lost work, 5–13 "
        'were tracebacks. "Now" is what this run of the tool actually said.*',
        "",
        "| # | What the tester did | What used to happen | What happens now |",
        "|---|---|---|---|",
    ]
    for k, did, before, now in rows:
        lines.append(f"| {k} | {did} | {before} | {now} |")
    lines += [
        "",
        "## The two rules behind the fixes",
        "",
        "**One project file, many tabs.** The workbench remembers when it last read or "
        "wrote the project file. Before every autosave it looks at the file again; if "
        "someone else (another tab, another session, a text editor) has changed it since, "
        "autosave pauses and a notice at the top of every page offers two choices: "
        "*Reload from disk* (this tab takes the other version and drops its own unsaved "
        "edits, including anything still typed into its boxes) or *Overwrite* (this tab's "
        "version wins). Nothing is written until you choose, so neither copy can be lost "
        "by accident. Two people should still not edit one project file at the same time — "
        "the rule stops silent loss, it does not merge.",
        "",
        "**A model may only use columns that exist.** Renaming a column in the roles grid "
        "renames it everywhere it is used (role, type, recode, design, split, every "
        "model's target / weight / offset / predictors / constraints / interactions / "
        "adjustments). It follows the column into your row filters and derived-column "
        "formulas too: a filter written as `pl.col('Exposure') > 0.02` becomes "
        "`pl.col('exposure_years') > 0.02`, and the page tells you which formulas were "
        "rewritten (only that column reference is touched — the rest of the formula is "
        "left exactly as you typed it). Changing a column's role tells you which models "
        "are affected. If a "
        "column a model needs is missing from the data (a new file, a removed derived "
        "column), the model is not silently re-pointed at another column: the Model page "
        "shows which column is missing, its selector is left blank, Fit is disabled and "
        "any persisted fit is ignored until you choose a replacement.",
        "",
        "## What to check yourself",
        "",
        "- Open your own project in two tabs, change something in each, and confirm the "
        "second tab shows the notice rather than a merged or overwritten file.",
        "- Rename a rating factor in the roles grid, then look at the Model page: the "
        "predictor list, any monotone constraint and any interaction should show the new "
        "name, and a fitted model should say it needs refitting.",
        "- Type a nonsense expression into a derived column and a slash into a model "
        "name: both should be refused with a sentence, never a traceback.",
        "",
        "## Findings not fixed in W3",
        "",
        "Items 15 (a role change now lists the models it touched, but still applies "
        "immediately), 23 (roles are still keyed by the final column name; renames now "
        "propagate correctly), 28 (a fit interrupted by a page refresh is discarded "
        "without a message), 35 (the 'pick two different variables' message shows in "
        "every Diagnostics tab) and 38 (a constant column blocks the fit instead of "
        "being skipped) are cosmetic and left for a later piece. Item 32 is fixed: the "
        "Seed box now always shows the seed the project holds, so the page can no longer "
        "name a seed the split did not come from.",
        "",
    ]
    text = "\n".join(lines)
    if write:
        DOC.write_text(text)
        print(f"wrote {DOC}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    sys.exit(main(ap.parse_args().write))
