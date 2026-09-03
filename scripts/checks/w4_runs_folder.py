"""Actuarial check for piece W4 — the persisted-run folder, and the rest of
the second breaker session (docs/reviews/w3-breakage-2.md).

Replays the findings against the current workbench with Streamlit's test
harness on a small synthetic motor book — two "browser tabs" are two AppTest
sessions on one project file — and writes a plain-language page: what the
tester did, what used to happen, what the tool says now, and the three rules
that decide what may be written to or deleted from the runs folder.

Usage: python scripts/checks/w4_runs_folder.py [--write]
  --write regenerates docs/checks/w4-runs-folder.md; otherwise it is printed.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "checks" / "w4-runs-folder.md"
N = 2000


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
            "IDpol": np.arange(N),
            "ClaimNb": rng.poisson(mu * expo).astype(float),
            "Exposure": expo,
            "DrivAge": age,
            "BonusMalus": bm,
            "Region": region.astype(object),
            "constant": np.full(N, 3.0),
            "traintest": (rng.random(N) < 0.7).astype(int),
        }
    )


def _project(data_path: Path):
    from easy_glm.workflow import Project

    p = Project(name="w4check")
    p.data.source.type = "parquet"
    p.data.source.path = str(data_path)
    p.data.roles = {
        "ClaimNb": "target",
        "Exposure": "weight",
        "IDpol": "id",
        "DrivAge": "predictor",
        "BonusMalus": "predictor",
        "Region": "predictor",
        "constant": "ignore",
        "traintest": "split",
    }
    p.data.split.mode = "column"
    p.data.split.column = "traintest"
    p.new_model("freq", divide_target_by_weight=True)
    p.models["freq"].penalty.alpha = 0.002
    p.models["freq"].penalty.cv = None
    return p


def _script(
    page: str, project_path: str, *, fit: bool = False, prelude: str = ""
) -> str:
    return f"""
import importlib
import streamlit as st
from easy_glm.app import state as S
from easy_glm.workflow import Project

S.init_state()
if not st.session_state.get("_loaded"):
    S.set_project(Project.from_json({project_path!r}), {project_path!r})
    st.session_state._loaded = True
    {prelude}
if {fit!r} and not st.session_state.get("_fitted"):
    S.fit_model("freq")
    st.session_state._fitted = True
importlib.import_module("easy_glm.app." + {page!r}).render()
st.session_state["_project"] = S.project()
"""


def _run(script: str, timeout: int = 240):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(script, default_timeout=timeout)
    at.run()
    return at


def _texts(at) -> str:
    parts = [w.value for w in at.warning] + [e.value for e in at.error]
    parts += [i.value for i in at.info] + [s.value for s in at.success]
    return "\n".join(parts)


def _short(text: str, n: int = 170) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _pkls(folder: Path) -> list[str]:
    return sorted(f.name for f in folder.glob("*.pkl")) if folder.exists() else []


def _button(at, label: str):
    return [b for b in at.button if b.label == label][0]


def main(write: bool) -> int:
    from easy_glm.workflow import Project
    from easy_glm.workflow.project import validate_model_name

    tmp = Path(tempfile.mkdtemp(prefix="easy_glm_w4_"))
    data = tmp / "policies.parquet"
    _frame().write_parquet(data)
    project = tmp / "w4check.easyglm-project.json"
    runs = tmp / "w4check.easyglm-runs"
    _project(data).to_json(project)
    path = str(project)
    rows: list[tuple[str, str, str, str]] = []

    def reset() -> None:
        for f in list(runs.glob("*")) if runs.exists() else []:
            f.unlink()
        _project(data).to_json(project)

    # ---------------------------------------------------------------- 1
    tab_a = _run(_script("pages_model", path, fit=True))
    tab_b = _run(_script("pages_model", path))
    before = _pkls(runs)
    tab_a.text_input(key=wk(tab_a, "notes_freq")).set_value("A wins").run()
    tab_b.text_input(key=wk(tab_b, "notes_freq")).set_value("B loses").run()
    tab_b.button(key=wk(tab_b, "fit_freq")).click().run()
    after_fit = _pkls(runs)
    fit_note = [w.value for w in tab_b.warning if "this tab only" in w.value]
    _button(tab_b, "Delete").click().run()
    after_delete = _pkls(runs)
    del_note = [
        w.value for w in tab_b.warning if "removed from this tab only" in w.value
    ]
    still_there = "freq" in json.loads(project.read_text())["models"]
    fresh = _run(_script("pages_model", path))
    rows.append(
        (
            "1 (data loss)",
            "Two tabs on one project. Tab A saves an edit, so tab B gets the "
            "conflict notice. In tab B, without resolving it: **Fit**, then **Delete**.",
            "Tab B's Fit rewrote the run file and tab B's Delete removed the saved "
            "fit of a model the project on disk still contained. Every other tab, and "
            "the next session, said 'Not fitted yet.' for a model fitted a minute ago. "
            "No message anywhere.",
            f"Nothing in the folder changes: {len(before)} fit before, "
            f"{len(after_fit)} after the Fit, {len(after_delete)} after the Delete "
            f"(same file). Fit says “{_short(fit_note[0] if fit_note else 'MISSING', 120)}” "
            f"and Delete says “{_short(del_note[0] if del_note else 'MISSING', 120)}”. "
            f"The project file still holds the model: {'yes' if still_there else 'NO'}, "
            f"and a fresh session still shows "
            f"“{'Fitted and up to date' if 'Fitted and up to date' in _texts(fresh) else 'NOT FITTED'}”.",
        )
    )

    # ---------------------------------------------------------------- 2
    reset()
    tab_a = _run(_script("pages_model", path))
    tab_b = _run(_script("pages_model", path))
    tab_a.number_input(key=wk(tab_a, "alpha_freq")).set_value(0.004).run()
    tab_a.button(key=wk(tab_a, "fit_freq")).click().run()
    a_files = _pkls(runs)
    tab_b.button(key=wk(tab_b, "fit_freq")).click().run()
    both = _pkls(runs)
    third = _run(_script("pages_model", path))
    rows.append(
        (
            "2 (data loss)",
            "Tab A changes alpha to 0.004 and fits (the project on disk now says "
            "0.004). Tab B, still showing 0.001, clicks **Fit**. A third tab is opened.",
            "Tab B's fit deleted tab A's run — the only one matching the project on "
            "disk — under the 'latest run per model' rule. The saved project said "
            "'Not fitted yet.'; two simultaneous fits could empty the folder.",
            f"Both fits are kept ({len(a_files)} file after A's fit, {len(both)} after "
            f"B's; A's file is still there: "
            f"{'yes' if a_files and a_files[0] in both else 'NO'}). The third tab opens "
            f"the saved project and says "
            f"“{'Fitted and up to date' if 'Fitted and up to date' in _texts(third) else 'NOT FITTED'}” "
            f"for alpha "
            f"{third.session_state['_project'].models['freq'].penalty.alpha:g}.",
        )
    )

    # ---------------------------------------------------------------- 3
    reset()
    at = _run(_script("pages_model", path))
    at.text_input(key=wk(at, "model_new_name")).set_value("freq_v2").run()
    _button(at, "Create").click().run()
    picked = at.selectbox(key=wk(at, "model_select")).value
    at.text_input(key=wk(at, "notes_freq_v2")).set_value("a note for v2").run()
    p_after = at.session_state["_project"]
    rows.append(
        (
            "3 (misleading)",
            "Model page → New model name `freq_v2` → **Create**, then type a note.",
            "“Model 'freq_v2' created” — and the picker, the whole configuration panel "
            "and the Fit button stayed on the champion. The note, the predictors and "
            "the fit all went to the old model.",
            f"The picker moves to the model that was created (it now shows "
            f"{picked!r}), the New-model box is cleared, and the note lands on "
            f"freq_v2: notes are {p_after.models['freq_v2'].notes!r} on freq_v2 and "
            f"{p_after.models['freq'].notes!r} on freq.",
        )
    )

    # ---------------------------------------------------------------- 28
    reset()
    started = _run(_script("pages_model", path))
    at = _run(
        _script("pages_model", path).replace(
            'st.session_state["_project"] = S.project()',
            "S._mark_fit_started('freq', S.run_key(S.project(), 'freq'))\n"
            'st.session_state["_project"] = S.project()',
        )
    )
    markers = sorted(f.name for f in runs.glob("*.fitting"))
    nxt = _run(_script("pages_model", path))
    notice = [w.value for w in nxt.warning if "interrupted" in w.value]
    del started
    rows.append(
        (
            "28 (was open)",
            "Start a cross-validated fit and press F5 after three seconds.",
            "The page came back saying 'Not fitted yet.'; nothing said a fit had been "
            "started, so a ten-minute fit could be lost without a word.",
            f"A fit writes a marker next to where its result will be saved "
            f"({len(markers)} marker after the interrupted fit) and removes it once "
            f"the fit is saved. The next session says: "
            f"“{_short(notice[0] if notice else 'MISSING', 150)}”",
        )
    )
    for f in runs.glob("*.fitting"):
        f.unlink()

    # ---------------------------------------------------------------- 38
    reset()
    p = Project.from_json(project)
    p.data.roles["constant"] = "predictor"
    p.models["freq"].predictors.append("constant")
    p.to_json(project)
    at = _run(_script("pages_model", path))
    at.button(key=wk(at, "fit_freq")).click().run()
    dropped = [w.value for w in at.warning if "constant" in w.value]
    rows.append(
        (
            "38 (was open)",
            "Add a column that holds one value everywhere (`constant`) as a predictor "
            "and fit.",
            "“Fit failed: Cannot derive knots for 'constant'” — one useless column "
            "blocked the whole model, with no hint on the Design page.",
            f"The fit runs without it and says which column it left out: "
            f"“{_short(dropped[0] if dropped else 'MISSING', 150)}”. The fit itself is "
            f"“{'Fitted and up to date' if 'Fitted and up to date' in _texts(at) else 'MISSING'}”.",
        )
    )

    # ---------------------------------------------------------------- 31
    reset()
    at = _run(_script("pages_model", path))
    ticked_before = at.checkbox(key=wk(at, "div_freq")).value
    at.selectbox(key=wk(at, "wgt_freq")).set_value("(none)").run()
    box = at.checkbox(key=wk(at, "div_freq"))
    rows.append(
        (
            "31 (was open)",
            "Model page → set Weight to `(none)`, then look at “Divide target by "
            "weight”.",
            "The box was greyed out but stayed **ticked**, while the project held "
            "`divide_target_by_weight: false` — the page said the model divides by a "
            "weight it does not have.",
            f"With a weight the box is ticked ({ticked_before}); with the weight "
            f"cleared it is unticked and disabled (ticked: {box.value}, disabled: "
            f"{box.disabled}), and the project holds "
            f"{at.session_state['_project'].models['freq'].divide_target_by_weight}.",
        )
    )

    # ---------------------------------------------------------------- 32 / 5
    reset()
    p = Project.from_json(project)
    p.data.split.mode = "random"
    p.data.split.column = "split_flag"
    p.data.split.seed = 99_999
    p.to_json(project)
    at = _run(_script("pages_split", path))
    shown = at.number_input(key=wk(at, "split_seed")).value
    at.number_input(key=wk(at, "split_seed")).set_value(-5).run()
    refused = [e.value for e in at.error if "seed" in e.value]
    kept = at.session_state["_project"].data.split.seed
    back = at.number_input(key=wk(at, "split_seed")).value
    _project(data).to_json(project)
    am = _run(_script("pages_model", path))
    am.number_input(key=wk(am, "alpha_freq")).set_value(1e9).run()
    alpha_msg = [e.value for e in am.error if "alpha" in e.value]
    rows.append(
        (
            "32 and 5 (misleading)",
            "Type a seed of 99999 (and then -5) into the Seed box; paste `1e9` into "
            "the alpha box.",
            "The box kept showing the number that had been typed while the project "
            "kept the old one — the page named a seed the split did not use and a "
            "penalty the fit did not use, with no message.",
            f"A seed the split can use is taken and shown ({shown}); one it cannot is "
            f"refused, and the box is put back to the seed in use "
            f"(project {kept}, box {back}): “{_short(refused[0] if refused else 'MISSING', 120)}”. "
            f"The alpha box behaves the same way: "
            f"“{_short(alpha_msg[0] if alpha_msg else 'MISSING', 120)}”",
        )
    )

    # ---------------------------------------------------------------- 33
    reset()
    p = Project.from_json(project)
    p.data.split.mode = "random"
    p.data.split.column = "split_flag"
    p.data.split.fraction = 1.0
    p.to_json(project)
    at = _run(_script("pages_split", path))
    clamped = [w.value for w in at.warning if "0.50–0.95" in w.value]
    on_disk = json.loads(project.read_text())["data"]["split"]["fraction"]
    rows.append(
        (
            "33 (caveat)",
            "Hand-edit the project file to a training fraction of 1.0 and open the "
            "Split page.",
            "The slider showed 0.95, the file was quietly rewritten to 0.95 and the "
            "sentence explaining it was lost in the page redraw.",
            f"The file is still repaired (it now holds {on_disk}), but the page says "
            f"so and the message survives the redraw: "
            f"“{_short(clamped[0] if clamped else 'MISSING', 160)}”",
        )
    )

    # ---------------------------------------------------------------- 10
    reset()
    at = _run(_script("pages_split", path))
    at.selectbox(key=wk(at, "split_col")).set_value("Region").run()
    warn = [
        w.value
        for w in at.warning
        if "split indicator" in w.value or "Region" in w.value
    ]
    before_role = at.session_state["_project"].data.roles.get("Region")
    at.button(key=wk(at, "split_role_btn_Region")).click().run()
    after_role = at.session_state["_project"].data.roles.get("Region")
    rows.append(
        (
            "10 (caveat)",
            "Split page → choose a rating factor (`Region`) as the train/holdout "
            "indicator column.",
            "The column lost its predictor role immediately and silently; the Model "
            "page then complained about a predictor that had gone missing.",
            f"Nothing changes on the pick: the role is still {before_role!r} and the "
            f"page asks first — “{_short(warn[0] if warn else 'MISSING', 150)}”. After "
            f"the confirming click the role is {after_role!r}.",
        )
    )

    # ---------------------------------------------------------------- 24 / 30
    reset()
    at = _run(_script("pages_design", path))
    at.selectbox(key=wk(at, "design_detail_var")).set_value("DrivAge").run()
    at.text_area(key=wk(at, "knots_DrivAge")).set_value("30, 40, 999999").run()
    at.button(key=wk(at, "apply_knots_DrivAge")).click().run()
    knot_warn = [w.value for w in at.warning if "999999" in w.value]
    rows.append(
        (
            "24 (caveat)",
            "Design page → knots `30, 40, 999999` on a column whose largest value is "
            "79.",
            "Accepted in silence: the top band had no training rows and nothing said "
            "so.",
            f"Still accepted (the knots are saved), but flagged: "
            f"“{_short(knot_warn[0] if knot_warn else 'MISSING', 170)}”",
        )
    )
    reserved = {n: validate_model_name(n) for n in ("CON", "NUL", "PRN", "COM1")}
    rows.append(
        (
            "30 (caveat)",
            "Create a model called `CON` (or `NUL`, `PRN`, `COM1`).",
            "Accepted; its downloads were called `…_CON_rate_tables.xlsx`, which "
            "Windows cannot write.",
            "Refused before the model is created: "
            + "; ".join(
                f"{n} → “{_short(m or 'ACCEPTED', 90)}”" for n, m in reserved.items()
            ),
        )
    )

    typo = tmp / "typo_dir" / "deep" / "x.json"
    at = _run(
        _script("pages_project", path).replace(
            'st.session_state["_project"] = S.project()',
            f'st.session_state["_err"] = S.save_project({str(typo)!r})',
        )
    )
    rows.append(
        (
            "11 (cosmetic)",
            "Project & data → type a project path whose folders do not exist → "
            "**Save project**.",
            "“Saved …” — the workbench created the folder tree and moved the project "
            "into it, so a typo quietly filed your project somewhere new.",
            f"Refused, with the folder named: "
            f"“{_short((at.session_state['_err'] or 'SAVED ANYWAY').replace(str(tmp), '<your folder>'), 150)}”. The "
            f"folder was not created: "
            f"{'yes' if not (tmp / 'typo_dir').exists() else 'NO'}.",
        )
    )

    lines = [
        "# W4 — the runs folder, and the second breaker session's findings",
        "",
        "*Generated by `scripts/checks/w4_runs_folder.py` on a synthetic "
        f"{N:,}-policy book. Two browser tabs are two test sessions on one project "
        "file. “Now” is what this run of the tool actually said.*",
        "",
        "| # | What the tester did | What used to happen | What happens now |",
        "|---|---|---|---|",
    ]
    for num, did, before_text, now in rows:
        lines.append(f"| {num} | {did} | {before_text} | {now} |".replace("\n", " "))
    lines += [
        "",
        "## Where your fits live, and who may change them",
        "",
        "Every fit is saved as a file in a folder next to your project file, called "
        "`<your project>.easyglm-runs/`. That is what lets you close the browser and "
        "come back to a fitted model instead of waiting for it to fit again. The "
        "folder belongs to the **project file**, not to a browser tab: if you open the "
        "same project in two tabs — or two people open it from a shared drive — both "
        "of them are writing into the same folder. Before this piece, each tab tidied "
        "the folder using its own idea of the project, which is how a tab that was one "
        "edit behind could delete the fit belonging to the version that was saved.",
        "",
        "Three rules now decide what a tab may do:",
        "",
        "1. **A tab showing the conflict notice may fit, but nothing it does reaches "
        "the folder.** The conflict notice (“this file was changed by another browser "
        "tab”) already pauses saving the project file; it now pauses the fits folder "
        "too. You can still press Fit and look at the results — the page says the "
        "result is being kept in this tab only — and you can still remove a model from "
        "this tab's copy of the project, which changes nothing on disk. As soon as you "
        "choose *Reload from disk* or *Overwrite*, saving resumes.",
        "",
        "2. **Nothing is deleted while your tab is out of step with the file on "
        "disk.** Saving a fit only ever adds a file, so it is allowed whenever there "
        "is no conflict notice; deleting one can destroy work you cannot get back "
        "without refitting, so it needs your tab to be looking at the same project "
        "file everyone else is.",
        "",
        "Resolving the conflict — either way — lets this tab save again straight "
        "away, including into the fits folder. It does not tidy the folder there "
        "and then: the version that lost keeps its fit until the next time that "
        "model is fitted, which is when the tidy-up below runs.",
        "",
        "3. **The fit that matches the saved project is never tidied away.** The "
        "folder keeps one fit per model — the newest — so it does not grow for ever. "
        "That tidy-up now spares two things: the fit that matches the project *as it "
        "is saved on disk*, and the fit that matches what your own tab is showing. "
        "Everything else (a fit of a penalty you tried and moved on from, a fit of a "
        "model you deleted) is removed when the next fit of that model is saved.",
        "",
        "In short: **what is kept** is the newest fit of each model, plus the fit the "
        "saved project points at; **what is pruned** is a fit that matches neither, "
        "and any file belonging to a model that no longer exists in either your "
        "project or the saved one; **a paused tab** writes and deletes nothing at all.",
        "",
        "One project file open in one tab is still the simplest way to work. These "
        "rules are there so that the day you forget, you lose nothing.",
        "",
        "## A fit that never finished",
        "",
        "If you reload the page (or the app stops) while a model is fitting, the "
        "result is gone — there is no way to catch a half-finished fit. What the "
        "workbench can do is tell you. Before a fit starts it leaves a small marker "
        "file next to where the result will be saved, and removes it once the result "
        "is safely there. If a session finds a marker with no result beside it, it "
        "says so once: *“A fit of freq was interrupted … Fit it again on the Model "
        "page.”* The model is honestly shown as not fitted, instead of quietly looking "
        "like a model that was never fitted at all.",
        "",
        "One caveat, because there is no way for one tab to see another tab's work in "
        "progress: **if you open a second tab while a fit is running, that second tab "
        "may report the running fit as interrupted.** Look at the Model page before "
        "you refit — if the first tab is still working, let it finish. For the same "
        "reason a tab never removes a marker that is not its own until it is five "
        "minutes old: taking it away would cost the tab that is really fitting its own "
        "warning. The notice is drawn once per session, and it keeps appearing in new "
        "sessions until that model is fitted again, because until then it is true.",
        "",
        "## What to check yourself",
        "",
        "- Open your project in two tabs, fit in both, and look in the "
        "`…easyglm-runs/` folder: you should see both fits, and both tabs should still "
        "show their own numbers.",
        "- Make one tab show the conflict notice, then press Fit and Delete in it. "
        "Nothing in the folder should change — no new files, no timestamps moving, not "
        "even the small marker files a fit leaves while it runs — and the tab should "
        "say so in both cases.",
        "- Create a second model and check the picker moves to it before you type "
        "anything else.",
        "",
        "## Findings left open",
        "",
        "Two findings are not fixed here. **6** — choosing a column such as a premium "
        "as the *offset* is accepted without a warning, although the offset is added "
        "to the linear predictor and so is exponentiated; deciding when an offset is "
        "implausible is a judgement about your data, and the sentence that would warn "
        "you has to be written with an actuary. **8** — a saved fit is matched to its "
        "data file by size and modification time, so restoring that file from a backup "
        "makes every fit look stale and everything has to be refitted; matching on the "
        "contents instead means reading the whole book (which can be several GB) every "
        "time a page is drawn, so it needs its own piece of work. Both are recorded in "
        "`docs/reviews/w3-breakage-2.md`. Nothing else in that report is left open.",
        "",
    ]
    text = "\n".join(lines)
    if write:
        DOC.write_text(text)
        print(f"wrote {DOC}")
    else:
        print(text, end="")  # the page, byte for byte, with no extra newline
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    sys.exit(main(ap.parse_args().write))
