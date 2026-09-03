# W4 review — the persisted-run folder, and the second breaker session

Independent review of piece **W4** on `release-0.4`
(`git diff 0145f80..HEAD -- src tests scripts docs/checks AGENTS.md CHANGELOG.md`,
commits d52ea28, 637944d, 9b92d9e, d15b1eb, ddc57b9; the plan edit d354d33 is
out of scope). The two blocking findings of `docs/reviews/w3-breakage-2.md`
were re-tested with the reviewer's **own** two-session scripts, not the
builder's tests, on a different synthetic book (1,500 policies, different
column and model names), and then attacked with cases the builder's tests do
not cover.

## Verdict

**Accept.** No blocking findings.

Both data-loss findings are genuinely fixed, and they are fixed in the right
place: the rule is *"a tab that is not looking at the project file everyone
else is looking at may add to the fits folder but may not take anything out of
it"*, and it is enforced in the two functions (`persist_run`, `_prune_runs`)
that every path goes through, not patched into the pages. I could not construct
any sequence of two or three sessions — including a resolved conflict, a
deleted project file and two fits written before either tidy-up ran — in which
the fit belonging to the project *as saved on disk* was removed. Finding 3
(Create) is fixed, and so are the ten should-fix items. The full suite, both
Streamlit versions, the linters, the golden test and the persona e2e run are
all green, and the plain-language check page regenerates from its script
unchanged.

Three **should-fix** items remain. None of them can lose a fit; two of them are
places where the tool is *quieter than the page promises*, which is the same
class of complaint the breaker session raised, so they should not go out
unfixed.

---

## Blocking

**None.**

For the record, here is what the two blocking findings do now, measured by the
reviewer's own scripts (the folder is compared by file name, byte size,
modification time in nanoseconds **and** a SHA-1 of the contents — not just by
listing it).

### Finding 1 — a paused tab fits and then deletes

Two sessions on one project file. A saves an edit; B's next edit raises the
conflict notice. Then, in B, **Fit** and **Delete**:

| after | fits folder | project file |
|---|---|---|
| B's Fit | byte-identical (same size, same mtime_ns, same SHA-1) | unchanged |
| B's Delete | byte-identical | unchanged, still holds the model |

B says *"The fit of 'm1' is shown in this tab only …"* and *"Model 'm1' was
removed from this tab only …"*, B's own screen shows its own fit, and a fresh
third session opens the saved project and reports **"Fitted and up to date"**.
The model is gone from B's copy of the project and from nowhere else.

### Finding 2 — a stale tab fits a different penalty

A changes alpha 0.002 → 0.005 and fits (one file; the saved project now says
0.005). B, still on 0.002, clicks Fit: **two files**, A's still present. A third
session opens the saved project and reports "Fitted and up to date" with alpha
0.005. A fourth session then fits alpha 0.009 while in step with the file: the
folder goes back to two files — the saved project's fit (0.005) and the new one
— and the **only** file removed is B's 0.002, exactly the one that matches
neither. The "worse variant" the breaker reported (two tidy-ups running after
both writes, leaving the folder empty) is now impossible: every tidy-up keeps
its own key plus the saved project's key, so it cannot empty the folder. I
reproduced that interleaving directly (both runs written, then both prunes run
late) and both files survived.

### The attacks on the new rules

| What I tried | Result |
|---|---|
| Three tabs, three different penalties, one of them paused | Two files (the two unpaused fits); the paused tab wrote nothing; a fresh session found the saved project's fit |
| Delete the project file from disk mid-session, then edit and refit | No traceback; autosave recreates the file; the refit prunes the older run, which by then matches no saved project. The other tab's fit is re-created when it fits again. No fit of the saved project is lost |
| Resolve the conflict with **Overwrite**, then fit | Correct: the overwriting tab's version is now the file on disk, deleting resumes immediately (`conflict` cleared, stamp refreshed), and the next fit prunes the *other* version's run, as it should |
| Resolve with **Reload from disk**, then Delete | Deleting resumes; the fit files and the model really are removed |
| Does deleting ever stay locked for good? | No. Every path that refreshes the tab's picture of the file — a successful save, an autosave, Reload, Overwrite — clears the pause. I found no state that stays locked once the tab and the file agree |

---

## Should-fix

### S1. A paused tab *does* touch the folder — it deletes another tab's "fit in progress" marker

`state.py: _clear_fit_markers()` and `interrupted_fits()` unlink files with no
pause check at all, and `_clear_fit_markers` matches **every** marker of that
model (`{tag}-*.fitting`), not the one this fit wrote.

Measured: with tab B showing the conflict notice and a live marker in the
folder (tab A mid-fit), clicking **Fit** in B left the fits themselves
untouched but **removed A's marker**. So the rule as written on the check
page — *"A tab showing the conflict notice may fit, but nothing it does reaches
the folder"*, and *"The folder's timestamps should not move"* — is not true as
stated, and the actuary who follows the page's own "What to check yourself"
instruction will see it move.

Nothing is lost but the *notice*: if tab A's fit is then interrupted, nobody is
told. Still, a documented rule that the code does not keep is worth one line of
code.

**Fix:** in `state.py`, return early from `_clear_fit_markers` when
`runs_write_paused()` (a paused tab never wrote a marker, so it has none to
clear), and clear only the marker for the key just fitted rather than globbing
the model; guard the `marker.unlink()` inside `interrupted_fits()` with
`not runs_delete_paused()`.

### S2. "Removed from this tab only" is swallowed exactly when the user most needs it

`pages_model.py` lines 53–62 call `S.remove_model_runs(sel)`, keep its
sentence in `kept`, then call `S.touch()` — and only then `ui.flash(...)`.
`touch()` calls `st.rerun()` the moment it notices the file changed on disk, so
the flash line is **never reached**.

Measured: tab A saves an edit; tab B, which has not edited anything yet and so
has no conflict notice up, clicks **Delete**. Correct on disk (the fit files and
the project file are untouched, verified byte-for-byte), but what tab B shows is
the conflict banner and *"Create a model to start."* — the model has vanished
from the page and **nothing says the deletion did not happen**. The one
sentence written for this case is the one that is lost.

This is also the rule this very piece added to `AGENTS.md` ("any message drawn
immediately before `st.rerun()` must go through `ui.flash`"): the message does
go through `ui.flash`, but after a call that can rerun.

**Fix:** move the `ui.flash("warning" if kept else "info", ...)` line **above**
`S.touch()` in `pages_model.py`. A flash survives a rerun by design, so nothing
else changes.

### S3. A fit running in another tab is announced as "interrupted"

Measured: opening a second session while a fit is running reports
*"A fit of m1 was interrupted — the page was reloaded…"* and deletes the
marker, so the tab that really is fitting loses its own warning if it is then
interrupted. The code's docstring owns this trade-off honestly, but the page
the actuary reads (`docs/checks/w4-runs-folder.md`, "A fit that never
finished") does not mention it at all — it says only "If a session finds a
marker with no result beside it, it says so once."

**Fix (cheapest):** one sentence on the check page — *"If you open a second tab
while a fit is running, that tab may report the running fit as interrupted;
check the Model page before refitting."* **Fix (better):** the marker already
records `pid` and `started_at` — skip a marker whose `pid` is this process and
is still alive, or one written in the last few minutes.

---

## Nits

1. **A fractional seed is taken silently.** Typing `2.5` into the Seed box
   leaves the project holding `2`, on both Streamlit 1.57 and 1.63, with no
   message — while `-1` gets a full sentence ("The seed must be 0 or more…").
   The box and the project do agree, so nothing lies; it is only inconsistent.
2. **Grammar in the knot warning:** *"knot(s) 999999 are above the training
   largest value"* — singular value, plural verb, and "training largest value"
   reads awkwardly. Suggest "knot 999999 is above the largest training value
   (84)".
3. **The project file is written in place** (`project.py:708`,
   `path.write_text(...)`). A tab reading it at that instant sees truncated
   JSON; `_saved_project()` then returns `None` and the tidy-up's protected set
   shrinks to this tab's own key. In practice `runs_delete_paused()` closes the
   window (a changed file pauses deleting), so I could not turn it into a real
   loss — but writing to a temporary file and renaming it would close it for
   good, and would help the conflict check too.
4. **A deleted project file is not "changed".** `file_changed_on_disk()`
   requires the file to still exist, so if it is removed from under the app
   nothing is paused and the next fit prunes freely. Benign as things stand
   (autosave recreates the file with this tab's version first, so the tidy-up
   is consistent with what is on disk), but it is the one gap in rule 2.
5. **After Overwrite, the losing version's fit is not tidied until the next
   fit.** That is what rule 3 says, but the sentence "As soon as you choose
   *Reload from disk* or *Overwrite*, saving resumes" sits next to it and could
   be read as "and the folder is tidied then". Worth half a sentence.
6. `scripts/checks/w4_runs_folder.py` printed to stdout emits one trailing
   blank line more than the committed page; `--write` does not. Harmless.

## Missing tests

`tests/test_w4_runs_folder.py` is good work — one test per finding, and two
real `AppTest` sessions for the two-tab cases rather than a mock. What it does
not cover:

1. **The `.fitting` markers versus the pause rules** (S1). No test asserts that
   a paused or out-of-step tab leaves markers alone. Add one that writes a
   marker, puts a second session into conflict, clicks Fit and asserts the
   folder — markers included — is unchanged.
2. **The Delete message when the file changed but the conflict notice is not up
   yet** (S2). `test_breakage2_01` only exercises the case where the notice is
   already showing, which is exactly the case where `touch()` returns early and
   the flash survives. The failing variant is one line away from the existing
   test.
3. **That deleting resumes.** Every test asserts the *paused* state; none
   asserts that after **Reload from disk** or **Overwrite** a Delete really
   removes the files and the model. That is the half of the rule that could
   silently rot into "nothing can ever be deleted".
4. **Delete moves the picker.** `test_breakage2_03` checks that Create selects
   the new model but nothing checks the companion half added in the same
   commit — that deleting the selected model leaves `model_current` and the
   picker on a model that still exists (and on `None` when the last one goes).
5. **A fractional seed** (nit 1) and a **base-rate override below zero** —
   `ui.number_in_range` is tested through alpha and the seed only.

## What I re-ran, with numbers

| Check | Command | Result |
|---|---|---|
| Full suite | `.venv/bin/python -m pytest -q` (Streamlit 1.57) | **460 passed**, 17 warnings, 2 min 56 s |
| App tests on Streamlit 1.63 | `st163/bin/python -m pytest -q tests/test_app.py tests/test_ui.py tests/test_app_state.py tests/test_w2_pages.py tests/test_w3_hardening.py tests/test_w4_runs_folder.py` | **153 passed**, 21 s (20 / 10 / 24 / 42 / 37 / 20) |
| Lint | `ruff check .` | All checks passed |
| Format | `black --check .` | 89 files unchanged |
| Golden | `git diff 0145f80..HEAD -- tests/test_golden.py tests/fixtures` | empty — untouched (and it passes inside the 460) |
| Persona e2e | `EASY_GLM_E2E=1 EASY_GLM_SERVER_PYTHON=.venv/bin/python <playwright venv> -m pytest tests/e2e -q` | **3 passed**, 1 min 34 s; server stopped, port released |
| Check page reproduces | `.venv/bin/python scripts/checks/w4_runs_folder.py` vs `docs/checks/w4-runs-folder.md` | identical apart from one trailing newline |
| `PERSIST_FORMAT` | `state.py` | bumped 2 → 3 with the "bump whenever the shape of a pickled class changes" docstring, and the same instruction repeated in `AGENTS.md` |
| `AGENTS.md` | runs-folder rule, widget rule, flash-before-rerun rule, `test_w4_runs_folder.py` row | all present |

Reviewer's own scripts (twelve, in the session scratchpad, none committed):
finding 1 with a byte-level folder snapshot; finding 2 with four sessions;
three tabs; a project file deleted mid-session; Overwrite and Reload as
conflict resolutions; Create / rapid Create / double-click Create / Delete down
to an empty project; the interrupted-fit marker in four states (hand-written,
after a success, after a failed fit, after a real `BaseException`
interruption); the seed box at 99999, −1, 2.5, 99999999999 and the alpha box at
1e9, −1, 20 on **both** Streamlit versions; a constant and an all-null
predictor dropped, persisted and restored in a fresh session; every predictor
constant; the split-role confirmation against three consecutive reruns;
`validate_model_name` on eleven reserved and near-reserved names; knots above,
below and inside the training range; a hand-edited fraction of 1.0; a zero-row
book; a one-row exploration sample.

## The check page itself

`docs/checks/w4-runs-folder.md` is accurate on what is kept, what is pruned and
what a paused tab may do, with one exception (S1: a paused tab does move the
folder's timestamps, because of the markers) and one omission (S3: a fit
running in another tab can be reported as interrupted). The three rules are
stated in the order an actuary would ask them, the "In short" paragraph is the
right summary, and the two findings left open (6, the implausible offset, and
8, matching a fit to its data file by modification time) are named with an
honest reason for each. Fix S1 and S3 and the page is correct as written.
