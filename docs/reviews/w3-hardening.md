# Review of piece W3 — workbench hardening (fixes for the breaker session)

*Reviewer: independent. Branch `release-0.4`, commits `e27e81d`, `5496f28`, `06c5329`,
`01cae9e` (`git diff 88fd757..HEAD -- src tests scripts docs/checks`). Contract:
`docs/reviews/w2-breakage.md` findings 1–13 (blocking), `docs/RELEASE_0.4_PLAN.md`
§"GUI quality" ("never a raw traceback, never lose the project"). Date 2026-09-03.*

## 1. Verdict

**Accepted — no blocking items. Five should-fix items, none of which touches the engine
or the fitted numbers.**

All thirteen blocking findings are genuinely fixed, and I reproduced each one against the
code rather than trusting the builder's tests. I ran a hostile session of my own against a
live server (Streamlit 1.57, Chromium 1500 × 1000, port 8641, the French-motor 50k fixture
plus a synthetic `current_premium`, the breaker's project shape): 38 assertions, 37 passed,
and the server log contains **zero** tracebacks. Loading a data file whose columns do not
match the project renders all eight pages with no traceback, names the orphaned roles on
Variables and the missing columns on Model. Two browser tabs on one project file now
behave in both directions — *Reload from disk* takes the other tab's version and drops this
tab's typed edits, *Overwrite* wins, and autosave resumes after either choice (the e2e
covers only Reload; I checked Overwrite by hand). "New empty project" leaves the previous
file byte-for-byte unchanged. A broken project file, a bad save path, a slash in a model
name, a derived column that cannot run, a split column named after a data column and a text
indicator compared with `1` all end in a sentence.

`Project.rename_column` is the strongest part of the change: on a project that used one
column in every possible place (role, type override, recode, design entry, split column,
leakage lists, and a model's target/weight/predictors/monotone/interaction/main-effect
adjustment/interaction-cell adjustment) a single rename moved every reference — the old name
does not appear anywhere in the serialised project afterwards. Renaming onto an existing
name (or onto a derived column's name) is refused with nothing saved; clearing the cell —
as `""`, `"   "` or the editor's `NaN` — restores the original name, its role and the
model's predictor list.

The missing-column rule holds end to end. With a fitted model and a data source that no
longer has one predictor: the Model page shows the missing name, leaves the selector blank,
disables Fit, `get_run` returns `None`, no page claims "Fitted and up to date", the
persisted `.pkl` is **not** deleted, and pointing the project back at the original file
restores the persisted fit with no refit.

What is left is a cluster of "the message is wrong or incomplete" items. The worst of them
(S1) is a red banner that keeps saying "edits are not being saved" long after they are being
saved again — a false alarm rather than false reassurance, which is why it is not blocking.

## 2. Blocking

**None.** For the record, the two candidates I weighed and rejected:

* *The mtime-only conflict check* (S3). On this machine's filesystem it provably cannot miss
  a write, so it does not fail the rule it was written for; the risk is a filesystem the tool
  has not been shown to run on.
* *A momentarily unreadable data file deleting the persisted fit* (S4). This is real loss of
  work, but it is byte-identical to the code at `88fd757` — it predates W3 and is outside
  this diff. It should not ship in 0.4; it should not hold up this merge.

## 3. Should-fix

### S1 — "Autosave failed" never goes away, so the tool lies about not saving

*What.* `state.touch()` appends `Autosave failed: …` to `session_state.errors` and nothing
ever removes it. `ui.show_errors()` draws every entry as `st.error` at the top of **every**
page, and `main.py` adds a sidebar `st.error("Autosave is failing — edits are not being
saved.")`. Only `save_project()` (the explicit Save buttons) prunes those entries; the
autosave path does not, although `show_errors`' own docstring promises "they clear when the
cause is fixed (a successful save drops them)".

*Failure scenario.* The actuary's project file is on a share that goes read-only for a
minute (or they `chmod` it, or the share drops). They get the correct red banner. The share
comes back; autosave silently starts working again. The banner and the sidebar note stay for
the rest of the session, on every page, telling them their work is not being saved while it
is. The rational response is to stop trusting the tool, or to redo work by hand.

*Verified.* Live browser: after `chmod 444` → edit Notes → banner appears; `chmod 644` →
edit Notes → the file on disk holds `after chmod back` (so autosave recovered) **and**
"Autosave failed" is still on the page. Also reproduced under AppTest.

*Exact fix.* In `state._write_project()` (or right after the successful
`_write_project(path)` in `touch()`), drop the autosave entries the way `save_project`
already does:

```python
st.session_state.errors = [
    e for e in st.session_state.get("errors", []) if not e.startswith("Autosave")
]
```

Putting it in `_write_project` covers both callers. Add the AppTest case: fail, recover,
assert the banner is gone.

### S2 — a rename does not follow the column into row filters or derived expressions

*What.* `Project.rename_column` rewrites roles, types, recodes, design, split, leakage lists
and every model reference. It does not touch `data.filters` or the `expr` of
`data.derived`, and nothing warns.

*Failure scenario.* The breaker's own project has the filter `pl.col('Exposure') > 0.02`.
The actuary renames `Exposure` to `exposure_years` in the roles grid. The rename is accepted
and saved; from then on every page that needs the data says *"The data steps fail: unable to
find column Exposure…"*. The project is not lost and the message points at the Variables
page, so this is not a crash and not data loss — but the user is told their data steps are
broken without being told that the rename they just made is what broke them, and the only
way out is to notice the filter and edit it.

*Verified.* Probe on a project with a filter and a derived column naming the renamed column:
`apply_roles_grid` returns `changed=True` with the single notice
`'Exposure' renamed to 'exposure_years' in model(s): freq`; `p.data.filters` and the derived
`expr` are untouched.

*Exact fix.* Cheapest honest version: in `apply_roles_grid`, before applying the rename,
scan `p.data.filters` and `[d.expr for d in p.data.derived]` for the old name (a
`re.search(rf"""['"]{re.escape(old)}['"]""", text)` is enough for `pl.col('X')`) and emit a
`("warning", …)` notice naming each one: *"the row filter `pl.col('Exposure') > 0.02` still
refers to Exposure — edit it on the Row filters tab"*. Rewriting the expressions
automatically is the nicer answer but is string surgery on user code; the warning is the
safe one. Either way, add the missing sentence to the "A model may only use columns that
exist" rule in `docs/checks/w3-hardening.md`, which currently reads as though a rename
follows the column everywhere.

### S3 — the two-tab check compares only the modification time, not size or content

*What.* `state._file_mtime` / `file_changed_on_disk` compare `os.stat(path).st_mtime_ns`
against the value stored when this session last read or wrote the file. There is no size
comparison and no hash of the bytes last written.

*Measured.* On this machine (APFS, `/System/Volumes/Data`) six back-to-back
`write_text` calls produced six distinct `st_mtime_ns` values ~40 µs apart, and rewriting the
same content still changed the timestamp. So on the tested platform the check **cannot** miss
a write within the same second, and finding 2 is genuinely fixed there. But `st_mtime_ns` is
only as fine as the filesystem: NFS and some SMB mounts round to the second, FAT/exFAT to
two seconds. An actuary keeping projects on a departmental share is a likely deployment.

*Failure scenario / verified.* I forced the condition rather than guessing: another session
writes the file, then `os.utime` puts the timestamp back to the value this tab had seen.
This tab's next edit autosaves straight over the other session's work, with no conflict
notice and no message — exactly finding 2, on a coarse-timestamp filesystem.

*Exact fix.* Store a cheap identity instead of a bare mtime, and treat any difference as a
conflict:

```python
def _file_stamp(path):
    st = os.stat(path)
    return (st.st_mtime_ns, st.st_size, hashlib.blake2b(Path(path).read_bytes(),
                                                        digest_size=16).hexdigest())
```

Project files are small (the breaker's is a few kB), so hashing on every `touch()` is free.
The rest of the machinery — `conflict`, `resolve_conflict`, the notice — is unchanged.

### S4 — a data file that is unreadable for one moment permanently deletes the persisted fit

*What.* `state.load_persisted_run` wraps unpickling **and** `prepared_frame()` in one
`try`, and its `except` calls `_remove_run_file(target)`. When the data file cannot be read,
`prepared_frame()` returns `None`, the code raises `ValueError("data not available")` and
the persisted `.pkl` and its sidecar are deleted.

*Failure scenario.* The project's parquet lives on a network share. The share blips (or the
file is locked, or permissions change for a minute). The page correctly says *"Could not
load …: Permission denied"*. When the file comes back, the fit is gone: the model shows "Not
fitted yet" and a cross-validated lasso has to be run again.

*Verified.* Fit, then `chmod 000` the data file, render the Model page once, `chmod 644`
back: `rev.easyglm-runs/*.pkl` is empty and stays empty; the model does not come back fitted.

*Not a W3 regression* — `git show 88fd757:src/easy_glm/app/state.py` has the identical block.
It is in this review because W3 is the piece whose brief is "never lose work", and because
the new "a persisted run is ignored while a column is missing" rule makes people spend more
time on pages where a data problem is being reported.

*Exact fix.* Only delete when the *file* is unusable. Move the data check out of the
delete-on-failure block:

```python
    try:
        with target.open("rb") as fh:
            run = pickle.load(fh)
        if not isinstance(run, ModelRun) or run.name != model:
            raise ValueError("not a persisted run for this model")
        if not _design_matches(p, model, run):
            raise ValueError("design no longer matches")
    except Exception:            # a corrupt or foreign pickle: drop it
        _remove_run_file(target)
        return None
    df = prepared_frame()
    if df is None:               # data temporarily unavailable: keep the file
        return None
```

### S5 — the lumped bucket is still called "Other / Unknown" in the actuary's table

*What.* The `Other`-clash fix works where it was aimed: `run.encoder_for` /
`DesignSpec.from_frame` give the encoder `other_label="Other (lumped)"` when a real level is
called `Other`, and a fit that used to fail with *"other_label 'Other' clashes with a level"*
now succeeds. But `engine.models.level_label` renders the fallback row as the fixed string
`"Other / Unknown"`, so the words "Other (lumped)" never reach a rate table, the Excel or the
Rate tables page.

*Verified.* 6,000-policy book with levels `Other`, `A`, `B` and 40 rare levels,
`min_level_share = 0.05`: the fit succeeds, `enc.other_label == "Other (lumped)"`, and
`rate_tables(run.fit)["Area"]` has the rows `Other`, `A`, `B`, `Other / Unknown`. The
actuary reads two adjacent rows called "Other" and "Other / Unknown" and has no way to tell
which is their real level and which is the leftovers bucket.

*Exact fix.* Pass the encoder's `other_label` through to the label: in `core/tables.py`, for
a `CategoricalEncoder`, render the `from_=None, to_=None` row as `enc.other_label` (falling
back to today's `"Other / Unknown"` when the label is the default). The builder's stated rule
— "a real level named `Other` makes the lumped label `Other (lumped)`" — is then true where
it matters, on the page and in the workbook.

## 4. Nits

1. **The "new project" confirmation flag is sticky.** `confirm_new_project` is set on the
   first click and cleared only by the successful second click. Click once, change your mind,
   do something else on the page, and a *single* later click closes the project with no
   confirmation. (Verified under AppTest: path `None`, name `untitled` after one click.) The
   project file is safe — that is finding 1's fix — so this only costs you your place. Clear
   the flag whenever any other Project-page control is used.
2. **The button label lags a rerun.** `label = "Click again to start a new project" if
   confirm else "New empty project"` is computed before the click that sets `confirm`, and no
   `st.rerun()` follows, so the button the user actually sees after the first click still says
   "New empty project". The warning carries the instruction, so it works; the label is dead
   code in practice.
3. **`save_project` creates directories.** `Path(path).parent.mkdir(parents=True,
   exist_ok=True)` means a typo'd project path silently creates a folder tree instead of
   saying "that folder does not exist".
4. **Windows device names pass `validate_model_name`.** `CON`, `NUL`, `PRN`, `AUX`, `COM1`
   are accepted and `safe_filename` leaves them alone; `CON.xlsx` is not a usable file name on
   Windows. One line in `_MODEL_NAME_BAD`'s neighbourhood.
5. **`errors` is now append-only.** `show_errors` used to clear the list after drawing it; it
   no longer does, and `persist_run`'s `Could not persist the fit: …` has the same problem as
   S1 — one transient failure is shown on every page for the rest of the session.
6. **Missing-column pages say the wrong reason.** With a predictor missing, Rate tables and
   Diagnostics show *"No fitted model yet — fit one on the Model page"* and Export shows
   *"This model is not fitted (or its spec changed)"*. True in the letter, misleading in
   spirit — the fit exists and is deliberately being ignored. Reuse the Model page's sentence
   ("`freq`: predictor(s) not in the data: …") on those pages.
7. **`get_run` reads the prepared cache without checking its hash.** `st.session_state
   ["prepared"]` may be one rerun stale, so `missing_columns` is occasionally evaluated
   against the previous frame's columns. It errs safe (a run is hidden, never shown), but it
   can blank the "Fitted" chip for one render after a rename.
8. **Only-numeric selectors exclude booleans.** `_column_pick` filters on
   `NUMERIC_DTYPES`, so a `Boolean` 0/1 column can no longer be chosen as target or weight.
   Correct for finding 14, slightly over-tight.
9. **Item 32 is still visible.** The Seed box gained `help="0 – 10000"` but can still display
   a number the project does not hold, so the page can show a seed that did not produce the
   split you are looking at.

## 5. The four questions put to the reviewer

**The widget-key change is consistent.** 105 call sites use `S.widget_key(...)`. Grepping
every `key=` in `src/easy_glm/app/` leaves exactly four that do not carry the token:
`ui.py:81` `key="conflict_reload"` and `ui.py:88` `key="conflict_overwrite"` (buttons — no
value to leak, and a stable key is what lets the e2e and the tests find them), and
`pages_model.py:74` / `ui.py:162`, which take `key` from their callers — and every caller
(`_column_pick` from `_config`, `run_selector` from Rate tables and Diagnostics) passes
`S.widget_key(...)`. The remaining unkeyed widgets are buttons and defaults whose Streamlit
identity already changes with their arguments. `set_project`'s blanket drop keeps exactly the
right things: the `_`-prefixed keys survive (so `ui.flash`'s `_flash` and `main.py`'s
`_cli_loaded` bootstrap guard are not destroyed), while `model_current`,
`confirm_new_project` and the diagnostics search results are dropped, which is what you want
when another project is loaded. Popping widget keys mid-render (`open_project_file` is called
from a button handler on the page whose widgets already exist this run) does not raise — the
live session reopened a project from the Project page with no exception.

**Nothing that should be kept is lost.** Within one project the token never changes: I
navigated Design → Explore → Design in one session and `project_token` was identical. The
selected variable *is* forgotten across that navigation — but that is Streamlit garbage
collecting widget state for widgets it did not render, not the token: a control-group script
with a hard-coded `key="fixedkey"` loses its value in exactly the same way. Pre-existing,
unchanged by W3.

**Do the AppTest helpers hide anything?** One small thing. `wk(at, name)` derives the key
from the live `at.session_state["project_token"]`, so a spurious `set_project` mid-session
would rotate every key and no test would notice — there is no test asserting the token is
stable while you stay in one project. I checked it by hand and it is.

**Multi-tab.** Answered in full above and in the table: both directions verified in a real
browser and under AppTest, autosave resumes after Reload *and* after Overwrite, and the other
tab correctly sees the conflict on its next edit. mtime granularity is the one caveat (S3):
measured safe here, no size or hash fallback.

## 6. Findings 1–13

| # | Status | How I verified it |
|---|---|---|
| 1 New empty project overwrote the open file | **fixed** | Live browser: first click warns "Click the button again"; after the second click the sidebar drops the old path, and typing a new project name leaves `reviewer.easyglm-project.json` byte-for-byte identical. AppTest: `project_path` becomes `None`, project `untitled`. |
| 2 Two tabs overwrite each other | **fixed** (caveat S3) | Live browser, two pages on one file: tab 2's edit is refused with the notice, disk still holds tab 1's text; *Overwrite* writes tab 2's version and autosave resumes; tab 1 then sees the notice, *Reload* shows tab 2's text and autosave resumes. AppTest: same both ways, plus a third-party text-editor edit is detected. |
| 3 Renamed target silently re-pointed to the id column | **fixed** (see S2) | `rename_column` on a project using one column in roles, types, recodes, design, split, leakage lists, predictors, monotone, an interaction, a main-effect adjustment and an interaction-cell adjustment: every reference moved, old name absent from the serialised project. Model page with a target column that is gone: blank selector, `target column 'claims' is not in the data`, Fit disabled, `get_run` → `None`, no "Fitted and up to date". |
| 4 Random split named after a data column | **fixed** | Live browser: *"'ClaimNb' is already a column in the data; the random split would overwrite it"*, the project keeps `traintest`. A blank name gives "The split column needs a name". `add_split_column` refuses independently, so a hand-edited file cannot slip through. |
| 5 New file with different columns → traceback on Variables and Split | **fixed** | Live browser: loaded a parquet with `DrivAge`→`1st_age`, `Region`→`région/zone`, `VehAge` dropped, then visited Variables, Split, Model, Design, Rate tables, Export, Diagnostics, Explore — **0** tracebacks; Variables lists the orphaned roles, Model says which columns are missing. Server log: 0 tracebacks. |
| 6 Rename onto an existing name | **fixed** | `apply_roles_grid`: refused with *"another column already has that name. Rename not saved."*, `renames` empty, the role kept. Also refused when the target name belongs to a derived column. |
| 7 Clearing a "rename to" cell → `float has no strip` | **fixed** | `apply_roles_grid` with the cell as `NaN`, `""` and `"   "`: all mean "no rename"; `renames` empties, the role and the model's predictor list come back to the original name. |
| 8 / 9 Derived column that cannot run | **fixed** | Live browser: `pl.col('no_such_column') * 2` → message, nothing written to the project file. AppTest: self-reference and `pl.col('Region') / 2` both refused, `derived` stays empty, a good expression is added with role `predictor`. |
| 10 "Existing indicator column" auto-picked the id column | **fixed** | Live browser: switching to indicator mode picks nothing and says so; `IDpol` keeps role `id` in the saved file; choosing the text column `Region` while TRAIN is `1` gives *"No row of 'Region' equals the TRAIN value '1'"*, never a compare error. |
| 11 Five kinds of broken project file | **fixed** | Live browser: truncated JSON → *"Not a valid easy_glm project…"*, the open project survives in the sidebar. AppTest: parquet-as-JSON, truncated, `{"version": 99}`, `[1,2,3]`, `roles` as a string — all one message, project untouched; missing path and a folder both handled. |
| 12 Save / autosave to a bad path | **fixed** (caveat S1) | Live browser: `/nonexistent_dir_easy_glm/x.json` → *"Could not save…"*, the project keeps autosaving to its own file; a read-only project file reports *"Autosave failed: …"* on the page and recovers once writable. The banner not clearing is S1. |
| 13 Model named `a/b` killed Rate tables and Export | **fixed** | Live browser: typing `a/b` disables Create and shows "Model name cannot contain '/'", nothing written to the file. AppTest: a legacy project already holding `a/b` still renders Rate tables and Export and downloads as `a_b.xlsx`. |

## 7. The items left open (15, 23, 28, 32, 35, 38)

I agree they are cosmetic and none of them shows an actuary a wrong number, with one
half-exception.

* **15** is really *fixed*, and the doc undersells it: changing a role now flashes
  *"VehAge was removed from model freq_v1: its role is now ignore"* and drops the affected
  interactions with a notice — which is exactly the fix the breaker asked for. That it applies
  immediately rather than asking first is a preference, not a defect.
* **23** — roles are still keyed by the final name, but the failure it caused (a failed
  rename losing the role) is gone, because renames now propagate atomically and a refused
  rename saves nothing. Architectural tidiness, no user-visible effect.
* **28** — a fit interrupted by F5 is discarded silently. Nothing wrong is shown; the model
  simply reads "Not fitted yet", which is true.
* **32** is the half-exception. The Seed box can still display a number the project does not
  hold, so the page can claim the split came from seed 99999999999 when it came from 7. It is
  a reproducibility annoyance rather than a wrong result — the split, the balance table and
  the fit are all the stored seed's — but of the six it is the only one I would call
  *misleading* rather than cosmetic. The `help="0 – 10000"` tooltip is the minimum; an
  explicit "seed must be 0–10000; still using 7" would close it.
* **35** — an error repeated in sibling Diagnostics tabs. Ugly, harmless.
* **38** — a constant column blocks the fit with the accurate message *"Cannot derive knots
  for 'constant' (constant or all-null on train)"*. Refusing to fit is the safe direction.

The genuinely misleading items that *were* fixed deserve a mention because they were the ones
an actuary would have been burned by: 14 (a text id can no longer be a weight), 18 (a cleared
"map to" cell no longer creates a level literally called `nan`), 19 ("no level reaches the
minimum level share (60.00% of training rows; 4 distinct values)" instead of "all null"), 21
(relativities must now be > 0, and the Model page labels metrics "metrics include N manual
adjustment(s)"), 24 (`nan`/`inf`/`1e400` knots refused, so the project file stays valid JSON),
26 (percentages beyond ±10,000 % print as "—" and a base-rate override more than 100× the
fitted rate warns), 27 (autosave failures now show on every page — see S1 for the other half)
and 17 (uploads land in `<project>.easyglm-data/` next to the project, with a caption saying
so, instead of a temp folder the project will not find after a reboot).

## 8. The actuary document (`docs/checks/w3-hardening.md`)

Good: it is short, it is organised as "what I did / what used to happen / what happens now",
it quotes the tool's actual sentences, it explains the two rules in English, it tells the
reader three things to try themselves, and it names what was *not* fixed. The "One project
file, many tabs" paragraph is the best thing in it — including the honest line "Two people
should still not edit one project file at the same time — the rule stops silent loss, it does
not merge."

Three things to change:

1. **It contains code.** Row 1 says "path shown: None"; row 6 says "nothing saved
   (changed=False)"; rows 5 and 11 contain `['DrivAge', 'Region']` and a bare "(yes)"; rows 8/9
   quote `pl.col('foo') + 1`. `None`, `changed=False` and `(yes)` are generator artefacts that
   an actuary will trip over. The polars expressions are defensible — they are what the user
   typed — but the rest should be English ("the new project has no file yet", "nothing was
   saved", "in all five cases").
2. **The rename rule overstates itself** (S2). "renames it everywhere it is used" needs one
   sentence: *"Row filters and derived-column formulas still spell the old name; if you have
   one, edit it after the rename."*
3. **The lumped-bucket claim does not reach the page** (S5). Either fix the label or drop the
   claim.

Nothing in it is dishonest, and the "Findings not fixed in W3" section is more candid than it
needed to be.

## 9. Missing tests

* No browser-level coverage of the **roles grid** — the one control behind findings 3, 6, 7
  and 23. `_helpers.edit_grid_cell` can only reach the *last* column of a grid, and "rename to"
  is column 2 of 7, so the rename path is covered by `apply_roles_grid` unit tests and by
  AppTest, never by a real canvas edit. Worth a helper that takes a column index.
* No test that `project_token` is **stable** while you stay in one project (see §5).
* No test that autosave **recovers** after a failure and that the banner clears (S1).
* No test that a rename with a filter or derived expression naming the old column is handled
  (S2).
* `tests/e2e/test_breakit.py` exercises only the *Reload* branch of the conflict; the
  *Overwrite* branch and the resumption of autosave after it are only in AppTest.

## 10. What I re-ran

| What | Command | Result |
|---|---|---|
| Full suite, repo venv (Streamlit 1.57) | `.venv/bin/python -m pytest -q -p no:randomly` | **431 passed**, 17 warnings, 171.4 s |
| The four app suites, Streamlit 1.63 venv | `pytest -q tests/test_app.py tests/test_app_state.py tests/test_w2_pages.py tests/test_w3_hardening.py` | **114 passed**, 15.6 s |
| The same four, repo venv (1.57) | as above | **114 passed**, 4.9 s |
| Lint | `.venv/bin/python -m ruff check .` | All checks passed |
| Format (the documented one) | `.venv/bin/python -m black --check .` | 87 files unchanged |
| Format (other tool, FYI) | `ruff format --check .` | 5 files flagged; every hunk is a ruff-vs-black disagreement (assert wrapping, one `# noqa` line), no content. Black is what `AGENTS.md`/`CONTRIBUTING.md` document, so this is not a finding |
| Golden | `pytest -q tests/test_golden.py -p no:randomly` | **7 passed**, 1.05 s |
| Golden baseline untouched | `git diff 88fd757..HEAD -- tests/fixtures tests/test_golden.py` and `git show --stat` on each of the four commits | empty — no golden number and no fixture changed |
| Persona + break-it e2e (documented command) | `EASY_GLM_E2E=1 EASY_GLM_SERVER_PYTHON=.venv/bin/python <playwright venv>/bin/python -m pytest -q tests/e2e -p no:randomly` | **3 passed**, 86.1 s (`test_breakit`, `test_actuary_rate_review`, `test_data_scientist_model_comparison`); every server fixture's `"Traceback" not in log` assertion passed |
| Check script reproduces its page | `PYTHONPATH=src .venv/bin/python scripts/checks/w3_hardening.py` (stdout mode; `--write` is the writing one) | Output identical to the committed `docs/checks/w3-hardening.md` except a trailing newline from `print()` |
| My own reviewer probes (AppTest) | 12 cases: two-tab Overwrite/Reload/third-party-editor/forced-same-mtime, rename-everywhere, rename-refused/blank/whitespace, rename-vs-filters, missing predictor across seven pages and back, `get_run` with no prepared frame, widget state across pages, autosave recovery | 10 passed, 2 failed — both deliberate: the widget-navigation one documents pre-existing Streamlit widget GC (a fixed-key control script behaves identically), the other is S1 |
| My own live break-it session | server on **port 8641** (Streamlit 1.57, repo venv), Chromium 1500 × 1000, findings 1, 2, 4, 5, 10, 11, 12, 13, 14, 25, 27 replayed | 38 assertions, **37 passed**, 1 failed (S1); server log **0 tracebacks**; port released |
| Filesystem timing measurement | six consecutive writes to one file | `st_mtime_ns` all distinct, ~40 µs apart (APFS) — the basis for S3's assessment |
| Other end-to-end checks | recode default `Other` + `Area` predictor fits and exports (finding 22); a real level `Other` with lumping active | fit succeeds, `other_label = "Other (lumped)"`, but the table row reads `Other / Unknown` (S5) |

I changed no file except this review, and `git status --short` is clean apart from it.
