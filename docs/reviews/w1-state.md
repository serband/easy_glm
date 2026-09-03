# Review of piece W1 — workbench state: exploration sample vs full-data fits (D2) and persisted fitted runs (D1)

*Reviewer: independent. Branch `release-0.4`, commits `dd56e1b`…`e74adec` (`git diff c1a2e6b..HEAD`).
Contract: `docs/RELEASE_0.4_PLAN.md` §D items 1–2 and R6 (S2, S3), the D1/D2 test rows;
`docs/reviews/00-plan-review.md` S2, S3 and the §5 lines tagged D1/D2. Date 2026-09-03.*

## 1. Verdict

**Changes requested — two blocking items, both one-line fixes.**

The substance is right. Every fit, diagnostic, rate table and the leakage report read the
full prepared frame; only the Explore charts, the Design-page exposure/rate previews and the
Variables-page level counts read the seeded sample. I verified on the Design page itself
(AppTest, sample of 500 rows on a 3,000-row book) that the knots shown in the knots box are
exactly `DesignSpec.from_data(full_train)` and *not* the sample's knots (the two lists differ
at 15 of 19 positions). `model_hash`, `data_hash`, `source_hash` and the leakage key ignore
`sample_rows` / `sample_seed`; the leakage report still caps itself at 50,000 training rows
internally. Persisted runs restore with a maximum holdout-prediction difference of exactly 0,
in 0.006 s on the 3,000-row test book and 0.04 s on the 50,000-row fixture; a touched mtime,
a rewritten file, a spec change, a monkeypatched version and a corrupt pickle all miss and
clean up; the write is temp-file + rename; an adjustment removed from the project after the
fit is gone from the restored run (relativity 2.0 → 1.4498, `config.adjustments` empty);
unsaved projects write nothing anywhere. Suite 355 green, ruff and black clean, golden test
and fixtures untouched, the check script reproduces `docs/checks/w1-state.md` to the digit.

The two blocking items are (1) the **Derived columns → Preview** button on the Variables page
now shows an error instead of a preview, with or without a sample — a polars truthiness bug
introduced by this piece; (2) a model whose name starts with another model's name plus a
hyphen (`freq` and `freq-cap`) has its persisted fit **deleted every time the other model is
fitted or restored**, so D1 ("runs survive a reload") silently fails for a natural naming
pattern. Both fall under the plan's merge rule (a broken control; silent loss of a persisted
artefact).

## 2. Blocking

### B1. The "Preview" button for a derived column is broken (Variables page)

**What.** `src/easy_glm/app/pages_variables.py:273`:

```python
base = apply_variables((S.raw_sample() or raw).head(2000), p.data)
```

`S.raw_sample()` returns a polars `DataFrame`; `DataFrame or raw` calls `DataFrame.__bool__`,
which polars refuses (`TypeError: the truth value of a DataFrame is ambiguous`). The
`except Exception` around it turns the crash into a red box, so the page does not lock, but
the preview never appears. Reproduced by AppTest with a sample of 60 rows *and* with no
sample at all (`raw_sample()` returns the whole frame in that case, still a DataFrame):
`errors shown: ['the truth value of a DataFrame is ambiguous ...']` both times.

**Failure scenario.** The actuary types `pl.when(pl.col('Lic') == 'Q')...` and presses
Preview to check the expression before adding it — the page answers "the truth value of a
DataFrame is ambiguous", which reads as if the expression were wrong. They will either
abandon the derived column or add it unchecked.

**Fix.** `raw_sample()` never returns `None` when `raw` is loaded, so:

```python
sample = S.raw_sample()
base = apply_variables((raw if sample is None else sample).head(2000), p.data)
```

and add the button click to `tests/test_app_state.py::TestSampleVsFull::test_pages_render_with_a_sample`
(set `derived_name` / `derived_expr`, click `derived_preview`, assert `at.error` is empty and
`at.dataframe` or `at.markdown` grew).

### B2. Persisting or restoring model `freq` deletes the persisted fit of model `freq-2`

**What.** `state._run_files(folder, model)` globs `f"{model}-*.pkl"`. For `model="freq"` that
pattern also matches `freq-2-<key>.pkl`, `freq-cap-<key>.pkl`, `freq-v2-<key>.pkl` — any
model whose name is `freq` + hyphen + anything. Both `persist_run` (“remove old files for this
model”) and `load_persisted_run` (“remove stale files for this model”) then delete the other
model's file. Reproduced: models `freq` and `freq-2` fitted → two files; refit `freq` → only
`freq-…pkl` remains; new session → `get_run("freq")` restores, `get_run("freq-2")` is `None`.
The existing test `test_two_models_persist_independently_latest_only` passes only because it
names the second model `freq2`.

**Failure scenario.** The actuary keeps `freq` and `freq-capped` (or `sev` and `sev-2024`)
side by side. Every fit of `freq`, and every reload that restores `freq`, throws away the
persisted `freq-capped` fit; after each browser refresh the second model refits — the reload
promise of D1 is broken for one of the two models and nothing says why. With glob
metacharacters in a name (`freq*`) the deletion reaches every model.

**Fix.** Match the key, not a wildcard — in `_run_files`:

```python
_KEY_RE = re.compile(r"[0-9a-f]{16}\.pkl$")
def _run_files(folder, model):
    prefix = f"{model}-"
    return sorted(f for f in folder.iterdir()
                  if f.name.startswith(prefix) and _KEY_RE.fullmatch(f.name[len(prefix):]))
```

(or put each model in its own sub-folder). Rename the second model in the existing test to
`freq-2` so the test guards this, and reject or sanitise model names containing `/`, `\` or
`:` before they become file names (today `p.new_model("a/b")` makes `persist_run` fail with an
OSError that is only appended to `errors`).

## 3. Should fix

### S1. Opening another project on the Project page overwrites its exploration-sample setting with the previous project's value — and autosaves it

`pages_project.py:78–94`: the sample `number_input` now has `key="sample_rows"` and the page
writes `p.data.sample_rows = int(sample) or None; S.touch()` whenever the widget value differs
from the project. Streamlit keeps a keyed widget's value across reruns and ignores a changed
`value=` default, so after `set_project(B)` (open project, "New project", `--project`) the
widget still holds project A's number. Reproduced: A with sample 60 rendered, then B (sample
`None`) opened in the same session → `B.data.sample_rows == 60` in memory **and in B's JSON
on disk**. The field is exploration-only after this piece, so no fitted number changes, but a
saved project file was silently modified without a user action. Fix: clear the widget in
`set_project` (`st.session_state.pop("sample_rows", None)`) or use an `on_change` callback
instead of the value comparison. Test: the two-project AppTest sequence above.

### S2. Stale persisted files are deleted on *read*, so a transient spec edit erases the last fit

`load_persisted_run` removes every file for the model whose key differs from the current key
before it even checks whether the current key exists. `get_run` is called from `status()`
(sidebar, every rerun), `run_selector`, the Model page etc. So: fit, reload, nudge alpha in the
number box (or toggle a monotone flag to look at it), and the persisted fit is gone the moment
the sidebar re-renders — before any refit, and even if the edit is reverted. In-session the
old run survives via `stale_run`; after a reload it does not, and the "results below are from
the previous fit" message can never appear after a reload. Fix: delete old files only in
`persist_run` (the "latest per model" rule then holds at write time), never in
`load_persisted_run`; a file that fails to unpickle can still be deleted there.

### S3. The temp file name is fixed, so two sessions persisting the same model at once corrupt each other's write

`persist_run` writes `<model>-<key>.pkl.tmp` and renames. Two browser tabs on the same project
(the project JSON is already shared) fitting the same model at the same time open the *same*
temp file with `"wb"`; the second truncates the first mid-write, and the first's `replace`
leaves the second's `replace` with `FileNotFoundError` → "Could not persist the fit" in
`errors`. Worst case is a truncated pickle, which the loader treats as a miss, so no wrong
numbers — but the atomicity claimed by temp+rename is not there. Fix: unique temp name
(`tempfile.NamedTemporaryFile(dir=folder, prefix=f"{model}-", suffix=".tmp", delete=False)` or
`f"{target.name}.{os.getpid()}.tmp"`), then `os.replace`.

### S4. The version component of the key is the *installed metadata* version, which is stale in an editable checkout — add a format constant

`easy_glm.__version__` and `_versions()["easy_glm"]` come from `importlib.metadata`; in this
venv that reports **0.2.2** while `pyproject.toml` says 0.3.0 and the code is 0.4-dev (the
sidecar in my run: `"easy_glm": "0.2.2"`). For a released wheel the number is right and the
key does what S2 asks. In a development checkout it does not move between commits, so a pickle
written by yesterday's `ModelRun` / `GLMFit` / `RateModel` / `DesignSpec` layout is loaded by
today's code. `pickle.load` restores `__dict__` without `__init__`: a field added since the
pickle was written is simply absent and raises `AttributeError` at first use — which may be on
a page render, *outside* the `try` in `load_persisted_run`. `_design_matches` and
`rebuild_rate_model` (which re-scores train and holdout) exercise most of the object graph, so
the practical window is small, but not closed. Fix: add `PERSIST_FORMAT = 1` to `run_key` and
bump it whenever any pickled class changes shape; run `pip install -e .` so metadata matches
`pyproject.toml`; and set `pyproject.toml` to `0.4.0.dev0` now so released and dev keys can
never collide. State this in AGENTS.md ("bump `PERSIST_FORMAT` when …").

### S5. A refused adjustment on load costs a refit instead of a drop

`load_persisted_run` calls `rebuild_rate_model`, which raises `AdjustmentError` for an
adjustment the RateModel refuses (e.g. a non-positive value on a linear band); the generic
`except` deletes the file and returns `None`, and the Model page then refits, and only then
`_drop_refused_adjustment` runs. Wrap the rebuild in the same `while True / except
AdjustmentError → _drop_refused_adjustment` loop as `refresh_adjustments`. (I could not trigger
this on the categorical test book — a relativity of 0.0 on `Region` is accepted — so it is
code-read only.)

## 4. Nits

- **Explore "Rows" metric** (`pages_explore.py:41–48`) shows `u["n"]` from the sample under
  the bare label "Rows"; the caption above explains, but the label should say "Rows (sample)"
  when `S.is_sampled()`. Same for the Design-page chart titles (`"{var}: 19 knots → 20 bins"`
  over sample-sized bars — fine, the knots count is full-data; but the y-axis exposure is the
  sample's). No chart claims full-data *numbers* while showing the sample; the captions cover
  it.
- **README.md:190** still says "optional sample"; make it "optional exploration sample
  (Explore / previews only)" and mention `<project>.easyglm-runs/` in the Project & data row.
- `run.project_snapshot` in a restored run is the fit-time snapshot (my probe: it still lists
  the removed adjustment) while `run.config` is current. Either refresh the snapshot in
  `rebuild_rate_model` or document that `project_snapshot` is the spec *at fit time*.
- `_versions()` runs four `importlib.metadata.version` lookups (0.66 ms) on every `get_run`
  miss; `status()` calls `get_run` for every unfitted model on every rerun. Cache it with
  `functools.lru_cache` — tests monkeypatch `S._versions` at module level, which still works.
- `train_frame()` re-filters the full frame on every call (Design page once per render,
  `_design_matches` once per restore). Fine today; cache on `data_hash` before G lands.
- `to_json` / `spec_hash` serialise `np.int64` / `np.bool_` via `default=str` as `"10"` /
  `"True"`; the hash agrees before and after a round trip (verified), but the JSON then holds a
  string where an int belongs. Pre-existing, not introduced here; coerce in `to_dict`.
- "Save project" to a new path leaves the old `<old>.easyglm-runs/` folder behind — harmless,
  worth one sentence in the CHANGELOG.
- `pyproject.toml` version is still 0.3.0 on a 0.4 branch.

## 5. Missing tests

- The Derived-column Preview click (B1) — none of the AppTests presses a button on the
  Variables page.
- Two models where one name is a hyphenated prefix of the other (B2).
- A Design-page test that reads the knots text box and compares it with
  `DesignSpec.from_data(full_train)` **and** asserts it differs from the sample's knots
  (`test_knots_come_from_the_full_training_rows` compares `encoder_for(train_frame)` with
  `from_data(train_frame)` — same frame both sides — so it never sees the page).
- Open-project sequence on the Project page (S1).
- A run persisted with an adjustment, then the adjustment removed from the project → restored
  run has none (my probe passes; the suite only tests the *adding* direction).
- A persisted run of a model that has since been deleted from the project: `load_persisted_run`
  returns `None` early and never removes the orphan file.
- The plan's D1 row asks for a Playwright reload; `tests/e2e/` is empty. AppTest with a fresh
  session is a fair stand-in, but say so in the plan or add the e2e test when the harness
  exists.
- `persist_run` on a read-only folder (the `except OSError` branch is `pragma: no cover`).

## 6. What I re-ran

All on `e74adec`, interpreter `.venv/bin/python` (Python 3.14.7, polars 1.40.1, glum 3.4.1,
numpy 2.4.5, streamlit 1.57.0).

- Full suite: **355 passed** in 167 s. `tests/test_app.py` 20 + `tests/test_app_state.py` 18 =
  38 passed in 1.8 s (second run, warm). `ruff check src tests scripts`: all checks passed;
  `black --check`: 69 files unchanged. `git diff c1a2e6b..HEAD -- tests/test_golden.py
  tests/fixtures`: empty.
- `scripts/checks/w1_state.py`: CHECK PASSED; output identical to `docs/checks/w1-state.md`
  (50,000 policies, sample 10,000, train 34,887 = fit train rows, holdout A/E 0.9935 · Gini
  0.3316, restored in 0.04 s, max diff 0, not restored after the data change, 0 files after).
- Nine pages × {no fit, fit} on a 3,000-row book with `sample_rows=500`, a
  `DrivAge × Region` interaction and `BonusMalus` piecewise-linear: no exceptions on any of the
  18 renders; `train_rows` 2,086 (= full training count) on every fitted page; sample captions
  appear on Variables, Explore and Design only.
- Design-page knots box for `DrivAge` (sample 500 → 338 training rows in the sample) equals
  `DesignSpec.from_data(train_frame)` knots exactly; the sample's own knots differ at 15 of 19
  positions. Caption: "Knots and levels are derived from all 2,086 training rows; the preview
  charts use the exploration sample (338 rows)."
- Hash stability: `model_hash` equal before and after `json.loads(json.dumps(to_dict(),
  default=str))` with numpy knots (`np.quantile`), `np.int64` n_bins/seed and `np.float64`
  interaction exposure; reversed top-level key order → same hash; `0.1+0.2` vs `0.3` → different
  (as it must be).
- Persistence probes (AppTest, fresh session per step): restore max |Δ| = 0 in 0.006 s;
  adjustment 2.0 persisted then removed from the project → restored 1.4498, 0 adjustments on
  `run.config`; sidecar shows the resolved data path, size, `mtime_ns`, versions; pickle 22 KB
  (3k-row book) and 181 KB (50k fixture with an interaction and a linear term); project file
  copied to another folder without its runs folder → not restored, original still restores;
  unsaved project → no `.pkl` anywhere under the temp tree; `sample_rows` larger than the data →
  `is_sampled()` False and no caption.
- B1 reproduced with and without a sample (error text shown in place of the preview).
  B2 reproduced (`freq` refit deletes `freq-2-….pkl`; new session restores `freq` only).
  S1 reproduced (B's file on disk ends with `sample_rows: 60`).

## 7. Answers to the questions put to the reviewer

- **Concurrent sessions.** Write is temp+rename but with a shared temp name (S3). Reads are
  safe: a half-written or truncated file is a cache miss.
- **Project file moved.** The runs folder is a sibling of the project file, so moving the
  `.easyglm-project.json` alone loses the persisted fits (refit, no error); moving the data file
  changes the resolved path in the key (miss + cleanup on next load); copying project + folder
  + data to the same paths on another machine restores only if versions match too. This should
  be one sentence in the CHANGELOG ("keep the `…-runs` folder next to the project file").
- **Pickling the whole `ModelRun` across `rebuild_rate_model`.** Robust in practice: the pickle
  holds `spec`, `fit` (glum estimator), `rate_model`, polars `tables`, `metrics`, the snapshot;
  `rebuild_rate_model` rebuilds `rate_model`, `tables`, `metrics` and `config` from `fit` and
  the *current* project, which is exactly the source-of-truth rule S2 asked for. The remaining
  exposure is layout drift between pickle and code (S4).
- **Security note.** Present in the module docstring and the CHANGELOG ("trusted local content,
  like derived-column expressions"); AGENTS.md mentions the folder but not that it is pickles —
  add the half-sentence. README has nothing yet (nit).
- **Version string.** `easy_glm.__version__` is read from package metadata, not hard-coded.
  With stale metadata the key under-invalidates in a dev checkout (S4); with correct metadata
  it behaves as designed. It never over-invalidates.

## 8. Re-check (commits `a9eb142`…`aa924da`, `git diff e74adec..aa924da`) — 2026-09-03

### Final verdict: **Approved.**

Both blocking items and all five should-fix items are addressed in the diff, each with a
test that would have caught the original finding. I re-ran my own probes against `aa924da`
rather than only the builder's tests.

### Blocking items

- **B1 (Derived-column Preview) — fixed.** `pages_variables.py:273` now reads
  `sample = S.raw_sample(); base = apply_variables((raw if sample is None else sample).head(2000), p.data)`.
  Re-run of my AppTest: with a 60-row sample and with no sample, clicking Preview raises no
  exception, shows no error box, and the dataframe count grows from 2 to 3 (the preview table).
  Guarded by `TestReviewFollowUps::test_derived_preview_works_with_and_without_a_sample`.
- **B2 (`freq` / `freq-2` collision) — fixed.** File names are now
  `<sha1(model)[:10]>-<key>.pkl` (`_model_tag`, `run_file`), and `_run_files` accepts only
  `<tag>-<16 hex>.pkl`. Re-run: `freq` and `freq-2` fitted → two files; refit `freq` → still
  two files; new session restores both. A model named `f*` (glob metacharacter) persists as a
  third file and all three restore. Deleting `freq-2` and `f*` from the project and refitting
  `freq` removes their orphans (`_remove_orphans`), leaving one file. The existing two-model
  test now uses `freq-2` and also asserts no `.tmp` file is left behind.

### Should-fix items

- **S1 (sample widget carried into the next project) — fixed.** `set_project` pops
  `sample_rows`, `src_path`, `proj_name`, `proj_path` from session state. Re-run of the
  A-then-B sequence: B shows `sample_rows=None` in memory and on disk; a user then setting 100
  on B takes effect (100 in memory and on disk). Guarded by
  `test_sample_widget_does_not_leak_into_another_project`.
- **S2 (stale file deleted on read) — fixed.** `load_persisted_run` no longer deletes any file
  whose key merely differs; "latest per model" is enforced in `persist_run` after a successful
  save. Re-run: fit, new session, set alpha 0.01 → `get_run` None, `status()["fitted"]`
  False, the file is still there; revert to 0.002 → restored from disk. Refit with the new
  alpha → exactly one file, the old one gone. A corrupt pickle is still removed (folder empty
  afterwards). The check script and `docs/checks/w1-state.md` were updated accordingly ("1
  (kept until a new fit replaces it)" after the data change) and the document explains the
  rule in one plain sentence. Guarded by
  `test_transient_spec_edit_does_not_erase_the_persisted_fit`.
- **S3 (shared temp-file name) — fixed.** Temp name is
  `<target>.<pid>.<uuid8>.tmp`, `os.replace`, and a `finally` that unlinks a leftover temp
  file. No `.tmp` files remain after any of my runs.
- **S4 (stale version in the key) — fixed as asked.** `PERSIST_FORMAT = 1` is in `run_key`
  and the sidecar, with a comment saying when to bump it; AGENTS/CHANGELOG mention the
  rule. The `pyproject.toml` version (0.3.0) and the stale editable-install metadata (0.2.2)
  are unchanged — release-process items, not this piece.
- **S5 (refused adjustment on load) — fixed.** `load_persisted_run` now runs
  `rebuild_rate_model` in the same `AdjustmentError → _drop_refused_adjustment` loop as
  `refresh_adjustments`; any other failure is still a cache miss. Guarded by
  `test_refused_adjustment_is_dropped_on_load_not_refitted` (the bad entry is dropped, the
  good one kept, the file remains, no refit).

### Nits

Not addressed (as expected for nits): the Explore "Rows" label, README wording, the stale
`project_snapshot`, `_versions()` caching, numpy ints as strings in `to_json`, and the
orphan `…-runs` folder after "Save as". None affects a number.

### What I re-ran on `aa924da`

- Full suite: **361 passed** in 168 s (my run; the coordinator's independent run on `aa924da` also reports 361 passed). `ruff check`: all checks passed; `black --check`: 69 files
  unchanged. Only `docs/reviews/w1-state.md` differs from the branch after my runs.
- `scripts/checks/w1_state.py`: CHECK PASSED, output equal to the committed
  `docs/checks/w1-state.md` (50,000 policies, sample 10,000, train 34,887 = fit train rows,
  A/E 0.9935 · Gini 0.3316, restored in 0.05 s, max diff 0, not restored after the data change,
  1 file kept).
- The four scenarios above via AppTest (Preview click ×2, `freq`/`freq-2`/`f*` persistence
  and orphan removal, A-then-B sample widget with a subsequent user edit, edit/revert/refit
  with file inspection), plus the corrupt-pickle removal.
