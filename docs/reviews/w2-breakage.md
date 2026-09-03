# W2 breakage report — easy_glm Workbench (release-0.4 @ c5da2ee)

Breaker session against the live app: `streamlit run src/easy_glm/app/main.py`
(Streamlit 1.57, repo venv), port 8631, driven with Playwright/Chromium at
1500×1000 (plus a 375×667 pass). Starting project "breaker": the French-motor
50k fixture plus a synthetic `current_premium`, roles as in the e2e template
(ClaimNb target, Exposure weight, IDpol id, 8 predictors), recode Area E/F→D,
filter `Exposure > 0.02`, random split 70 % seed 7, one model `freq_v1`
(Poisson, claims / exposure, lasso, fixed alpha 0.001, 7 predictors). The
project was restored from a pristine copy before every abuse; after each one I
read the autosaved project JSON and grepped the server log. Everything in
§"Break-it catalogue" of `docs/RELEASE_0.4_PLAN.md` was tried except the items
in section 3, plus about twenty abuses of my own.

## 1. Summary

The core modelling path is robust: bad expressions, injection attempts, empty or
one-row files, zero/negative exposure, NaN targets, wrong families, 3,000-level
categoricals, odd variable names in Excel, double clicks, back/forward, a narrow
phone screen and a server restart all end in a clear message or a graceful
fallback, and the server log never shows a traceback for those. The problems
cluster in three places: (a) the **Variables and Split pages call the data
pipeline without a safety net**, so any data step that fails (renamed column,
bad derived column, new file with different columns, text indicator column)
shows a raw traceback there — while every other page shows a proper message;
(b) **project file handling** — "New empty project" silently overwrites the
project you had open, two tabs overwrite each other, and opening or saving a
bad file/path gives tracebacks; (c) **silent re-pointing**: when a column the
model uses disappears (renamed target, ignored predictor) or a split column
name collides with a data column, the model quietly changes and still reports
"Fitted and up to date" with wrong numbers.

Counts: **data loss 4 · crash 9 · misleading 13 · cosmetic 12** (38 findings).
The server log contains 30 tracebacks, all matching findings below. The
data-loss and crash findings block the merge.

## 2. Findings (sorted by severity)

Severity: **data loss** = project/fit lost or corrupted, or wrong numbers shown
silently · **crash** = raw traceback in the browser · **misleading** = wrong or
confusing output without an error · **cosmetic**.

| # | What I did (exact steps) | What happened | Severity | Suggested fix |
|---|---|---|---|---|
| 1 | Open the breaker project (fitted). Project & data → click **New empty project** → type any name into "Project name" and press Enter (any later edit does the same). | The sidebar still shows the *old* project path under "untitled"; the empty project is autosaved **over the original file**: roles, recode, filter, model and champion are gone from `breaker.easyglm-project.json`. Screenshot: sidebar shows the old path, header "untitled". | **data loss** | `set_project(..., None)` must clear `project_path` (and the "Project file" field) so a new project never autosaves onto the previous file. |
| 2 | Open the same project in two browser tabs. Tab A: Model → Notes = "note from tab A". Tab B: Variables → Row filters → add `pl.col('DrivAge') > 20`. Tab A: change Notes again. Tab B: Model → Delete; Tab A: Fit. | Each autosave writes that tab's whole in-memory project: after B's filter, A's note is gone from the file; after A's second edit, B's filter is gone; after B deletes the model A still shows it fitted and its next edit resurrects it. No warning anywhere. | **data loss** | Before autosaving, compare the file's mtime/hash with what this session last wrote; if it changed, refuse and tell the user to reload (or a per-file session lock). |
| 3 | Fit `freq_v1`. Variables → in the roles grid type `claims` into the "rename to" cell of ClaimNb. Go to Model. | The Target box silently jumps to the first column (**IDpol**, the policy number) and `models.freq_v1.target` is autosaved as `IDpol`. Clicking Fit *succeeds*: "Fitted and up to date", holdout Gini 0.393, dev. explained 11.8 % — a Poisson model of policy numbers. | **data loss** (wrong numbers silently) | If `cfg.target/weight/offset` is not in the prepared columns, show an error and disable Fit instead of `index=0`; propagate renames into every model config. |
| 4 | Split page (random split) → "Split column name" = `ClaimNb` (also tried `Exposure`). Fit. | The random 0/1 flag **replaces the target column** (or the weight). With `ClaimNb`: fit succeeds, train A/E 1.000, holdout A/E 0.000, Gini "—", still "Fitted and up to date". With `Exposure`: Split page shows exposure totals of 0–30k policies, fit fails with the cryptic "Weights sum to zero". | **data loss** (wrong numbers silently) | Refuse a split column name that already exists in the data (or that has any role). |
| 5 | Project & data → upload `weird_names.parquet` (columns `policy id`, `claim.count`, `expösure`, `1st_age`, `véh âge`, `région/zone`, `from`, `Other`, `a'b"c`) → Load data → Variables (or Split). Same with any new file whose columns differ from the project's. | Raw traceback `ColumnNotFoundError: unable to find column "Exposure"` on **Variables** and **Split** (the filter references a column that no longer exists). Design/Model/Rate tables show the proper "The data steps on the Variables page fail: …" message. | **crash** | Wrap `prepared_frame()`/`apply_variables()` on the Variables page (recode preview, bottom caption) and the Split page in the same guard as `ui.require_data()`. |
| 6 | Variables → roles grid → rename `VehAge` to `DrivAge` (a name that already exists). | Traceback `DuplicateError: column 'DrivAge' is duplicate` on Variables and Split; the rename is saved and VehAge's role is dropped from the project. | **crash** | Validate renames against existing/renamed names and show an error; do not save the colliding rename. |
| 7 | After #6 (or any rename), select the "rename to" cell and press Delete to clear it. | Traceback `AttributeError: 'float' object has no attribute 'strip'` (an emptied text cell comes back as NaN). The rename stays in the file and the role stays lost; the only way out is to retype the original name. | **crash** | Treat NaN/None in text cells as empty (`str(x) if x == x else ""`); restore the role when a rename is undone (roles are keyed by the *new* name, see also #23). |
| 8 | Variables → Derived columns → name `foo`, expression `pl.col('foo') + 1` → Add derived column. (Preview correctly says "unable to find column foo", but Add only checks syntax.) | Traceback `ColumnNotFoundError` on Variables and Split; other pages show the guarded message. The ✕ button still works, so it is recoverable. | **crash** | Evaluate the expression on the sample before adding (as Preview does) and refuse; plus the guard of #5. |
| 9 | Variables → Derived columns → name `bad`, expression `pl.col('Region') / 2` → Add. | Traceback `InvalidOperationError: division with 'String' datatypes is not allowed` on Variables and Split. | **crash** | Same as #8. |
| 10 | Split page → click **Existing indicator column** (nothing else). Also: pick `Area` as indicator while "Value meaning TRAIN" is still `1`. | Traceback `ComputeError: cannot compare string with numeric type` because the first column (`IDpol`, text) is silently chosen as indicator and compared with `1`. Side effect saved to the file: IDpol's role changes id → split, then → **ignore** when you pick another column. | **crash** | Compare as strings (cast both sides) or catch and show "value 1 does not match a text column"; do not auto-assign the first column / never demote other roles when the indicator changes. |
| 11 | Project & data → "drop a project JSON here": upload (a) a parquet file renamed `.json`, (b) a truncated JSON, (c) `{"version": 99}`, (d) `[1,2,3]`, (e) a JSON with `roles` as a string → Load uploaded project. Also type the path of the truncated file in "Project file" → Open project file. | Tracebacks: `UnicodeDecodeError`, `JSONDecodeError`, `ValueError: Project version 99 is newer…` (a good message, but shown as a traceback), `TypeError: object is not iterable`, `ValueError: dictionary update sequence…`. The open project is not lost. | **crash** | Wrap `_open_project` in try/except and show "Not a valid easy_glm project: …". |
| 12 | Project & data → "Project file" = a path in a read-only folder, or `/nonexistent_dir/x.json`, or `/` → **Save project** (main page button). | Tracebacks `PermissionError`, `FileNotFoundError`, `FileExistsError`. (Autosave itself catches OSError — see #27.) | **crash** | Catch OSError on both Save buttons and show the message. |
| 13 | Model → New model name `a/b` (Enter) → Create → Fit → Rate tables (or Export) with that model selected. | Traceback `xlsxwriter FileCreateError … /tmp/…/a/b.xlsx` — the Excel bytes are built on page load, so the whole Rate tables page and Export page are unusable for this model. | **crash** | Sanitise the temp file name (or write to BytesIO); reject `/` and `\` in model names. |
| 14 | Model → Weight = `IDpol` (text policy number) → Fit. | "Fitted and up to date", 0 / 95 non-zero coefficients, holdout Gini 0.000, holdout A/E 0.910 — the policy numbers were used as weights, no warning. | misleading | Only offer numeric columns as weight/offset/target (or fail with "weight must be numeric"). |
| 15 | Variables → roles grid → set the role of `VehAge` (a predictor of `freq_v1`) to `ignore`. Go to Model. | `VehAge` silently disappears from the model's predictor list (autosaved), the fit is invalidated, nothing tells the user the model changed. | misleading | Show "VehAge was removed from freq_v1" (flash) or keep it and show the validation error. |
| 16 | Project & data → File path = `/Volumes/nowhere/huge_2GB_file.parquet` (or `/tmp`, or a zero-byte parquet) → Load data. | The Project page shows **nothing**: no error, the preview just vanishes, the status bar keeps "✓ Data ✓ Prepared", the sidebar keeps "✅ data loaded". The message "Could not load …" only appears on other pages — and for a missing file it reads "Could not load X: X." (the path twice). | misleading | Show `load_error` on the Project page and turn the Data chip off; say "file not found" for FileNotFoundError. |
| 17 | Upload any data file with "…or upload a data file" → Load data. Read the project JSON. | `data.source.path` now points to `/var/folders/…/T/tmpXXXX/<file>` — a temporary folder. After a reboot or temp clean-up the saved project cannot find its data. No hint in the UI. | misleading | Copy uploads next to the project file (or into a `<project>.easyglm-data/` folder) and say so. |
| 18 | Variables → Level recodes → column VehGas → type `Petrol` into "map to", then clear the cell (Delete) → Apply recode. Also: type spaces only. | Recode saved as `Regular → "nan"`: a new level literally called **nan** appears in the data and the rate tables. | misleading | Treat NaN/None/whitespace in "map to" as "no mapping". |
| 19 | Design → Defaults → Min level share = `0.5` → Fit. Same message for a 3,000-level categorical (`zip3000`), or predictors = `IDpol` only. | Design page shows no warning; Fit fails with "Cannot derive levels for 'Region' (**all null on train**)". Region is not null — every level is simply below the share threshold. | misleading | Distinguish "no level reaches min_level_share (x %)" from "all null"; warn on the Design page. |
| 20 | Design → Density → Kind linear → untick "Clamp to the training range" → Clamp lo `1e300`, hi `1e308` → Apply → Fit. | Accepted; fit succeeds; Density becomes a flat line (every value is below the clamp) with no warning. | misleading | Warn (or refuse) when the clamp range does not overlap the training range. |
| 21 | Rate tables → VehGas → set the working relativity of Diesel to `0`. | Accepted silently (min_value is 0): Diesel is priced at zero. The Model page metrics then read train A/E 2.054, holdout Gini 0.035 under the label "Fitted and up to date", with no hint that manual adjustments are included. | misleading | Require > 0 for every table (as linear tables already do); label Model-page metrics "with N manual adjustments". |
| 22 | Variables → Level recodes → Area → Unmapped levels "→ Other" → Apply; add Area to the model → Fit. Same when a level is explicitly mapped to `Other`. | Fit refuses: "CategoricalEncoder('Area'): other_label 'Other' clashes with a level". The tool's own recode option produces a state the tool cannot fit. | misleading | Either merge a real "Other" level into the lumped bucket or rename the lumped bucket when the level exists; do not offer a default that fails. |
| 23 | Variables → rename `VehAge` to something valid, e.g. `véh âge (yrs)` — then rename it back to `VehAge`. | Works, but roles/types/designs are re-keyed on every rename; combined with #6/#7 a failed rename loses the role. (Unicode/space names otherwise fit and export fine.) | misleading | Key roles by the raw column name and apply renames on output. |
| 24 | Design → DrivAge → Knots `nan` (also `inf, 40`, `1e400, 30`, `-1e309, 30`) → Apply knots. Also `30, 40, 999999`. | Accepted and written to the project file as `NaN` / `Infinity` (not valid JSON — other tools cannot read the file). A message "knots must be finite" appears and blocks the fit; the knot above the data max is accepted silently (empty top band). | misleading | Reject non-finite tokens in `_parse_numbers`; warn for knots outside the training range. |
| 25 | Project & data → Project name = `RENAMED BY USER` → open a different project file (Open project file). | The "Project name" field keeps showing `RENAMED BY USER` for the newly opened project; in one run (after the next interaction on the page) that stale name was written into the newly opened project's file. | misleading | Give the name/path text inputs a key that changes with the project (or set the value via session_state on open). |
| 26 | Model → Base rate override = `1e12` (a fitted model). | Accepted; Model page shows train A/E 0.000 and "holdout dev. explained -919206693535870.2 %"; Rate tables base rate 1,000,000,000,000. | misleading | Bound the override to a plausible range relative to the fitted base rate; format huge percentages as "—". |
| 27 | Make the project file read-only (chmod 444) → Model page → change Notes → go to Variables. | Nothing on Model/Variables says the save failed; edits are silently not persisted. "Autosave failed: Permission denied" only appears when you visit the Project page (which is also the only page that shows the `errors` list). | misleading | Show `errors` on every page (in `status_bar`) and a red sidebar note while autosave is failing. |
| 28 | Model → Fit with cross-validated penalty (10 folds, 100 alphas) → press F5 after ~3 s. | No traceback; page comes back "Not fitted yet." and no run is persisted — the fit in progress is discarded without a message. | cosmetic | Show "a fit was interrupted by the reload" or keep the fit running and pick it up. |
| 29 | Model → Target = `Exposure` and Weight = `Exposure` → Fit. Also alpha (fixed) = `0` → Fit. | "Fit failed: No variation in y. Coefficients can't be estimated." / "Fit failed: A singular matrix detected: slice(s) [0] are singular." — safe but meaningless to an actuary. | cosmetic | Validate target ≠ weight; translate solver errors ("alpha = 0 gives an unpenalised fit that cannot be solved; use a small alpha"). |
| 30 | Model → New model name `" "` (a space), `freq_v1 ` (trailing space), `..`, `CON`, 300 × `x` → Create. | All accepted; the sidebar shows "Models: freq_v1, a/b, …, freq_v1 , .., CON" and a 300-character name; the blank model is indistinguishable in the picker. | cosmetic | Strip names, require a non-empty printable name, cap length. |
| 31 | Model → Weight = `(none)`. | The "Divide target by weight" checkbox is disabled but still shown **ticked** while the project stores `false`; re-adding a weight turns it back on. | cosmetic | Uncheck the box when it is disabled. |
| 32 | Split page → Seed: type `-5`, `99999999999`, `1e9`. | The field displays `-5` / `99999999999` while the project keeps the old seed (and `1e9` shows as `1`). | cosmetic | Streamlit widget behaviour; add a caption "0–10000". |
| 33 | Edit the project file to `split.fraction = 1.0` → Open project file → Split page. | Slider shows 1.00 outside its own 0.50–0.95 range; Model page correctly disables Fit with "split.fraction must be in (0, 1)". | cosmetic | Clamp the slider value and warn. |
| 34 | Split page → "Split column name" = empty string. | Accepted; a column named "" is created; Model says "Not fitted yet", no hint. | cosmetic | Require a non-empty name. |
| 35 | Diagnostics → A/E by pair → Rows = Columns = VehGas. | The error "Pick two different variables." is rendered inside every other tab too (Lift, Residual factors…). | cosmetic | Render the error inside its tab only. |
| 36 | Model → Delete the only model; look in `breaker.easyglm-runs/`. | The deleted model's `.pkl`/`.json` stay on disk until another model is persisted. | cosmetic | Remove the run files on delete. |
| 37 | Design → grid → monotone = `increasing` on `Region` (categorical). | Accepted in the grid and saved; no message on the Design page; Fit fails later with a correct message. | cosmetic | Validate in the grid (`p.validate()` already knows). |
| 38 | Model → add `constant` (a numeric column with one value) as predictor → Fit. | "Fit failed: Cannot derive knots for 'constant'" — the whole fit is blocked by one useless column instead of dropping it. | cosmetic | Warn on the Design page and skip constant columns. |

Things that behaved well (for the record): empty / one-row / 30-row files,
mixed-type CSV, case-duplicate columns, 3,000-level categorical (12 s fit,
tables and Excel fine once min share is 0), all-null and single-level columns,
NaN/±inf target ("Target contains NaN or infinite values"), zero/negative
exposure ("Weights must be finite and strictly positive"), a filter that drops
every row / keeps one row, injection attempts in expressions (`__import__`,
`open`), non-expression filters, wrong families (binomial on counts, gamma on
zeros, Tweedie on negatives), zero predictors (Fit disabled), CV on 30 rows,
double-click Fit / Add interaction, delete during fit, reset-while-editing a
relativity, relativity edits to text/negative (browser blocks; "empty; change
not saved"), editing the null row, variables named `from`, `a/b`, `Sheet[1]`
and 44 characters in Excel (sheet names sanitised), unicode model names in all
downloads, no target assigned, browser back/forward, 375 px viewport (no
horizontal overflow), SIGHUP/restart of the server (fit restored from
`breaker.easyglm-runs/`).

## 3. Catalogue items not exercised, and why

* **"Open the full relativity editor in a new tab"** — starts a second server
  on port 8502; not launched because another agent may be using ports.
* **100 % / 0 % training fraction via the GUI** — the slider is bounded
  0.50–0.95 (a disabled control, which is the right answer); tested only by
  editing the file (#33).
* **Pasting text into number fields / alpha = 1e9 / relativity = text or
  −1** — Chrome's number inputs drop letters and the widgets' min/max reject
  out-of-range values before Streamlit sees them; only the in-range abuses
  above were possible. `1e12` in a relativity cell arrives as `112`.
* **A 2 GB file** — only the "path does not exist" variant (#16); no large
  file was created.
* **sas7bdat / xlsx nasty files** — not created; the loader path is shared
  with parquet/csv.
* **"Close the terminal that launched the server"** — the server ran under
  `nohup`, so it survived SIGHUP; I killed it with SIGTERM and restarted it
  instead (fit restored).
* **Model named `""`** — the Create button is disabled for an empty name
  (correct); the blank-space variant is #30.
* **Recode every level to the same value** — done for VehGas (both levels →
  `X`): fit succeeds with a single-level variable; only the "nan" mapping bug
  (#18) came out of it.

## 4. Misleading output explained for the actuary

* **#3 — renamed target.** After you rename the claims column, the model's
  "Target" box silently falls back to the *first* column of the data, the
  policy number. The fit "succeeds" and even shows a healthy Gini (0.39),
  because policy numbers are correlated with vehicle age, region, etc. The
  relativities you would export are relativities of *policy numbers*, not
  claim frequency. Nothing on screen says the target changed.
* **#4 — split column named after a data column.** The random split creates a
  new 0/1 column with the name you typed. If that name is already a column,
  the original is overwritten. Named `ClaimNb`, your "claims" become 1 for
  every training row and 0 for every holdout row: the fit reports train A/E
  exactly 1.000 and holdout A/E 0.000 — a model of the split itself. Named
  `Exposure`, every weight becomes 0 or 1.
* **#14 — text id as weight.** Policy numbers cast to numbers become the
  weights, so a policy numbered 2,000,000 counts as 2 million years of
  exposure. The lasso then keeps 0 of 95 coefficients (Gini 0.000). The page
  still says "Fitted and up to date".
* **#15 — ignoring a column used by the model.** The model silently loses that
  predictor; the next fit is a different model with no notice.
* **#18 — the "nan" level.** Clearing a "map to" cell maps that level to the
  text "nan". You get a rating level called nan in the tables and the Excel.
* **#19 — "all null on train".** With min level share 50 % (or a 3,000-level
  postcode, or an id used as a predictor) no level reaches the threshold, so
  every level is lumped into Other and the message says the column is all
  null. It is not; the threshold is the cause.
* **#20 — clamp outside the data.** A linear term whose clamp range does not
  overlap the data gets one flat band; the fit reports success but that
  variable contributes nothing.
* **#21 — zero relativity.** A relativity of 0 means a premium of 0 for that
  level; the tool accepts it. The Model page then shows A/E and Gini computed
  *with* the adjustment under the label "Fitted and up to date", so it looks
  as if the fit itself got worse.
* **#26 — base rate override 1e12.** The override multiplies every prediction;
  metrics become nonsense ("dev. explained −919,206,693,535,870 %") without any
  warning.
* **#17 / #27 — where your work is saved.** An uploaded data file is stored in
  the computer's temporary folder and the project remembers that path, so the
  project may not open after a restart. And if the project file cannot be
  written (read-only, network drive offline), only the Project & data page
  tells you; on every other page you keep working and nothing is being saved.
* **#1 / #2 — overwritten projects.** "New empty project" keeps the old file
  path, so your first edit saves an empty project over the one you had open.
  Two browser tabs on the same project each save their own copy: whichever
  tab you touched last wins, and the other tab's changes vanish from the file
  without any message.

## Appendix — nasty files used

All built from `tests/fixtures/french_motor_50k.parquet` (+ synthetic
`current_premium`): `empty_rows.parquet` (schema, 0 rows), `zero_bytes.parquet`
and `.csv` (0 bytes), `one_row.parquet`, `thirty_rows.parquet`,
`mixed_types.csv` (a `Mixed` column with `12`, `abc`, `3.5`, blank, `1e3`,
`2020-01-01`), `weird_names.parquet` (columns `policy id`, `claim.count`,
`expösure`, `1st_age`, `véh âge`, `région/zone`, `from`, `Other`, `a'b"c`),
`case_dup.parquet/.csv` (adds `area` and `exposure` next to `Area`/`Exposure`),
`high_card.parquet` (adds `zip3000` with 3,000 levels, `all_null`, `constant`
= 1, `const_str` = "X"), `neg_exposure.parquet` (500 rows Exposure = 0, 500
rows −0.3), `bad_target.parquet` (100 NaN, 50 +inf, 50 −inf in ClaimNb),
`binary_neg.parquet` (adds `has_claim` 0/1 and `neg_target` = ClaimNb − 1),
`project_as_data.json` (the project file — rejected by the data dropzone,
correct), `data_as_project.json` (the parquet renamed), `corrupt.json`,
`v99.json`, `list.json`, `badtypes.json`.
