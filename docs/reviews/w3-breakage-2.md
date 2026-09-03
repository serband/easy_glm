# W3 breakage report (second session) — easy_glm Workbench (release-0.4 @ e3d7eca)

Second breaker session, after the hardening piece W3 and its five should-fix
follow-ups. Live server from the repo venv (Streamlit 1.57) on port **8651**,
driven with Playwright/Chromium at 1500 × 1000 (plus a 50 % and a 200 % zoom
pass), project restored from a pristine *fitted* copy before every abuse; after
each block I read the autosaved project JSON, listed the persisted-run folder
and grepped the server log.

Starting project "breaker2" — the same shape as the first session: French-motor
50k fixture plus a synthetic `current_premium`, roles as in the e2e template
(ClaimNb target, Exposure weight, IDpol id, 9 predictors), recode Area E/F→D,
filter `Exposure > 0.02`, random split 70 % seed 7, one fitted model `freq_v1`
(Poisson, claims / exposure, lasso, fixed alpha 0.001, 7 predictors).

Two things were done: **(a)** every one of the 38 findings of
`docs/reviews/w2-breakage.md` was replayed by its recorded steps, and **(b)**
about thirty abuses the first session did not try, concentrated on the newest
code (the interaction and linear-term editors), the persisted-run folder, the
multi-tab conflict dialog, uploads, number widgets, keyboard-only use and
browser zoom.

## 1. Summary

**New findings: data loss 2 · crash 0 · misleading 6 · cosmetic 6 (14).**

**The 38 old findings: 34 fixed, 4 still broken** (28, 31, 32, 38 — all
cosmetic, all unchanged from the first session; the W3 review had already
looked at 28, 32 and 38 and left them open deliberately, 31 is simply
untouched). Four of the 34 are fixed with a caveat worth reading: 10, 24, 30
and 33 (see §3).

There was **not a single raw traceback** in this session: `stException` never
appeared on any of the nine pages during ~40 abuse blocks, and the server log
contains **0** occurrences of "Traceback". The project file was never lost or
silently emptied. The error-boundary and project-file work of W3 holds up.

Both data-loss findings are in the same place, and it is a place W3 did not
touch: **the persisted-run folder is shared mutable state, and every session
prunes it using its own copy of the project.** The conflict dialog protects
`*.easyglm-project.json` and nothing else, so a second browser tab — the exact
situation the dialog was built for — can delete the fit that belongs to the
project on disk, silently. Neither is a W3 regression: `persist_run`'s "latest
run per model" cleanup and `remove_model_runs` are byte-identical at `88fd757`.
They block the merge under the rule in `docs/RELEASE_0.4_PLAN.md` anyway,
because the outcome is lost work with no message.

The misleading findings are led by one that is easy to hit and easy to fix:
**"Create" does not switch to the new model.** The page confirms "Model
'freq_v2' created" and then keeps showing, editing and fitting `freq_v1`. I
have deliberately not rated it *data loss*, because the picker on screen does
say `freq_v1` throughout — nothing is hidden, the user is simply led to the
wrong model — but it is the one misleading finding I would fix before the
merge, since the model it quietly edits is usually the champion.

## 2. New findings (sorted by severity)

Severity as in the plan: **data loss** = project/fit lost or corrupted, or
wrong numbers shown silently · **crash** = raw traceback in the browser ·
**misleading** = wrong or confusing output without an error · **cosmetic**.

| # | What I did (exact steps) | What happened | Severity | Suggested fix |
|---|---|---|---|---|
| 1 | Open the fitted project in two tabs, both on **Model**. Tab A: Notes = "A wins" (autosaved). Tab B: Notes = "B loses" → B gets the conflict notice ("changed by another browser tab", autosave paused). In B, **without resolving the conflict**, click **Fit model**, then click **Delete**. | B's **Fit** rewrites the run file even though B's autosave is paused, and B's **Delete** then removes the model's `.pkl` **and** `.json` from disk. The project file still contains `freq_v1` (B's delete never reaches it — the conflict notice is still up), so the project on disk now has a model whose persisted fit has been deleted by a tab that was not allowed to write. Tab A and every new session show "Not fitted yet." for a model that was fitted a minute ago. No message anywhere. | **data loss** | While `st.session_state.conflict` is set, refuse `persist_run` and `remove_model_runs` the way `touch()` already refuses to write the project file. |
| 2 | Two tabs on the fitted project, both on **Model**. Tab A: alpha 0.001 → 0.002, **Fit** (a new run file appears, project on disk now says alpha 0.002). Tab B (still showing alpha 0.001, one edit behind): click **Fit**. Open a third tab. | B's fit persists its own key and `persist_run`'s "latest run per model" cleanup **deletes A's file** — the only run that matches the project on disk. The third tab (and tomorrow's session) says "Not fitted yet." for the saved project, with no hint that a fit existed. Clicking Fit in both tabs at the same instant reproduced a worse variant once: both cleanups ran after both writes and the runs folder ended **empty** while both tabs said "Fitted and up to date". | **data loss** | Delete a stale run only when its key belongs to *this* session's own previous save (or keep the newest N per model), and never delete a file written after the one just saved. |
| 3 | Model page → "New model name" = `freq_v2` → **Create**. Then type Notes = "this note is meant for freq_v2" and click **Fit model**. | The flash says *"Model 'freq_v2' created"* — and the Model picker, the whole configuration panel and the Fit button stay on **freq_v1**. The note is saved to `models.freq_v1.notes`, `freq_v2.notes` stays empty, and Fit fits and persists **freq_v1**. Selecting `freq_v2` by hand works; after a browser reload the picker is back on `freq_v1`. Anything the user changes next (predictors, family, penalty) is silently applied to the champion instead of the new model, invalidating its fit. | **misleading** | `_model_picker` must seed the selectbox from `model_current` (set `st.session_state[S.widget_key("model_select")] = new_name` before the rerun), so Create selects what it created. |
| 4 | `chmod 555` the `breaker2.easyglm-runs/` folder → **Fit model** (the correct message *"Could not persist the fit: [Errno 13] Permission denied …"* appears) → `chmod 755` → **Fit model** again (the run file is written) → visit any page. | The banner never goes away. `persist_run` appends to `session_state.errors` and only the *autosave* path prunes that list (the S1 fix), so one transient failure tells the user their fits are not being saved on **every page for the rest of the session** while they are being saved. | misleading | Prune the "Could not persist" entries in `persist_run` on a successful save, exactly as `_clear_autosave_errors()` does for autosave. |
| 5 | Model page → click into the **alpha** box, select all, paste `1e9`, press Enter. | The box shows **190.00100** and keeps showing it — including after a fit — while the project holds `alpha: 0.001` and the metric row says `alpha 0.00100`. No message. Same class as old finding 32 (the Seed box, still open), but on the penalty: the page names a penalty the fit did not use. | misleading | Reject / clamp out-of-range values in `st.number_input` (or show "alpha must be 0–10; still using 0.001") instead of leaving the typed text on screen. |
| 6 | Variables → Derived columns → `prem_raw` = `pl.col('current_premium')` → Model → Offset = `prem_raw` (a premium of ~250, **not** a log) → **Fit model**. | "Fitted and up to date", train A/E 1.000, holdout A/E 0.583, holdout Gini 0.068, **holdout dev. explained −49.3 %**. Nothing says the offset is exponentiated, so a premium of 250 means `exp(250)`; the only clue is the label "(linear scale)". The first session's finding 14 (a text id as weight) is the same trap, and that one is now blocked. | misleading | Warn when the offset column's training range is implausible on the log scale (e.g. `|offset| > 20`): "offsets are on the linear predictor scale — did you mean log(current_premium)?" |
| 7 | Model page on a fitted model → change **Family** from `poisson` to `gamma` (do not fit). | The page says, correctly, *"Spec changed since the last fit — results below are from the previous fit. Refit to update."* — but the status chips above it still read **✓ Fitted** and the sidebar still reads **✅ fitted**. Two of the three fitted indicators contradict the third, on the page whose whole point is "is this model current?". | misleading | `ui.status_bar()` runs before `_config()` applies the change, so the chips are one render stale: call `st.rerun()` after a spec change (as the other edits do), or draw the chips after the configuration block. |
| 8 | Copy the project's data file away and back (`cp policies.parquet /tmp/ && cp /tmp/policies.parquet .`) — identical bytes, new modification time — then open the project. | Every model shows "Not fitted yet."; the persisted `.pkl` is still there but is never loaded, because the run key is `(path, size, mtime_ns)`. Restoring from a backup, a `rsync` without `-t`, a network copy or an antivirus touch throws away every fit with no explanation. (This is how I found it: `cp -R` without `-p` while building the fixture.) | misleading | Key the run on the file's size plus a cheap content hash (or say "the data file changed since this fit" instead of "Not fitted yet"). |
| 9 | Delete `breaker2.easyglm-runs/*.pkl` by hand while the app runs, then open the Model page. | Correct: "Not fitted yet.", no traceback. But the orphaned `…json` sidecar stays in the folder for ever — nothing ever removes a sidecar whose pickle is gone. | cosmetic | In `load_persisted_run`, when `target` does not exist, remove a stray `target.with_suffix(".json")`. |
| 10 | Design → add the interaction `DrivAge × VehGas` → Model → remove **VehGas** from the Predictors multiselect. | Fit is disabled with a good message (*"freq_v1: interaction parent 'VehGas' is not one of the model's predictors"*), but the caption above it reads *"⚠ parents no longer among the predictors: **DrivAge×VehGas**"* — it names the interaction, not the parent that went missing. And where the same thing done from the roles grid drops the interaction with a notice ("Interaction(s) VehGas×Region removed from model freq_v1"), this path leaves the model unfittable until the user finds the Design page. | cosmetic | Name `bad` parents (`it.a`/`it.b`), not `it.name`; offer "remove the interaction" on the Model page. |
| 11 | Project & data → "Project file" = `…/typo_dir/deep/x.json` (neither folder exists) → **Save project**. | "Saved …" — `save_project` calls `mkdir(parents=True)`, so a typo silently creates a folder tree and moves the project into it. (Already noted as nit 3 of the W3 review; confirmed in the browser.) | cosmetic | Create only the file; if the parent folder does not exist, say so. |
| 12 | Hand-edit the project file to `"fraction": 1.0` → open it → Split page. | The slider correctly shows 0.95 (old finding 33 is fixed), but the value is clamped **and autosaved back into the file** — 1.0 is gone from the project — and the warning that explains it ("Training fraction 1 … is outside the 0.50–0.95 range") is drawn before the rerun that follows the change, so the user never sees it. | cosmetic | Flash the warning (`ui.flash`) instead of `st.warning` so it survives the rerun. |
| 13 | Model page → "New model name" = `CON` (also `NUL`, `PRN`) → **Create**. | Accepted; the model is created and its downloads are `…_CON_rate_tables.xlsx`, which cannot be written on Windows. (Nit 4 of the W3 review, confirmed live.) Emoji and accented names (`modèle_ÉTÉ_🚗`) are accepted and export fine; `"  padded  "` is trimmed; `..`, a duplicate and >60 characters are refused with a reason. | cosmetic | Add the Windows device names to `_MODEL_NAME_BAD`. |
| 14 | Project & data → Exploration sample = `1` → Explore. Separately: load a parquet with columns but **0 rows**. | Explore says *"Charts use the exploration sample of 1 rows"* (grammar), and with a 0-row file the status chips say **✓ Data ✓ Roles ✓ Prepared** for a frame with no rows; the Design page and the fit are honest ("There are no training rows"; "Fit failed: No training rows after the split"). | cosmetic | Pluralise; turn the "Prepared" chip off when the prepared frame is empty. |

## 3. The 38 first-session findings, replayed

Status: **fixed** · **still broken** · *fixed (changed behaviour)* where the
outcome is now different but not what the finding described.

| # | First session | Status | What I saw now |
|---|---|---|---|
| 1 | "New empty project" overwrote the open file | fixed | First click warns; the second starts an empty project; typing a new name leaves `breaker2.easyglm-project.json` byte-for-byte unchanged. |
| 2 | Two tabs overwrite each other | fixed | B's edit is refused with the conflict notice, A's note stays on disk, A keeps saving, B's Delete never reaches the file. Both tabs in conflict at once (a third writer touches the file) and then Reload in A + Overwrite in B clicked together: no exception, disk = B's version, A shows B's version, both keep working. **But the runs folder is not covered — new findings 1 and 2.** |
| 3 | Renamed target silently re-pointed to the id column | fixed | Renaming `ClaimNb`→`claims` in the grid moves the role *and* `models.freq_v1.target`; nothing points at IDpol; no "Fitted and up to date" on a stale spec. |
| 4 | Random split named after a data column | fixed | *"'ClaimNb' is already a column in the data; the random split would overwrite it."*; the project keeps `traintest`. Same for `Exposure`. |
| 5 | New file with different columns → traceback on Variables and Split | fixed | `weird_names.parquet`: all nine pages, **0** exceptions; Variables lists the 13 orphaned roles by name; every page shows *"The data steps fail: unable to find column "Exposure"; valid columns: […]"*. |
| 6 | Rename onto an existing name | fixed | *"Cannot rename 'VehAge' to 'DrivAge': another column already has that name. Rename not saved."*; `renames` untouched, role kept. |
| 7 | Clearing a "rename to" cell → `float has no strip` | fixed | The cell empties, the rename disappears, `VehAge`'s role and the model's predictor list come back. No traceback. |
| 8 / 9 | Derived column that cannot run | fixed | *"pl.col('foo') + 1 fails: unable to find column "foo"…"*; nothing is written to the project; `pl.col('Region') / 2` likewise. |
| 10 | "Existing indicator column" auto-picked the id column | fixed (caveat) | Nothing is auto-picked (*"choose the train/holdout indicator column"*), `IDpol` keeps role `id`, and a text column against TRAIN `1` gives *"No row of 'Region' equals the TRAIN value 1"*. Caveat: picking `Region` as the indicator takes the `split` role from a column that is a model predictor without saying so (the Model page then reports the missing predictor). |
| 11 | Five kinds of broken project file | fixed | Parquet-as-JSON, truncated JSON, `{"version": 99}`, `[1,2,3]`, `roles` as a string, and a bad path typed into "Project file": one sentence each, open project untouched. |
| 12 | Save / autosave to a bad path | fixed | `/nonexistent_dir/x.json` → *"Could not save the project to …"*; `/` and a folder → *"… is a folder; give the project file a name"*; a read-only folder → the same shape of message. (New cosmetic 11: a typo path creates folders.) |
| 13 | Model named `a/b` killed Rate tables and Export | fixed | Create is disabled, *"⚠ Model name cannot contain '/'"*, nothing written. |
| 14 | Text id as weight | fixed | The Weight box offers numeric columns only: `(none), ClaimNb, Exposure, DrivAge, VehAge, BonusMalus, Density, VehPower, current_premium, traintest` — `IDpol` is not offered. |
| 15 | Ignoring a predictor changed the model silently | fixed | Flash: *"VehAge was removed from model freq_v1: its role is now ignore"*, and interactions on that column are dropped with their own notice. |
| 16 | Bad data path showed nothing on the Project page | fixed | *"Could not load /Volumes/nowhere/huge_2GB_file.parquet: the file does not exist. Check the path on the Project & data page."* on the Project page, and the Data chip goes out. A folder and a zero-byte parquet give their own reasons. |
| 17 | Uploaded data landed in a temp folder | fixed | Uploads are stored in `breaker2.easyglm-data/` next to the project and the caption says "(next to the project file)". |
| 18 | Cleared "map to" created a level called `nan` | fixed | An emptied cell means "no mapping"; the recode is saved with 0 mapped levels and no `nan` appears in the project or the tables. |
| 19 | "all null on train" for a min-level-share problem | fixed | *"No level of 'Region' reaches the minimum level share (50.00% of training rows; 22 distinct values). Lower the share on the Design page…"* |
| 20 | Clamp outside the training range accepted | fixed | *"Clamp range 1e+300 – 1e+308 does not overlap the training range 2 – 27000; the term would be flat everywhere"*; nothing saved. `lo == hi` gives "Clamp lo must be below clamp hi". |
| 21 | Relativity 0 priced a level at zero | fixed | 0 and −1 are refused by the editor (min 1e-4) and no adjustment is saved; a legitimate edit is labelled *"metrics include 1 manual adjustment(s)"* on the Model page. |
| 22 | The tool's own "→ Other" recode could not be fitted | fixed | The fit succeeds and the lumped row is now labelled **"Other (lumped)"** next to the real level `Other` (the W3 review's S5). |
| 23 | Roles re-keyed on every rename | fixed | `VehAge` → `véh âge (yrs)` → `VehAge`: role, type and the model's predictor list survive the round trip. |
| 24 | Non-finite knots written into the project file | fixed (caveat) | `nan`, `inf, 40`, `1e400, 30`, `-1e309, 30` are all refused ("'nan' is not a finite number") and the file stays valid JSON. Caveat: `30, 40, 999999` (a knot above the data maximum) is still accepted silently, which the first session also asked for. |
| 25 | Stale project name after opening another project | fixed | The name box, the sidebar and the file all show `second_project` immediately and after navigating away and back. |
| 26 | Base rate override 1e12 | fixed | *"Override is 27,300,639,269,600× the fitted base rate (0.0366292); every prediction is scaled by that much."*, and "holdout dev. explained" prints "—". |
| 27 | Autosave failure only visible on the Project page | fixed | `chmod 444` → the red banner on Model **and** Variables and a sidebar note; `chmod 644` + one edit → the file is written again **and the banner clears** (the W3 review's S1). |
| 28 | A fit interrupted by F5 is discarded silently | **still broken** | CV fit (10 folds, 100 alphas), F5 after 3 s: no traceback, the page comes back "Not fitted yet.", nothing says a fit was interrupted. Unchanged. |
| 29 | Target = weight / alpha = 0 gave solver noise | fixed | Target = weight = `Exposure` now **disables Fit** with a validation message instead of failing in the solver. (alpha = 0 could not be entered at all: the box keeps 0.001.) |
| 30 | Any model name accepted | fixed (caveat) | `" "`, `freq_v1 ` (duplicate), `..` and a 300-character name are refused with a reason (*"⚠ Model name is longer than 60 characters"*); `"  padded  "` is trimmed. Caveat: `CON`/`NUL`/`PRN` are still accepted (new cosmetic 13). |
| 31 | "Divide target by weight" ticked while disabled | **still broken** | Weight = `(none)`: the box is disabled and still **ticked**, while the project holds `divide_target_by_weight: false`. Unchanged. |
| 32 | The Seed box shows a seed the project does not hold | **still broken** | Typing `-5` or `99999999999` leaves that number in the box while the project keeps seed 7, with no message (`1e9` shows 1 and the project takes 1). The `help="0 – 10000"` tooltip is the only hint. Unchanged — and the same defect now also bites the penalty box (new finding 5). |
| 33 | Hand-edited `fraction = 1.0` shown on the slider | *fixed (changed behaviour)* | The slider shows 0.95 and the Model page is happy; but the file is silently rewritten to 0.95 and the explanatory warning is lost in the rerun (new cosmetic 12). |
| 34 | Empty split column name | fixed | *"The split column needs a name"*; the project keeps `traintest`. |
| 35 | Pair error rendered in every Diagnostics tab | fixed | Rows = Columns = VehGas: exactly one alert, inside the "A/E by pair" tab; not visible from the Lift tab. |
| 36 | Deleted model left its run files on disk | fixed | Delete removes both the `.pkl` and the `.json`. |
| 37 | Monotone on a categorical accepted in the grid | fixed | With `monotone: increasing` on `Region` in the file, the Design page says *"Region: monotone constraints apply to numeric step designs only; the constraint was not saved"*, drops it from the project and the fit succeeds. (Verified through the file: the grid's monotone column could not be driven from the browser canvas.) |
| 38 | A constant column blocks the whole fit | **still broken** | `constant` as a predictor: *"Fit failed: Cannot derive knots for 'constant' (constant or all-null on train)"*, no warning on the Design page. Unchanged — the W3 reviewer judged refusing the fit the safe direction. |

## 4. What the new abuses tried, and what behaved well

Thirty-two abuse blocks were run that the first session did not try. Besides
the fourteen findings above, all of these ended in a clear message, a disabled
control or a correct result, with no traceback:

* **Interaction editor.** Add `DrivAge × VehGas`, then rename `VehGas` to
  `fuel` in the roles grid — the interaction follows the rename and refits.
  Make a parent categorical (type override) — fits. Add an interaction, edit a
  cell on the Rate tables page (saved as a cell adjustment
  `VehGas×Region | Regular | Centre → 1.50`), then **Remove** the interaction —
  the interaction *and* its cell adjustments go, every page renders, the refit
  succeeds. Set a parent's role to `ignore` — the interaction is dropped with a
  notice. An interaction whose minimum cell exposure is the maximum 50 % — the
  preview, the fit and the rate tables are all fine. Same variable twice, and a
  duplicate pair, are refused before the button.
* **Linear-term editor.** Clamp then knots in every order: quantile knots are
  recomputed inside a narrower clamp (100–500 on data spanning 2–27000 gives
  knots 122…393 and 7 design columns); custom knots outside the clamp are
  refused by name (*"Knots outside the clamp range: 1000, 5000 (clamp 100 –
  500)"*) and nothing is saved; widening the clamp afterwards accepts them;
  `lo == hi` is refused. Duplicate knots (`4, 5, 6, 6, 6, 15`) are de-duplicated.
* **Persisted-run folder.** Delete the `.pkl` (graceful, finding 9 above);
  fill the `.pkl` with garbage (dropped, both files removed, "Not fitted yet.");
  corrupt the sidecar `.json` (the fit still loads — the sidecar is not read
  back); make the folder read-only (fit succeeds, message names the errno —
  finding 4 above); replace the folder with a regular file (same); make the
  **data** file unreadable (`chmod 000`) and then readable again — the
  persisted fit is kept and comes back (the W3 review's S4 is fixed); delete
  the data file — the fit is kept and the message is right.
* **Project file.** Deleted under the running app — autosave recreates it with
  the current project. `\r\n` line endings — opens normally. The same project
  opened by path in one tab and by upload in another — the uploaded copy has no
  path, so it never writes to the file, and the caption says so.
* **Uploads.** A CSV with a UTF-8 BOM (columns come through clean, no `﻿`
  in a name), a parquet with columns and 0 rows (all nine pages fine), a
  200-column parquet (auto-assign 213 roles in 2 s), a CSV whose columns are
  `from`, `to`, `label`, `select`, `group by` (roles, prepare and pages fine), a
  CSV renamed `.parquet` (*"A parquet file must contain a header and footer…"*),
  and an upload whose columns do not match the project (message, no traceback).
* **Widgets, keyboard, zoom.** Text pasted into number boxes never raises
  (`abc` into alpha, "not a number" into the base-rate override, `seven` into
  the seed — all ignored; `1e9` is finding 5). Keyboard-only navigation works:
  Tab reaches the sidebar links and Enter follows them. `body.zoom` at 50 % and
  200 % on Model, Rate tables and Design: no horizontal overflow
  (`scrollWidth == innerWidth`), no exception.
* **Names.** A 300-character column name and a name with an emoji, a slash and
  brackets (`🚗 âge/du véhicule [1]`) rename cleanly, fit, and export to Excel;
  a project named `../../etc/evil name` downloads as
  `_.._etc_evil name_freq_v1_rate_tables.xlsx`; two models differing only by
  case (`freq_v1`, `FREQ_V1`) coexist because run files are keyed by a hash of
  the name.
* **Modelling.** Six rapid Fit clicks — one fit, one run file. Tweedie on the
  integer count target — fits. Binomial on counts — *"Binomial target must lie
  in [0, 1]."* Gamma on a target with zeros — *"gamma target must be strictly
  positive."* A derived column named after an existing column — *"A column
  named 'Exposure' already exists"*. Deleting the champion moves the champion to
  the remaining model. An unfitted model on Export / Rate tables / Diagnostics —
  no crash.

## 5. The misleading numbers, in plain language

* **"Model 'freq_v2' created" but you are still editing freq_v1** (new
  finding 3). The message is true — the model exists — but the page below it
  never moves. Everything you type next, and the Fit button, belong to the
  model named in the picker, which is still the old one. If you were cloning
  your champion to try a variant, you have just changed the champion. Check the
  picker after Create until this is fixed.
* **"Not fitted yet." after somebody else's tab clicked Fit or Delete** (new
  findings 1 and 2). Fits live in `breaker2.easyglm-runs/` next to the project
  file. Every open tab prunes that folder using *its own* idea of the project,
  and unlike the project file the folder has no conflict check. A second tab
  that is one edit behind, or one that is showing the conflict notice, can
  delete the fit that belongs to the project you have saved. Nothing warns you;
  you find out when the model says "Not fitted yet." and you have to refit.
  One project file, one tab, is still the only safe way to work.
* **"Could not persist the fit" long after it was persisted** (new finding 4).
  The same false alarm as the old finding 27, in the other half of the
  machinery: once the message appears it stays on every page for the rest of
  the session, even though the `.pkl` files are being written again. Look at
  the folder's timestamps before believing it.
* **alpha 190 in the box, alpha 0.001 in the fit** (new finding 5). If you
  paste a value the box cannot hold, the box keeps showing your text and the
  fit quietly uses the old number. The metric row under the Fit button
  ("alpha 0.00100") is the one that tells the truth. The same is true of the
  Seed box on the Split page (old finding 32, still open).
* **An offset that is not a log** (new finding 6). The offset is added to the
  *linear predictor*, i.e. it is exponentiated. Choosing `current_premium`
  (~250) instead of `log(current_premium)` (~5.5) means every prediction is
  multiplied by e²⁵⁰. The tool still says "Fitted and up to date" and even
  prints train A/E 1.000 — A/E on the training rows is 1.000 by construction
  for a Poisson fit with an intercept, so it is not evidence of anything. The
  −49.3 % "dev. explained" on the holdout is the number that matters here: it
  means the model is worse than no model at all.
* **"✓ Fitted" next to "results below are from the previous fit"** (new
  finding 7). After you change the family (or anything else in the spec) the
  numbers on the page are the *old* model's. The banner says so; the chips at
  the top and the sidebar tick do not. Trust the banner.
* **Every fit gone after restoring the data file** (new finding 8). A fit is
  matched to its data by path, size and modification time. Copying the file
  back from a backup changes the modification time, so all the fits are ignored
  — the work is not corrupt, it is simply no longer recognised, and refitting
  is the only way back.

## 6. How this session was run

* Server: `.venv/bin/python -m streamlit run src/easy_glm/app/main.py
  --server.port 8651 --server.headless true --browser.gatherUsageStats false --
  --project=<work>/breaker2.easyglm-project.json`, stderr to a log file, killed
  at the end; port released.
* Driver: Playwright/Chromium from the scratchpad venv, using the same
  techniques as `tests/e2e/_helpers.py` (sidebar links only for navigation; the
  retrying canvas grid editor; `settle()` around every action).
* Before each abuse the working folder was restored from a pristine **fitted**
  copy (project file, data file and `breaker2.easyglm-runs/`), so every block
  started from "Fitted and up to date"; the copy preserves modification times,
  which matters (new finding 8).
* After each block: the autosaved project JSON was parsed, the runs folder
  listed, `stException` counted on all nine pages, and the server log grepped
  for "Traceback" (final count: **0**).
* Two abuses could not be driven from the browser and were checked another way:
  the Design grid's *monotone* column (old finding 37) was set by hand-editing
  the project file, and `alpha = 0` (old finding 29) could not be typed into the
  number box at all.
