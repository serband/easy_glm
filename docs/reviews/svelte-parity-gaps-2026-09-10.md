# Svelte workflow parity — 10 September 2026

Status: **0.460 approved for release on 10 September 2026, subject to release
checks. Full workflow parity is not established; the remaining gaps below are
documented limitations, with the legacy interface still available.**

The original read-only comparison covered the rendered Svelte controls and their handlers
on main (`aa7e753`) against the Streamlit workbench. It is not a new browser or
mathematical acceptance run. The successful Positron launch test establishes
launcher compatibility and a model-fit smoke path, not feature completeness.

## The old notice is not a current feature inventory

The screenshot headed “Features still available in Streamlit” comes from an older
UI build. That notice is absent from current `frontend/src/App.svelte`. At the time of that audit, the live
server on port 8765 served `index-B-Wwlknk.js`; an already-open tab could still
be running earlier JavaScript. Subsequent updates are recorded below. No live session or browser draft was
reset during this audit.

These capabilities are now present in Svelte:

- Local data-file and saved-project opening, including dataframe handoff from Python.
- Roles, names and modelling types; bulk role JSON.
- Random and existing-column training/holdout split settings.
- Interaction add/remove, minimum cell exposure and interaction penalty settings.
- Excel rate tables, `.easyglm` scorer, project JSON, Python script and HTML report
  exports, including an optional comparison model.

References: `frontend/src/ProjectOpen.svelte`, `App.svelte`,
`InteractionEditor.svelte`, `ExportPanel.svelte`, and
`src/easy_glm/desktop/loading.py`.

## Remaining gaps

| Workflow | Missing Svelte controls or behaviour | Existing reference |
| --- | --- | --- |
| Data preparation | Create/edit recodes, derived columns and row filters. Imported project settings execute, but there is no editor. | `app/pages_variables.py:499`; `workflow/prep.py:163`; `workflow/variables.py:211` |
| Exploration sampling | In-app controls for sample size and seed. The restored one-way view honours imported sample settings. | `app/pages_explore.py`; `desktop/exploration.py` |
| Leakage | Run the leakage scan, review flags and record ignore/acknowledge decisions. | `app/pages_explore.py:56` |
| Split review | Exposure and observed-rate balance across train and holdout. Row counts and split configuration are already present. | `app/pages_split.py:223`; `frontend/src/ModelWorkbench.svelte:776` |
| Numeric factor design | Exact/custom/integer knots, per-factor bin settings and null indicators, monotone constraints, main-factor penalty weights and linear clamp bounds. Svelte currently exposes factor inclusion/kind and global bin/category defaults. Imported detailed settings are retained. | `app/pages_design.py:403,579,686`; `frontend/src/ModelWorkbench.svelte:693,732,764` |
| Categorical factor design | Choose kept levels, reference level and per-factor category thresholds. | `app/pages_design.py:835`; `frontend/src/ModelWorkbench.svelte:732` |
| Interaction review | Pre-fit cell-exposure and retained-cell preview for the proposed threshold. Interaction editing itself is already present. | `app/pages_design.py:1110`; `frontend/src/InteractionEditor.svelte:62` |
| Base rate | Direct base-rate override and solving for an arbitrary target A/E or loss ratio. Current rebalance restores the fitted training total only. | `app/pages_model.py:429,512`; `frontend/src/ReviewPanel.svelte:1164` |
| Project saving | Project naming and an in-app save/autosave path. Svelte currently supplies an explicit project JSON download. | `app/pages_project.py:219`; `frontend/src/App.svelte:747` |
| Resume fitted work | Normal restart or project reopening does not restore fitted runs. Svelte starts with empty fit managers and temporary artifacts; project opening also clears session undo/redo. Streamlit autosaves applied project changes and restores valid cached fits alongside saved projects. | `desktop/jobs.py:40,253`; `desktop/server.py:274`; `app/state.py:426,661,684,848,912,1121` |

All Python paths in the table are relative to `src/easy_glm/`.

Browser reload while the server remains alive retains server state. The private
`--restore-session` facility supports controlled server upgrades, but is not the
normal save/reopen workflow and does not close the persistence gap.

## Release implications

The old notice mixed completed work with genuine omissions. Removing it does not
complete the migration. The remaining data-preparation, leakage and factor-design
controls and ordinary save/resume behaviour are release blockers for the promised
Svelte replacement and need to be available before claiming parity.
Existing shared transformation, leakage and design code should be reused;
there is no architectural requirement for users to switch interfaces.

Release validation must cover creating those settings in the UI, their effects
on the prepared data and fitted model, and saving/reopening the resulting work.
Preserving settings loaded from JSON is not equivalent to providing their editor.

## Follow-up: data opening

Data opening is now implemented locally. Project & data now
starts with visible Own data, Saved project and Example dataset choices, native
file browsing, a path alternative and explicit file-format selection. Existing
work is replaced only by a confirmed Open/Load action. Example choices do not fit
automatically. Developer review names are removed from the displayed seed project.

The independent browser walkthrough used the actual public datasets: 50,000
French motor rows and 64,548 Swedish motorcycle rows. Both loaded with the
expected editable roles, split and model, and completed an explicitly requested
fit. The reviewer also exercised native CSV selection, project JSON export and
reopening, sample switching, and a failed load after fitting that preserved the
project and fit ID. Screens were inspected at 884 and 919 pixels wide.

Focused validation passed: 35 backend loading/API tests, four onboarding browser
regressions, five browser reconnection tests, and the model-controls browser test.
The browser regressions include successful upload followed by failed connection
refresh, and a stale second tab that must reconnect and explicitly confirm Open
again. The UI adopts the confirmed new project after a successful Open even if
the next read fails. Reconnect does not replay the replacement request.

The reviewer found an empty predictor panel, truncated family descriptions, a
dead interaction link and irrelevant retained-settings copy in a new model.
These have been corrected: short family names have a selected-use description,
and the empty model points directly to assigning predictors in Variables.

The Swedish warnings were traced to the intercept-only diagnostic benchmark,
not the selected model or its cross-validation candidates. Its gradient threshold
was below the floating-point line-search accuracy on cost data. The benchmark
threshold is now 1e-8; main-model fitting is unchanged. A real Swedish run at ten
threads emits no warnings, and its benchmark prediction agrees with the weighted
mean oracle to 6.25e-10 relatively. All 150 focused null-benchmark, workflow,
family and export tests passed. One unrelated direct-fit test retains its own
convergence warning.

This closes data-opening discoverability and the observed benchmark warning,
not the remaining preparation, exploration, factor-design or persistence gaps.
Full workflow and actuarial sign-off remain open.

The local workbench on port 8765 was updated after a successful restore rehearsal
on port 8798. Verification retained the exact applied project, fit ID, all eight
rate tables and four Undo steps; diagnostic results agree to floating-point
precision. A fresh Chrome page loaded the new bundle with all three opening
choices visible and no browser errors. The live model was not refitted or reset.
No release tag or package publication was performed.

## Follow-up: one-way exploration

Explore now uses the shared observed-rate calculation on applied training data,
with numeric bands, categorical levels, missing/distinct summaries and a
downloadable values table. Observed response uses the left axis; exposure or
row counts use bars on the right. Variable, model and band changes update the
view automatically. No fit is needed. Imported recodes, derived variables,
renames, filters and modelling types are reflected in the available variables.

The chosen model supplies its target, weight and target-division convention
together. Unweighted models show a mean; weighted rate models show a weighted
mean; total-target models show total target divided by exposure. The training
split is applied before seeded sampling. Cached results are scoped to the
project revision and session, with stale response guards on both sides.

Shared aggregation fixes keep missing outcomes out of the rate denominator,
retain their exposure, and correct pooled categories with integer weights or
separate missing levels. Numeric NaNs now join the missing group. Zero exposure
produces an undefined rate, not a fabricated zero.

The backend passed 96 focused exploration, workflow and desktop API tests, plus
Black, Ruff and the core/workflow type check. Real French and Swedish cached
responses took less than 0.4 ms in the local check. A further 24 focused tests
passed after making the default selection prefer a model predictor over an ID.

Independent browser review exercised both real examples before fitting. French
frequency, pre-divided rates, unweighted means and weighted binary outcomes
matched direct group calculations. Replacing all holdout outcomes and predictors
with extreme values left the training charts unchanged. Imported derived numeric
and categorical variables, a type override and negative outcomes rendered
correctly. Long category labels were corrected and inspected at 884 and 919 pixels.

Four Variables regressions, five actual restart/reconnection regressions and four
data-opening regressions passed. The synthetic demo now configures its training
split; a genuine missing split offers a direct setup action. The restoration
rehearsal retained the exact project, fit ID, eight tables and four Undo steps.
Six Explore browser regressions also passed, including measured category-label
spacing, late responses, missing outcomes and setup links.

The verified update is running locally on port 8765. A fresh browser check
confirmed the one-way chart, 34,887 training rows and no browser errors. The
applied project, fit ID, all eight tables and four Undo steps were preserved;
no refit or adjustment was made. At that point, full workflow parity and release
approval remained open.

## Release decision

After reviewing the data-opening, one-way exploration and visual changes, the
user explicitly approved release on 10 September 2026. The package version is
0.460 so it sorts after 0.452 for normal pip upgrades. This approval supersedes
the earlier release hold; it does not establish full Streamlit feature parity.
The README and changelog describe the remaining legacy-only controls and explain
that reopening project JSON requires refitting. Release validation is recorded
separately from the historical browser checks above.
