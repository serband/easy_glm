# EasyGLM handover

**Snapshot: 11 October 2026.** Start here when taking over without the previous chat. Check Git and the current package registry before changing a status claim: this is a local snapshot, not proof of today's remote state.

## Start in the right checkout

| Purpose | Checkout and state |
| --- | --- |
| Current implementation | `/private/tmp/easyglm-model-variable-setup`, `codex/score-new-datasets`, based on `59ace19`; new-data scoring alongside named models, GLM-owned interactions and accepted workflow exports |
| User's primary workspace | `/Users/serban/Documents/Projects/easy_glm`, `codex/variable-screening`, `cb66d3648d47ebf62f39c0461daff6d2e18dbc69`; contains unrelated modified and untracked work |
| Last known public release | `v0.472`, locally verified tag at `132e1e14433f849517fc225d2efea5b76209ef23` (23 September 2026) |

The current stack is **local branch work, unpushed and unreleased**. Its `pyproject.toml` still says `0.472`; that version string does not make the stack part of the public package. The primary workspace's `main` is older (`4eb8c8a`) and must not be used as a release baseline. Remote GitHub/PyPI status was not live-verified for this snapshot because network name resolution was unavailable. Do not reset, clean, switch, or prune the primary checkout. Inspect `git status`, branch, commit, and file paths before acting.

On 10 October the previous `/private/tmp/easyglm-visual-model-flow` directory was
absent. This checkout was created from its surviving committed branch. Earlier
uncommitted changes must not be assumed recovered. The current plan has one name
and method per model. The previous selected-model builder is a historical
baseline, not the current screen layout.

The primary workspace carries an older copy of this handover and selected guidance, but its code does **not** contain the current stack. Read the [workflow map](../examples/workflows.md), `docs/plans/STACKED_MODELS.md`, `docs/LLM_USAGE_GUIDE.md`, and the source/tests **in the implementation checkout**; the primary workspace's documents may differ. `examples/pricing_walkthrough.md` covers the earlier released pricing flow. The updated usage guide separates released features from the current staged work. The locally available `v0.472` Git tag provides the release source for comparison (`git show v0.472:<path>` from the stack checkout); verify remote state separately when network access returns.

## What the package does

EasyGLM is an insurance-pricing Python package. The released v0.472 package fits regularised GLMs, turns their coefficients into exact rating tables, scores from those tables, and includes a local browser workbench and command-line fit/export/validate commands. The simple API is `EasyGLM.fit`; the interactive actuarial API is `PricingSession` → named `PricingModel` checkpoints; lower-level users can use `DesignSpec` → `fit_glm` → `rate_tables`/`to_rate_model` and `RateModel` for portable lookup scoring. The v0.472 interactive path also supports separately fitted two-factor CatBoost interaction tables, predictor screening, training/CV diagnostics, explicit holdout inspection, rate amendments, and save/reopen. Consult the release tag and `CHANGELOG.md` for exact released details.

The existing core accepts Poisson, Gamma, Gaussian, inverse Gaussian, Tweedie and binomial families, with log links by default except binomial's logit; compatibility and target constraints vary by feature. **Mixed stacks support only Poisson, Tweedie and Gaussian with log links.** The default browser workbench is the local FastAPI/Svelte app (`python -m easy_glm.desktop`, or `python -m easy_glm.app`); the older Streamlit UI is available through `python -m easy_glm.app --legacy-streamlit`. `easy-glm run|export|validate|workbench` is the project CLI. The standalone `.easyglm` scorer, project JSON, and fitted stack bundle serve different purposes; check which artifact a command expects.

## What exists and what remains

The stack branch implements an ordered multiplicative sequence of GLM, direct CatBoost, separately accepted two-factor interaction tables, and complete catalogue lookups. A component fits the original response with the log of the **complete accepted predecessor scorer** as offset. The scorer includes earlier amended tables/files, the declared external offset once, and exposure once when converting unit predictions to expected amounts. A correction-only first component gets an internal fitted baseline. Supported mixed-stack families are Poisson, Tweedie, and Gaussian with log link; do not assume other existing package families/links work in a mixed stack.

`PricingSession` and named `PricingModel` objects provide short public calls (`fit_glm`, `fit_catboost`, `fit_interaction`, `fit_catalogue`, `ae`, `compare`, `edit_rates`, `export_lookup`, `save`/`load`). The workbench has a stacked-model path and component inspection, comparison, amendment, and export views. Named fits are detached; append and amendment create new models. The Python example demonstrates GLM → two pair tables → vehicle → reviewed vehicle → occupation → postcode, then a separate CatBoost → GLM candidate. Catalogue fitting pins the full supplied key/attribute table, scores keys without training rows, checks joins/coverage, and exports an **incremental component** file. A catalogue-first file is a full unit prediction only when there is no external baseline; otherwise the external offset remains separate in the complete scorer. Missing keys error by default unless an explicit fallback is chosen.

**Current product work: interactions within named GLMs.** New plans define
GLM, CatBoost or Catalogue models. Interaction is no longer a new model type in
planning. A fitted GLM owns optional interaction tables added after residual
review. Candidate setup and fitting are separate from acceptance; only Keep
updates the GLM's accepted output and a downstream model's offset. Each later
pair uses the complete preceding table scorer, including amendments. New plans
write model-plan format 2 and project schema 7; existing flat plans keep format
1/schema 6 without inferred ownership or automatic regrouping. A sole GLM with
no tables retains its direct fitting and scoring path. Nested tables are
compiled to the existing native stack engine; earlier public pair APIs remain
unchanged. Read `docs/MODEL_FLOW.md` and
`docs/plans/PROJECT_MODEL_WORKFLOW.md` for the workbench workflow.

**Guided project setup.** **Project & data** shows one
setup section at a time, with a step strip and a compact summary of the loaded
data, target, weight and split. Open data, choose **Single model** or **Build in
stages**, then apply shared column assignments. Single-model setup continues
through variable selection, model details and treatment of the selected factors.
Staged setup continues through numbered model names/types and independent
variable selection for each model. Each variable editor names its model and
complete incoming offset. The strip supports returning to earlier steps; reload
resumes incomplete setup, and the URL identifies the current model editor.
**Variables** is no longer a sidebar destination. Predictor screening remains
under **Explore**.

**Model** is the fitting page. It shows the complete ordered sequence and a
named action, such as **Fit Model 2 — Vehicle**. Fitting options can be saved
without starting a fit. A model with no variables is labelled **Variables needed**
and links to its editor. Blank future models do not prevent fitting the configured
first model. Fitted settings lock immediately; revisions are explicit. The step
strip reports the current plan's fitted count rather than borrowing the status of
a historical fit. Unsaved shared assignments, variables and fitting options are
protected when navigating. Saving variables advances to the next model or opens
Fit without a false discard prompt. Existing saved plans remain supported.

**Add model** appends a GLM card named **Model N** and focuses its type. It does
not open the variables editor. Each card shows its complete preceding offset
source; an adopted root shows the exact saved name and **Saved fitted model**,
with its accepted scorer locked. There is no separate next-model type selector.
Appending or saving downstream variables preserves the adopted source and exact
fit reference; replacing that source or editing its frozen recipe is rejected.

The current plan is separate from immutable saved fitted versions. **Model structure** in the setup strip opens the Single model / Build in stages chooser. Choosing a
route and confirming replaces the working setup and exact fitted references;
**Keep current plan** returns without changes. Saved fitted versions are kept.
**Create revised plan** explicitly unlocks fitted setup: unchanged earlier fits
are reused by their exact identities, and the changed stage and suffix need new
fits. Existing fitted models or whole chains can be adopted explicitly as a
frozen starting scorer and extended. Its accepted adjustments and interaction
tables remain in the offset. Replacing that frozen starting scorer requires a
new independent plan; it is not an editable/refitted GLM placeholder. After a
fit, **Revise shared assignments** starts a plan copy before unlocking shared
roles. Applying role changes requires a new root fit; the old recipes remain,
but their results do not match the changed assignments.

Each component selects its own factors from safe predictor and unassigned source columns. Its **Variables for this model** editor provides searchable alphabetical Table and JSON views. Global roles identify response, weight, ID, split and ignore; those controls cannot be changed in this editor. GLMs and interaction tables now store their own defaults, types and bands in optional `ComponentRecipe.design` snapshots. Direct CatBoost uses raw values and local categorical/numeric treatment. Legacy recipes without this field retain their previous project-design behaviour and serialize without the new field, preserving existing fingerprints. A child's target, weight, family, log link and split remain inherited; its offset is the complete fitted predecessor scorer. Model-save validation rejects changes to the response convention or frozen prefix and refuses edits to global bands through a stack-model save.

New direct CatBoost and CatBoost catalogue drafts default to **Optuna tuning**:
eight trials and three training folds. Choose a trial count or a tuning-time
budget (15 minutes initially); these are separate limits. Time charges only
this component's candidate CatBoost fits across training folds. Preparing data,
refitting earlier components, scoring and the final full-training fit are outside
that budget. A CatBoost fit already in progress can finish before tuning stops.
Every counted trial is sampled by Optuna. The unchanged preceding score is a
separate comparison, not a trial; it is kept when no completed trial improves it.
The results record completed/pruned trials, the stopping reason and either the
chosen parameters or **No change selected**. **Manual settings** fits the entered
parameters once without Optuna or CV. Neither mode uses holdout rows for fitting
or selection. Recipes without `tuning_budget`, including older `tune: true`
recipes, keep their old combined trial/time behaviour when reopened or saved.
Existing interaction-table tuning is unchanged. Python `fit_catboost` accepts
explicit `tune=True`/`False` and `tuning_budget="trials"`/`"time"`; older calls keep
their previous behaviour.

Diagnostics now identifies **Saved chain (final model)** and **Fit to inspect**.
The default is the final combined fit; earlier choices score the original response
with the saved physical components through that model, including earlier offsets
and amendments. They are not isolated correction multipliers. The API accepts
optional `through=<component_id>`; omitting it preserves full-score behaviour.
Internal baselines and virtual legacy inspection pieces are not menu choices.
Grouping factors and catalogue coverage come from the selected prefix. Later
catalogue-only attributes are unavailable before their join. Switching models
resets to training and the final fit; stale asynchronous responses cannot replace
the current model, scope, subset or factor. See `tests/test_stack_diagnostic_scope.py`
and `frontend/tests/stack-diagnostics.spec.js` for the contracts.

A child's own complete fit can still be used as the next offset when its original incoming parent is unavailable; that missing parent blocks revision of just the last addition. If its own fit is missing, rebuilding creates a new named copy of the whole saved recipe. Fitted stack settings and fit endpoints reject in-place overwrite. Single GLMs keep their existing families, links and penalty controls. Existing saved models retain their scoring, inspection and export APIs. See `docs/MODEL_FLOW.md` for the user workflow.

The 9 October correction separates ordinary drafts from saved setups that need recovery. **Needs a new fit** means the saved setup's exact earlier starting fit is unavailable; a currently fitted model with the same name is not a replacement for it. **Fit saved recipe as new model** fits the saved sequence from the beginning under a new name. **Cannot score** means the selected model's own fitted files are unavailable. A usable complete scorer remains **Fitted** even when its original starting fit is missing. The header shows identity, context and actions in separate rows; a read-only saved recipe shows the fitting order.

`ModelConfig.lineage` records the parent name, fit identity, settings key and prefix fingerprint. It is provenance, excluded from fit/cache hashes; the complete ordered component recipe remains the scorer. New logical plans write schema version 7; old flat plans write version 6; lineage-only projects write version 5, and older projects without either retain their prior serialized version. `Project.model_plan` contains server-issued plan IDs and exact fit references. Logical references identify the named model and its accepted through-component checkpoint. Imported references are checked against the actual saved recipe and artifact; equal-looking recipes do not select unrelated historical fits. An appended candidate requires an exact applicable parent artifact and unchanged frozen prefix. Changing a parent does not mutate its existing children. Missing or changed artifacts produce historical links; an explicit full rebuild fits the whole saved recipe afresh and remains marked historical after job restore. Scope validation runs on save and fit, including imported projects, and protects response/control fields against catalogue-name collisions.

The Model builder work is local, unpushed and unreleased. Fresh validation for this redesign is recorded below; the earlier visual-flow results are a separate baseline. Keep discovery on training data, label in-sample and CV evidence accurately, and display holdout results only on explicit request. Owned interaction-table changes need preview then apply; affected later tables and models require refitting. The existing generic stack amendment API retains its separate explicit policy. A/E and searches must use the accepted complete scorer; earlier named fits must remain unchanged.

Known first-version boundaries: each named model scores a linear sequence; the diagram can show alternatives but does not combine two parent models; one-key, user-supplied catalogues; no arbitrary imported estimators, automatic hierarchical credibility, geographic smoothing, or mixed-stack HTML report. A direct CatBoost component requires its saved model artifact to score and cannot be represented exactly by GLM rate tables alone. Confirm any further limit in code before asserting it.

## Source map and checks

**New-data scoring, 11 October.** The workbench separates the modelling source
from named additional scoring datasets. Local CSV/Parquet references are optional
`Project.scoring_sources`; uploads and frames passed through
`launch(data=..., scoring_data={...})` are private session snapshots. These sources
do not enter a fit key. **Score on new data** selects a saved fitted version and
scores its complete chain, a cumulative checkpoint, an individual accepted model,
or a separate native catalogue candidate. Frozen column transformations apply;
training filters and split generation do not. Requirements follow the selected
scope. Exposure is optional and applied once; contributions and catalogue
candidates exclude it. Original rows and columns are retained.

`PricingModel.score` returns a `ScoringResult` with `preview`, `frame`, metadata
and CSV/Parquet `export`. Exports have a neighbouring JSON metadata file. The GUI
also downloads score details with the scored file. Accepted catalogue rates remain
lookup-based, including amendments. Newly fitted catalogues retain their native
model for scoring refreshed masters as candidates; older lookup-only artifacts
report that this option is unavailable. Neither candidate scoring nor registering
a dataset changes an accepted model. See
[`examples/new_data_scoring.md`](../examples/new_data_scoring.md).

Independent software, adversarial, rendered design and pricing-actuary reviews
passed. The full Python suite found one existing binomial/logit results-page
regression: 1,707 passed, one failed, one skipped and one deselected. That
regression was repaired, and the affected model-plan and scoring checks passed
on recheck. Final scoring, desktop and worked-workflow checks passed: 29 tests
from source and 24 scoring/API tests against the final installed wheel. The
full suite was not repeated after the narrow repair. Black, Ruff and mypy passed.

Five isolated browser cases passed, covering complete-book CSV scoring,
missing-input blocking, key-only vehicle-file Parquet scoring with numeric
read-back parity, separate native candidate output, delayed-response
invalidation, details JSON and the 797px layout. All six public workflow guides,
including this new scoring example, ran from their literal Python blocks
against an installed wheel, using the full example datasets. The launcher also
registered Polars and pandas scoring frames in a separate process. These are
local checks, not remote CI or release approval. The user's workbench on 8881
was preserved; the isolated scoring preview is on 8891.

**Documentation and worked workflows, 11 October.** Start at
[`examples/workflows.md`](../examples/workflows.md). The current GUI walkthrough
replaces obsolete Variables-sidebar instructions with guided Project & data
setup, independent model variables, inline GLM-owned pair review, cumulative
diagnostics, revisions and distinct scoring/rebuilding exports. The rating-file
example works through main effects, two pair checkpoints, vehicle review,
occupation and postcode catalogues, including keys absent from training.
Model-order examples cover standalone CatBoost, GLM → CatBoost, CatBoost → GLM,
tuning limits and fixed/refitted suffix amendments. The Project guide explains
`from_dict` → preparation → validation → fitting, and executes both frozen-score
and accepted-recipe exports. `model_plan` alone is not a public `run_model`
executor; that function fits explicit entries in `project.models`.
The model-family examples use the built-in Swedish motorcycle loader for
Tweedie pure premium, positive-payment Gamma severity and binomial claim
occurrence, with separate response and weight conventions.

These guides describe current source behaviour where marked and do not imply
that unreleased stack features ship in PyPI v0.472. No development installation
instructions or release changes were added.

The Python blocks were executed from the markdown, including the full French
motor sample, the 6,000-row catalogue example and the cached 64,548-row Swedish
motorcycle data. Four new regression tests cover catalogue coverage, rate
amendment policies, family units, and frozen-score/rebuilt-recipe prediction
parity; the two existing French walkthrough tests also pass. CI uses smaller
local fixtures for the new tests. Independent documentation and actuarial
reviews passed; Black, Ruff and local-link checks passed. The full Python suite
and installed-wheel checks were not rerun for this documentation change.

Both documented GUI exercises passed on an isolated workbench: the French
motor chain with an accepted interaction, every diagnostic checkpoint, and
syntax-checked Python scorer/workflow downloads; and four catalogue-workflow
fits with full lookup exports of 80 vehicles, 30 occupations and 120 postcodes.
The catalogue exercise now includes the required training-value selection and
Preview/Apply steps. The temporary validation server and environment symlink
were removed; the user's workbench on 8881 was not replaced.

**Workflow export, 11 October.** The stacked Export page now separates
**Score new data** from **Rebuild this model**. **Download Python workflow**
returns a `*_workflow.py` script that refits the selected accepted component
recipe in order, retaining its saved preparation, split, amendments and incoming
offsets. A frozen adopted starting scorer remains frozen. `SOURCE_PATH` is
editable near the top; the script writes a fitted bundle and Excel workbook.
Refitting can change fitted values. The existing Python scorer uses the saved
fits without refitting. Exploratory searches and discarded candidates are not
part of the accepted recipe replay.

The generator reads the fitted project snapshot and deep-copies its accepted
configuration. Catalogue data is pinned from the saved lookup when its source
file is unavailable. Downloads leave project, jobs and fitted artifacts unchanged.
Fresh validation: **29 Python export tests**, including a fresh-process
GLM → capped pair table → refitted CatBoost replay matching the accepted score
at `rtol=1e-6`, and a deleted-catalogue-source replay; a browser download check
with desktop and 390px layout review; Black, Ruff, mypy (39 source files),
Svelte check and production build. Independent design, adversarial and pricing
actuary reviews passed. The full Python suite and installed-wheel examples were
not rerun for this correction. Preview `127.0.0.1:8881` was upgraded with all
four current fitted versions restored; scoring values were verified unchanged.
Nothing was pushed or released.

**Review placement, 10 October.** Each fitted GLM's review is now directly
inside its own Model row, with a heading such as **Review Main effects**. The
next model follows that review and its continuation action. Separate reviews
retain their own candidates when collapsed; an upstream score change refreshes
an affected downstream review before its candidate can be kept. If that change
invalidates the downstream fit, its row shows that refitting is needed and its
review controls remain unavailable until it has a valid fit again. Shared plan
updates reject older revisions from another review's background request.
Independent design and pricing-actuary reviewers passed the live three-model
layout at desktop and narrow widths. This is a frontend correction; fitting and
scoring APIs are unchanged. Validation: Svelte check (zero errors/warnings),
production build, 30 frontend unit checks, two rendered Gaussian/binomial smoke
checks, and ten core browser cases covering inline ownership, candidate retention,
ordered reload, downstream invalidation/refit and delayed-search responses. Browser
cases were rechecked individually after fixes; the Python suite was not rerun
for this frontend-only correction. Nothing was pushed or released.

**GLM-owned interaction tables, 10 October.** New plans offer GLM, CatBoost and
Catalogue models. Candidate interaction tables belong to a fitted GLM and have
separate Fit, Keep and Discard actions. Independent rendered design and pricing
actuary reviews passed the planning, residual search, table review, acceptance,
amendment and downstream-refit workflow. Desktop and narrow layouts were reviewed.
The adversarial review verified source-bound acceptance, immutable fitted
checkpoints, failed/cancelled suffix retries and duplicate-fit prevention.

The broad Python source run passed **1,683 tests**, with one skipped and one slow
test deselected. Subsequent fixes for direct Gaussian/default-link and
binomial/logit plans passed **63 focused Python tests** and a separate **five-test
Gaussian/binomial review**. The reviewer also checked logit diagnostics and export,
and confirmed that a non-log single model cannot acquire multiplicative
interactions. Black, Ruff, mypy (39 source files), all **30 frontend unit tests**,
Svelte check and the bundled build passed.

Fresh browser checks passed **15 model-plan** and **seven core-interaction**
scenarios. The same validation cycle also passed **seven guided-setup** and
**five model-variable/flow** scenarios. The core checks cover candidate discard,
training-only search, two accepted tables after reload, amendment/refitting,
legacy plans and adopted roots. The public pricing walkthrough ran unchanged
against the source. Separate numerical checks confirmed the second pair offsets
the amended first lookup table, and that exported accepted scorer predictions
match training predictions to 1e-13 while an unfinished later model is excluded.

The user's existing preview retained its exact legacy plan, shared assignments,
50,000 rows and 13 columns after restart. The new-plan review preview is separate.
These are local macOS source checks. Windows, installed-wheel release checks,
GitHub CI and the separate 5M-row budget were not run for this change. Nothing
was pushed or released. Earlier records below are historical baselines.

**Guided-setup correction, 10 October.** This revision replaces the long combined
setup page with focused steps. Independent source review covered draft handling,
queued name/type saves and Svelte reactivity for offsets, fitted locks and step
statuses. Rendered design review covered 1373px and 797px widths. A separate
pricing-actuary reviewer configured independent factors, fitted a single GLM and
a GLM → Vehicle CatBoost sequence, and inspected named cumulative training A/E.
The focused Python model-plan suite passed **23 tests**, and the frontend unit
suite passed **27 tests**. The final frontend build passed Svelte checking with
no errors or warnings. **28 browser tests** passed: 15 model-plan/shared-role
scenarios, seven guided-navigation scenarios, five model-variable/flow scenarios
and one 500-variable scenario. These include real catalogue fitting, table review,
protected drafts and replacing data followed by reload. Independent design,
source, adversarial and pricing-actuary reviews closed their findings. These are
local source checks, not release validation.


**Numbered-model correction, 10 October.** The model list now keeps the current
name, type, variable action and cumulative offset together. Appending keeps the
new card visible and focuses its enabled type selector. Adoption checks found
and fixed a rejected append/save of an unchanged adopted source; frozen-root
and different-source edits remain rejected. Its variables view now describes
the preserved scorer instead of displaying an empty GLM placeholder.

The focused model-plan Python suite passed **23 tests**. Svelte check/build,
Black, Ruff and the **27 frontend unit tests** passed. Focused browser checks
passed **15 model-plan**, **five model-variable/flow**, and **one 500-variable**
scenarios, including strict new-card focus and adopted append. Independent
rendered review covered one and three cards at 1150px and 797px. The actuarial review
configured three independent variable sets, fitted GLM and Vehicle CatBoost,
confirmed the third model's eligibility, configured CatBoost → GLM, and extended
an API-seeded adopted scorer through the UI. The third and reverse-order models
were configured rather than fitted. The route chooser does not expose saved-fit
adoption; the adopted-plan test starts from the existing API contract. The live
preview was restored with its exact project, 50,000 rows and 13 columns.
These are local source checks, not an installed-wheel or Windows release check.

**Consolidated-setup baseline checks, 10 October.** The focused Python suite
passed **94 tests** covering model plans, shared roles, split settings, predictor
screening and feature selection. Svelte check/build, all **27 frontend unit
tests**, Black, Ruff and mypy passed. The final browser build passed **14**
single/staged-plan scenarios, **seven** Explore scenarios and all **five**
server-reconnection cases. The remaining affected legacy fitting, bands,
interaction, screening, diagnostics, export and wide-variable suites also passed
on isolated fixtures. The wide checks exercised 500 model variables and a
2,000-column shared editor; the latter kept at most 28 table rows rendered.

Independent rendered reviews covered single and staged setup at desktop and
narrow widths. A separate pricing-actuary review fitted GLM and CatBoost stages
with different variables, inspected cumulative training A/E, and exercised bulk
JSON selection and protected shared assignments. Independent software and
adversarial reviews closed their findings. Both public walkthroughs executed all
**48 Python blocks unchanged** against the source, including scoring and exports.

Recovery checks found and fixed a false Explore warning after a server restart,
raw JSON being reformatted on a repeated active-tab click, and stale project
metadata after explicit discard/reload. Reconnecting keeps drafts, invalidates
old previews and blocks applying a draft to a replacement project. These are
local macOS source checks; installed-wheel, Windows, GitHub CI and the separate
5M-row budget were not rerun for this setup change. Nothing was pushed or released.
The validation records below describe earlier changes and are historical.

Project-page layout correction on 10 October: one page title, matching card widths
and padding, consistent section typography, and aligned stage fields. Independent
designers reviewed rendered empty and staged pages at desktop and narrow widths;
their layout findings were fixed. Svelte check/build and the five model-plan and
four data-opening browser tests passed locally. No modelling or API changes,
push, or release were made for this correction.

The configured Model stages form received a further correction after user review:
Family/Link, Stage name/Method and the next-stage controls now share a two-column
grid, labels above fields and 38px control heights. Help text and the response
checkbox occupy separate rows. Independent rendered reviews covered staged
Poisson and Tweedie forms at desktop and 800px widths, plus a three-stage plan
and its Variables navigation. The primary agent verified the user's configured
Poisson form and adding a stage in an isolated project. Svelte check/build and all
five model-plan browser tests passed. The single-model form was inspected in
source and exercised by the browser tests, without a separate visual review.
This correction changes presentation only; nothing was pushed or released.

- `src/easy_glm/pricing/session.py`, `model.py`, `review.py`, `views.py`, `validation.py`, `persistence.py`, `excel.py`: public interactive setup, named checkpoints, edits, diagnostics, comparisons, save/load and Excel.
- `src/easy_glm/workflow/stack_types.py`, `stack.py`, `stack_tuning.py`, `catalogue.py`: component recipes/artifacts, cumulative scoring and offsets, fold-aware tuning, catalogue validation and lookup.
- `src/easy_glm/workflow/project.py`, `workflow/model_plan.py`, `desktop/model_plan.py`: shared roles, optional plan, local single-model design, JSON migration, prefix validation and immutable fit materialisation. `src/easy_glm/desktop/` plus `frontend/src/StackWorkbench.svelte` and `frontend/src/App.svelte`: workbench APIs and screens.
- `tests/test_pricing_stack_api.py`, `test_stack_tuning.py`, `test_stack_exports.py`, `test_desktop_stack.py`: targeted stack tests; `frontend/tests/stack-visual.mjs` and `stack-legacy-append.mjs` are local browser review scripts, not a dedicated `playwright.stack.config.js` CI job.

Fresh CatBoost-control checks on 10 October: the full Python source run recorded
**1,627 passed**, one skipped and one slow deselected. Nine launcher cases were
blocked by the sandbox's localhost-binding restriction; both launcher suites
then passed **10 tests** with local socket permission. The final focused fitting
suite passed **47 tests**, including strict setting validation, neutral selection
and poisoned-holdout checks. All **27 frontend unit tests**, **nine staged-model
browser scenarios**, Svelte check, Black, Ruff, mypy and the bundled build passed.
Both public walkthroughs ran all **48 Python blocks unchanged** against the source
checkout, including scoring and exports. Independent software, adversarial and
rendered actuarial reviews closed their findings. These are local macOS source
checks; installed-wheel, Windows, GitHub CI and the separate 5M-row budget were
not rerun for this change. Nothing was pushed or released.

Fresh project model-plan checks on 10 October: the broad source run passed
**1,643 Python tests**, with one skipped and one slow deselected. The two launcher
suites separately passed **10 tests** with localhost socket permission. After
the final catalogue-readiness and project-replacement fixes, the plan/onboarding
suite passed **35 tests**. Ruff, Black, mypy for all 49 checked source files,
Svelte check/build and all **27 frontend unit tests** passed.

The new browser coverage exercises sequential fitting, independent definitions,
training-only Optuna controls, a complete uploaded catalogue, 500-column JSON
selection, supported single-model families, cumulative diagnostics and reload.
Existing browser suites use real saved legacy models for compatibility checks;
they do not force a hidden workflow mode. Upload and server-restart recovery are
also tested. Every browser suite listed in the workbench CI job passed locally,
using isolated servers where ports overlapped. Independent design, software,
adversarial and pricing-actuary
reviews exercised the revised screens and closed their findings.

Both public walkthroughs ran all **48 Python blocks unchanged** against the final
wheel in a fresh environment, including scoring and exports. The installed
Python files and bundled frontend assets match the source. These are local macOS
checks; GitHub CI, Windows and the separate 5M-row budget were not rerun. No push
or release was performed.

Fresh per-model definition checks on 10 October: **71 Python pricing/stack
regressions**, **27 frontend unit tests** and **six browser scenarios** passed.
The browser scenarios include root/child fitting with different cuts and factors,
uploaded CSV catalogue fitting, draft recovery/retry, and bulk selection from
500 available columns at desktop and 804px widths. Svelte check, Black, Ruff,
mypy for all 48 checked source files, and the bundled frontend build passed.
Independent software, adversarial and rendered pricing-actuary reviews closed
their findings. CSV numeric join keys now align safely to numeric source IDs;
leading-zero identifiers remain protected. These are local source checks; no
fresh installed-wheel, GitHub CI, Windows or 5M-row run was performed for this
change. Nothing was pushed or released.

Fresh diagnostic-scope checks on 10 October: **52 Python stack/diagnostic
regressions**, **27 frontend unit tests** and **six browser scenarios** passed,
including a three-model fit, grouped A/E, explicitly requested holdout, switching
saved models, delayed stale responses and a 797px layout. Black, Ruff, mypy,
Svelte check and the bundled build passed. Independent software, adversarial and
rendered pricing-actuary reviews closed their findings. These are local source
checks, not installed-wheel, Windows, GitHub CI or 5M-row validation. No push or
release was performed.

Earlier visual-flow checks on 8 October, before the selected-workspace redesign: **1,615 Python tests passed**, 1 skipped and 1 slow deselected; Black, Ruff and mypy passed. The new model-flow browser scenario, existing model-editing and interaction-reuse scenarios, all 18 frontend unit tests, Svelte check and static build passed. Both public walkthroughs completed all **48 Python blocks** against an installed wheel, including Excel and complete catalogue exports. Software, adversarial and pricing-actuary reviews closed their findings. These are local macOS checks, not a GitHub CI or Windows release check.

Fresh selected-workspace checks on 8 October: the broad Python run recorded **1,614 passed**, 1 skipped, 1 slow deselected and one launcher failure while the frontend build replaced static assets. With the build finished, the desktop, model-flow and launcher suites passed **278 tests**, including that failed launcher check. The final missing-artifact guard then passed **276 tests** in those suites; two test fixtures needed the new availability field/completed-artifact contract and both closed in a **21-test** diagnostic/progress rerun. The focused stack/model-flow suite passed **8 tests**. Black, Ruff, mypy, all 18 frontend unit tests, Svelte check and the bundled static build passed. The selected-model browser scenario, both existing single-model scenarios, interaction-reuse scenario and all six Variables scenarios passed. Independent software, adversarial and pricing-actuary reviews closed their findings, including delayed fit status, draft recovery, missing artifacts and guarded mobile navigation.

Both public walkthroughs ran all **48 Python blocks unchanged** in a fresh installed-wheel environment. Fresh-process checks confirmed full catalogue-key coverage, accepted multiplier export parity, cumulative model scoring, exposure applied once and readable Excel XML. The final wheel's frontend assets match the source build. These are local macOS checks; GitHub CI, Windows and the separate 5M-row budget were not run for this redesign. No release or push was performed.

Fresh recovery-workspace checks on 9 October: all **18 frontend unit tests**, Svelte check and static build passed. The model-flow, recovery-view and retry browser suites passed on the final bundle. These cover desktop/mobile recovery layouts, unavailable starting fits, missing fitted files with an available parent, and a delayed failed-job response during a new retry. The retry keeps fitting controls locked until the newly accepted job finishes or is cancelled. Independent design, software, adversarial and pricing-actuary reviews closed their findings. The designer reviewed the rendered desktop and mobile states; the earlier review had missed the cramped desktop recovery card. Python fitting and the 48-block installed-wheel checks above are the 8 October baseline, not new runs for this frontend correction. Nothing was pushed or released.

The prior build reported **1,605 passed, 1 skipped, 1 slow deselected** and **17 stack example Python blocks against an installed wheel**. Both totals are historical baselines for the redesign; they are not fresh validation of later changes. CI currently specifies Python 3.10–3.13 tests, Ruff, Black, mypy for `core`, `workflow`, and `pricing`, a Windows installed-wheel job, and frontend check/build plus existing Playwright suites. The source branch may have no CI run because it is unpushed.

Use the implementation checkout as the working directory (`/private/tmp/easyglm-model-variable-setup` locally, or the fresh clone on another machine). Create **new** environments and verify their import paths. From that checkout:

```bash
checkout=$(pwd -P)
check_dir=$(mktemp -d "${TMPDIR:-/tmp}/easyglm-check-XXXXXX")
python3.13 -m venv "$check_dir/source-env"
"$check_dir/source-env/bin/python" -m pip install -e '.[dev]'
"$check_dir/source-env/bin/python" -c 'import easy_glm; print(easy_glm.__file__)'
"$check_dir/source-env/bin/python" -m pytest -q
"$check_dir/source-env/bin/python" -m ruff check .
"$check_dir/source-env/bin/python" -m black --check .
"$check_dir/source-env/bin/python" -m mypy src/easy_glm/core src/easy_glm/workflow src/easy_glm/pricing --ignore-missing-imports
```

Use Python 3.13 for this local pass to match a CI interpreter. The printed import must resolve to this checkout. CI uses Node 22; this local visual-flow pass uses the available Node 26 (`node --version`). `frontend/playwright.config.js` starts the server through `../.venv/bin/python`; bind that path to the new environment **only if absent or a broken symlink**:

```bash
cd "$checkout"
if [ -L .venv ] && [ ! -e .venv ]; then unlink .venv; fi
if [ ! -e .venv ] && [ ! -L .venv ]; then ln -s "$check_dir/source-env" .venv; fi
.venv/bin/python -c 'import easy_glm; print(easy_glm.__file__)'
if [ -L frontend/node_modules ] && [ ! -e frontend/node_modules ]; then unlink frontend/node_modules; fi
```

If `.venv` is a real environment, preserve it and confirm its import path before browser tests; if it points elsewhere, use a clean clone rather than replacing it. Likewise inspect a real `frontend/node_modules` directory before `npm ci`, which replaces its contents. From `frontend/`, run `npm ci`, `npm run check`, `npm run build`, `node --test tests/*.test.js`, and `npx playwright install chromium` before the relevant config suites (for example, `npx playwright test --config playwright.models.config.js`). Those suites use bundled Chromium. The separate `tests/stack-visual.mjs` and `tests/stack-legacy-append.mjs` launch the Chrome channel and require an installed Chrome browser; `npx playwright install chrome` is an optional browser installation if it is absent. CI's workbench job also checks that rebuilt static assets match tracked files. `pytest -m slow` is the separate 5M-row memory budget.

The project model-plan and compatibility checks use these configurations from
`frontend/`:

```bash
npx playwright test --config playwright.model-plan.config.js
npx playwright test --config playwright.model-variables.config.js
npx playwright test --config playwright.model-variables-wide.config.js
npx playwright test --config playwright.legacy-models.config.js
```

The plan suite covers fitting order, single-model support and diagnostics
selection. Model variables covers local definitions and catalogue fitting. Wide
variables creates 500 columns and checks bulk JSON selection, reopening, search
and a narrow layout. The legacy suite retains saved-model recovery, retry,
diagnostic scopes and delayed-response protection. Run configurations sequentially;
some legacy suites share a server port.

For package acceptance, build and install a wheel into a **second new** environment. Keep this in the same shell as `check_dir` above:

```bash
cd "$checkout"
"$check_dir/source-env/bin/python" -m build --wheel --outdir "$check_dir/dist"
python3.13 -m venv "$check_dir/wheel-env"
"$check_dir/wheel-env/bin/python" -m pip install "$check_dir"/dist/easy_glm-*.whl
mkdir -p "$check_dir/run"
cd "$check_dir/run"
EASYGLM_STACK_DOC="$checkout/examples/stacked_models.md" "$check_dir/wheel-env/bin/python" - <<'PYBLOCKS'
import os
import re
from pathlib import Path
import numpy as np
import easy_glm

package = Path(easy_glm.__file__).resolve()
assert "site-packages" in package.parts, package
source = Path(os.environ["EASYGLM_STACK_DOC"]).read_text()
blocks = re.findall(r"(?ms)^```python\n(.*?)^```", source)
assert blocks, "No Python blocks found"
namespace = {"__name__": "__main__"}
for number, block in enumerate(blocks, 1):
    exec(compile(block, f"stacked_models.md block {number}", "exec"), namespace)
    print(f"Passed block {number}/{len(blocks)}")
assert np.allclose(
    namespace["complete"].predict(namespace["data"].head()),
    namespace["saved"].predict(namespace["data"].head()),
)
for name in ("vehicle_factors.csv", "occupation_factors.csv", "postcode_factors.parquet", "motor_model.easyglm"):
    assert Path(name).is_file(), name
PYBLOCKS
```

Do not skip display or export blocks. Inspect the exported full-catalogue keys and accepted multipliers, and run `examples/pricing_walkthrough.md` against the installed wheel as a regression scenario. The old reported counts are not a substitute for this fresh check.
The 30 September iCloud incident left recovery bundles/patches/manifests at `/Users/serban/Documents/Projects/easy_glm/.internal/recovery-20260930` and `/private/tmp/easyglm-recovery-20260930`. Tracked stack files and Git history survived. The old stack worktree directory is no longer present. The visual-flow worktree was created from its surviving Git branch, with dependencies in a fresh isolated environment. Rebuild dependencies without deleting preserved source or unrelated work. Some old worktrees are prunable; no pruning is required for this handover. These handover edits were created after the recovery bundle and are not contained in it.

## Transfer to another machine or provider

This stack branch is unpushed. Check that all required changes are committed before creating a transfer bundle. The
implementation checkout is a linked Git worktree: copying its directory or
`.git` pointer alone will not transfer the repository. For local continuation,
open `/private/tmp/easyglm-model-variable-setup` directly. For another machine, make a
packet containing a Git bundle **and today's seven guidance files**. This is a
local backup/transfer operation, not a commit, push or release:

```bash
source_checkout=/private/tmp/easyglm-model-variable-setup
packet=/Users/serban/Documents/Projects/easy_glm/.internal/handover-20261010
mkdir -p "$packet/docs/plans"
git -C "$source_checkout" bundle create "$packet/easyglm.bundle" refs/heads/codex/model-review-placement refs/tags/v0.472
git -C "$source_checkout" bundle verify "$packet/easyglm.bundle"
cp "$source_checkout/AGENTS.md" "$packet/AGENTS.md"
cp "$source_checkout/docs/HANDOVER.md" "$packet/docs/HANDOVER.md"
cp "$source_checkout/docs/MODEL_FLOW.md" "$packet/docs/MODEL_FLOW.md"
cp "$source_checkout/docs/FUTURE_RELEASES.md" "$packet/docs/FUTURE_RELEASES.md"
cp "$source_checkout/docs/PRICING_WORKFLOW_BUILD.md" "$packet/docs/PRICING_WORKFLOW_BUILD.md"
cp "$source_checkout/docs/plans/STACKED_MODELS.md" "$packet/docs/plans/STACKED_MODELS.md"
cp "$source_checkout/docs/plans/PROJECT_MODEL_WORKFLOW.md" "$packet/docs/plans/PROJECT_MODEL_WORKFLOW.md"
```

Check that the bundle contains the current `codex/model-review-placement` commit
(`git rev-parse codex/model-review-placement`) and the locally verified `v0.472` tag. Copy the packet to the other machine, clone the bundle into a
**new** directory with the stack branch selected, then copy the seven guidance
files into that clone and inspect `git status` before work. Keep the packet's
directory structure; its docs are newer than the 30 September recovery bundle.
For example, with transferred packet and destination paths chosen by the
operator:

```bash
git clone --branch codex/model-review-placement "$packet/easyglm.bundle" "$destination"
cp "$packet/AGENTS.md" "$destination/AGENTS.md"
cp "$packet/docs/HANDOVER.md" "$destination/docs/HANDOVER.md"
cp "$packet/docs/MODEL_FLOW.md" "$destination/docs/MODEL_FLOW.md"
cp "$packet/docs/FUTURE_RELEASES.md" "$destination/docs/FUTURE_RELEASES.md"
cp "$packet/docs/PRICING_WORKFLOW_BUILD.md" "$destination/docs/PRICING_WORKFLOW_BUILD.md"
mkdir -p "$destination/docs/plans"
cp "$packet/docs/plans/STACKED_MODELS.md" "$destination/docs/plans/STACKED_MODELS.md"
cp "$packet/docs/plans/PROJECT_MODEL_WORKFLOW.md" "$destination/docs/plans/PROJECT_MODEL_WORKFLOW.md"
git -C "$destination" status --short --branch
```

Set `packet` and `destination` to actual paths on the receiving machine first.
A bundle clone's `origin` points to that local bundle. Before checking live GitHub
state, run `git remote set-url origin https://github.com/serband/easy_glm.git`
inside the clone and then `git fetch origin` (read-only);
this does not authorize pushing, merging or release. Do not copy the raw
worktree, broken environment links, primary checkout's
unrelated files, private data, or `.git` pointer. Rebuild dependencies in fresh
environments as above. The bundle contains committed source and history; the
separately copied docs carry the current handover. If code changes after
this snapshot without a commit, package those explicitly or commit them to a
branch before relying on a Git bundle.

## Working and release rules

Follow `AGENTS.md` for architecture and checks. Use the public examples as acceptance scenarios, and keep conversation excerpts, temporary review notes, verification logs, and local recovery material out of Git. Do not edit the README unless explicitly requested. Do not merge to `main`, push a release tag, or publish without the owner's explicit authorization; `v*` tags trigger PyPI publishing. A branch, fitted model, preview, built wheel, or local version number is not a release.

When the stack's capabilities or status change, update this snapshot and the concise status in `docs/FUTURE_RELEASES.md`; keep `docs/PRICING_WORKFLOW_BUILD.md` clearly scoped to the earlier pricing workflow. Record durable API examples and reproducible tests in the normal repository, not a conversation transcript. Before handoff, distinguish what code does now from an accepted requirement and from a proposal, and state which checkout and test run supports each claim.
