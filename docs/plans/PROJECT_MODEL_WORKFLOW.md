# Project model plan

The workbench defines the modelling process before choosing variables or fitting.
One starting dataset supplies every model. Later models fit the same response
with the complete preceding score as their offset.

## Screens

| Screen | Action |
| --- | --- |
| Project & data | Open data; choose one model or models in stages; assign shared column roles; define each model using searchable Table and JSON editors. |
| Model | Set fitting options and fit the next model in order. |
| Diagnostics | Choose a fitted version, then inspect the complete final fit or the cumulative fit through an earlier model. |

The project name identifies the work. Each model has one name and one method.
There is no separate component-name field. The choice between one model and
models in stages appears only in Project & data.

For a single model, choose its variables, then its model details and variable
treatment. For models in stages, define the shared response convention and the
named methods in order, then choose variables and treatment separately for each
stage. There is no separate Variables sidebar page.

Each model has a numbered card with **Model name**, **Model type**, its incoming
offset and its own **Choose variables** action. **Add model** appends a new GLM
card and focuses its type without opening the variables editor. There is no
separate type selector for a model that has not yet been added. Later cards name
the complete preceding sequence they offset. An adopted starter displays its
exact saved name and **Saved fitted model**, with its accepted scorer locked.

The response, weight/exposure, split and source data are shared. They cannot be
overridden for a downstream model. Each model has its own selected variables,
numeric bands or raw CatBoost types. IDs, ignored columns and shared control
columns cannot become model factors.

## Plan and fitted versions

The current plan is editable setup. Fitted versions are saved results and remain
unchanged when the plan changes.

- Change model structure opens the route chooser. Choosing a route and confirming
  replaces the working setup and selection; Keep current plan cancels unchanged.
  Saved fitted versions remain available in history.
- Add a model appends to the current plan. It does not require all future models
  to be declared before fitting the first model.
- Changes to an unfitted model do not invalidate earlier fitted models.
- Changing a fitted model requires an explicit revised plan. The changed model
  and every model after it need fitting again; the earlier fitted version remains
  available. Reuse of an unchanged fitted prefix must use its exact saved fit.
- Existing projects retain their saved models. Adopting a saved chain into a plan
  is explicit; loading a project must not pick a branch arbitrarily. The adopted
  scorer is a frozen first stage, including its accepted amendments. It can be
  extended; replacing it requires a new independent plan.
- After fitting, shared column assignments are locked. Revise shared assignments
  copies the plan before unlocking them. Applying changes requires a new root fit
  and retains the previous saved recipes and results.

Fitting saves the exact current options and starts the next eligible model in one
action. Model 3 cannot be fitted before models 1 and 2. Failures and cancelled
jobs leave a retryable plan. A late response from an earlier job cannot replace a
newer plan or selection.

## Backend contract

An optional `Project.model_plan` stores the current ordered setup. The existing
`Project.models`, fitted artifacts and lineage retain immutable fit history.
Legacy projects without a plan retain their serialization and fitting hashes.

`GET /api/model-plan` returns the plan, stage status, safe variables, fitted
references and next eligible stage. Atomic revision-checked saves create, revise
or adopt a plan. `POST /api/model-plan/fit-next` derives the starting score on the
server; callers cannot supply a substitute parent or downstream response.

Plans use project schema version 6. Plan IDs and fit references are issued by
the server and persisted with the setup. A reference is applicable only when its
artifact and exact materialised configuration still match. A renamed or revised
plan cannot silently bind another historical fit. Catalogue sources must be
validated, pinned uploads; their attributes become available at their join. A
catalogue can be declared before uploading it. Its upload and key are required
when fitting that stage, not when saving the initial model plan. Reopening
project JSON preserves its reference metadata; a fit is usable only when its
matching artifact is also present.

The existing fit/scoring engine stays in charge of cumulative offsets. A single
GLM keeps its existing family and link support. Mixed stages keep the current
supported log-link families. CatBoost defaults to training-only Optuna tuning;
holdout is used only for explicit evaluation.

## Acceptance

Walk through a new GLM → vehicle CatBoost → occupation CatBoost plan with different
variables for every model. Then append a model after fitting, revise an upstream
model, reopen the plan, start a new plan and inspect the retained fitted version.
Check Table/JSON selection with 500 variables and a narrow browser window.

Verify exact predecessor offsets, immutable earlier predictions, protected shared
controls, legacy import, stale requests, cancellation/retry, catalogue scoring,
diagnostic scopes and exports. Independent design and pricing-actuary review must
exercise the rendered workflow as well as the software tests.
