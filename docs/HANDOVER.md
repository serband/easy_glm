# EasyGLM handover

**Snapshot: 10 October 2026.** Start here when taking over without the previous chat. Check Git and the current package registry before changing a status claim: this is a local snapshot, not proof of today's remote state.

## Start in the right checkout

| Purpose | Checkout and state |
| --- | --- |
| Current implementation | `/private/tmp/easyglm-model-variable-setup`, `codex/stack-diagnostics-scope`, based on per-model variable setup revision `ce6f54e`; includes per-model definitions and selectable cumulative diagnostic fits |
| User's primary workspace | `/Users/serban/Documents/Projects/easy_glm`, `codex/variable-screening`, `cb66d3648d47ebf62f39c0461daff6d2e18dbc69`; contains unrelated modified and untracked work |
| Last known public release | `v0.472`, locally verified tag at `132e1e14433f849517fc225d2efea5b76209ef23` (23 September 2026) |

The current stack is **local branch work, unpushed and unreleased**. Its `pyproject.toml` still says `0.472`; that version string does not make the stack part of the public package. The primary workspace's `main` is older (`4eb8c8a`) and must not be used as a release baseline. Remote GitHub/PyPI status was not live-verified for this snapshot because network name resolution was unavailable. Do not reset, clean, switch, or prune the primary checkout. Inspect `git status`, branch, commit, and file paths before acting.

On 10 October the previous `/private/tmp/easyglm-visual-model-flow` directory was
absent. This checkout was created from its surviving committed branch. Earlier
uncommitted changes must not be assumed recovered. The current model form has one
name, a Method selector and Save/Fit after variable selection; these are covered
by the current browser tests.

The primary workspace carries a copy of this handover and selected guidance, but its code does **not** contain the current stack. Read `examples/stacked_models.md`, `docs/plans/STACKED_MODELS.md`, `docs/LLM_USAGE_GUIDE.md`, and the source/tests **in the implementation checkout**; the example and usage guide are absent from the primary workspace, while its untracked copy of the stack plan may differ from the implementation checkout. `examples/pricing_walkthrough.md` covers the earlier released pricing flow. The usage guide describes v0.472, not takeover. The locally available `v0.472` Git tag provides the release source for comparison (`git show v0.472:<path>` from the stack checkout); verify remote state separately when network access returns.

## What the package does

EasyGLM is an insurance-pricing Python package. The released v0.472 package fits regularised GLMs, turns their coefficients into exact rating tables, scores from those tables, and includes a local browser workbench and command-line fit/export/validate commands. The simple API is `EasyGLM.fit`; the interactive actuarial API is `PricingSession` → named `PricingModel` checkpoints; lower-level users can use `DesignSpec` → `fit_glm` → `rate_tables`/`to_rate_model` and `RateModel` for portable lookup scoring. The v0.472 interactive path also supports separately fitted two-factor CatBoost interaction tables, predictor screening, training/CV diagnostics, explicit holdout inspection, rate amendments, and save/reopen. Consult the release tag and `CHANGELOG.md` for exact released details.

The existing core accepts Poisson, Gamma, Gaussian, inverse Gaussian, Tweedie and binomial families, with log links by default except binomial's logit; compatibility and target constraints vary by feature. **Mixed stacks support only Poisson, Tweedie and Gaussian with log links.** The default browser workbench is the local FastAPI/Svelte app (`python -m easy_glm.desktop`, or `python -m easy_glm.app`); the older Streamlit UI is available through `python -m easy_glm.app --legacy-streamlit`. `easy-glm run|export|validate|workbench` is the project CLI. The standalone `.easyglm` scorer, project JSON, and fitted stack bundle serve different purposes; check which artifact a command expects.

## What exists and what remains

The stack branch implements an ordered multiplicative sequence of GLM, direct CatBoost, separately accepted two-factor interaction tables, and complete catalogue lookups. A component fits the original response with the log of the **complete accepted predecessor scorer** as offset. The scorer includes earlier amended tables/files, the declared external offset once, and exposure once when converting unit predictions to expected amounts. A correction-only first component gets an internal fitted baseline. Supported mixed-stack families are Poisson, Tweedie, and Gaussian with log link; do not assume other existing package families/links work in a mixed stack.

`PricingSession` and named `PricingModel` objects provide short public calls (`fit_glm`, `fit_catboost`, `fit_interaction`, `fit_catalogue`, `ae`, `compare`, `edit_rates`, `export_lookup`, `save`/`load`). The workbench has a stacked-model path and component inspection, comparison, amendment, and export views. Named fits are detached; append and amendment create new models. The Python example demonstrates GLM → two pair tables → vehicle → reviewed vehicle → occupation → postcode, then a separate CatBoost → GLM candidate. Catalogue fitting pins the full supplied key/attribute table, scores keys without training rows, checks joins/coverage, and exports an **incremental component** file. A catalogue-first file is a full unit prediction only when there is no external baseline; otherwise the external offset remains separate in the complete scorer. Missing keys error by default unless an explicit fallback is chosen.

**Current product work: the selected-model workspace.** The staged workbench has a compact model list and one workspace containing the selected model's factors, settings and fitting controls. **Fit model** saves the exact draft and starts fitting in one action; **Save draft** does not fit. Saved drafts reopen for editing. Fitted models open on Results, with read-only Settings and explicit **Add model after this** or **Revise model** actions. Revisions have new names and keep the same fitted parent. The model map is optional. Dirty drafts require Keep editing, Save draft or Discard changes when leaving them.

Each component selects its own factors from safe predictor and unassigned source columns. Its **Variables for this model** editor provides searchable alphabetical Table and JSON views. Global roles identify response, weight, ID, split and ignore; those controls cannot be changed in this editor. GLMs and interaction tables now store their own defaults, types and bands in optional `ComponentRecipe.design` snapshots. Direct CatBoost uses raw values and local categorical/numeric treatment. Legacy recipes without this field retain their previous project-design behaviour and serialize without the new field, preserving existing fingerprints. A child's target, weight, family, log link and split remain inherited; its offset is the complete fitted predecessor scorer. Model-save validation rejects changes to the response convention or frozen prefix and refuses edits to global bands through a stack-model save.

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

A child's own complete fit can still be used as the next offset when its original incoming parent is unavailable; that missing parent blocks revision of just the last addition. If its own fit is missing, rebuilding creates a new named copy of the whole saved recipe. Fitted stack settings and fit endpoints reject in-place overwrite. Existing single-model workflows remain available. See `docs/MODEL_FLOW.md` for the user workflow.

The 9 October correction separates ordinary drafts from saved setups that need recovery. **Needs a new fit** means the saved setup's exact earlier starting fit is unavailable; a currently fitted model with the same name is not a replacement for it. **Fit saved recipe as new model** fits the saved sequence from the beginning under a new name. **Cannot score** means the selected model's own fitted files are unavailable. A usable complete scorer remains **Fitted** even when its original starting fit is missing. The header shows identity, context and actions in separate rows; a read-only saved recipe shows the fitting order.

`ModelConfig.lineage` records the parent name, fit identity, settings key and prefix fingerprint. It is provenance, excluded from fit/cache hashes; the complete ordered component recipe remains the scorer. Projects with lineage write schema version 5; older projects without lineage retain their prior serialized version. An appended candidate requires an exact applicable parent artifact and unchanged frozen prefix. Changing a parent does not mutate its existing children. Missing or changed artifacts produce historical links; an explicit full rebuild fits the whole saved recipe afresh and remains marked historical after job restore. Scope validation runs on save and fit, including imported projects, and protects response/control fields against catalogue-name collisions.

The Model builder work is local, unpushed and unreleased. Fresh validation for this redesign is recorded below; the earlier visual-flow results are a separate baseline. Keep discovery on training data, label in-sample and CV evidence accurately, and display holdout results only on explicit request. Table changes need preview then apply and a clear choice to retain or refit later components. A/E and searches must use the accepted complete scorer; earlier named fits must remain unchanged.

Known first-version boundaries: each named model scores a linear sequence; the diagram can show alternatives but does not combine two parent models; one-key, user-supplied catalogues; no arbitrary imported estimators, automatic hierarchical credibility, geographic smoothing, or mixed-stack HTML report. A direct CatBoost component requires its saved model artifact to score and cannot be represented exactly by GLM rate tables alone. Confirm any further limit in code before asserting it.

## Source map and checks

- `src/easy_glm/pricing/session.py`, `model.py`, `review.py`, `views.py`, `validation.py`, `persistence.py`, `excel.py`: public interactive setup, named checkpoints, edits, diagnostics, comparisons, save/load and Excel.
- `src/easy_glm/workflow/stack_types.py`, `stack.py`, `stack_tuning.py`, `catalogue.py`: component recipes/artifacts, cumulative scoring and offsets, fold-aware tuning, catalogue validation and lookup.
- `src/easy_glm/workflow/project.py`: global data roles/design, model configuration and JSON migration. `src/easy_glm/desktop/` plus `frontend/src/StackWorkbench.svelte` and `frontend/src/App.svelte`: workbench APIs and screens.
- `tests/test_pricing_stack_api.py`, `test_stack_tuning.py`, `test_stack_exports.py`, `test_desktop_stack.py`: targeted stack tests; `frontend/tests/stack-visual.mjs` and `stack-legacy-append.mjs` are local browser review scripts, not a dedicated `playwright.stack.config.js` CI job.

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

The per-model definition checks use these two configurations from `frontend/`:

```bash
npx playwright test --config playwright.model-variables.config.js
npx playwright test --config playwright.model-variables-wide.config.js
```

The first covers root/child fitting, catalogue attributes, draft recovery and
diagnostic fit selection with delayed-response protection. The
second creates 500 available columns and checks bulk JSON selection, saved draft
reopening, search and a narrow layout without horizontal table scrolling.

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
packet containing a Git bundle **and today's six guidance files**. This is a
local backup/transfer operation, not a commit, push or release:

```bash
source_checkout=/private/tmp/easyglm-model-variable-setup
packet=/Users/serban/Documents/Projects/easy_glm/.internal/handover-20261010
mkdir -p "$packet/docs/plans"
git -C "$source_checkout" bundle create "$packet/easyglm.bundle" refs/heads/codex/stack-diagnostics-scope refs/tags/v0.472
git -C "$source_checkout" bundle verify "$packet/easyglm.bundle"
cp "$source_checkout/AGENTS.md" "$packet/AGENTS.md"
cp "$source_checkout/docs/HANDOVER.md" "$packet/docs/HANDOVER.md"
cp "$source_checkout/docs/MODEL_FLOW.md" "$packet/docs/MODEL_FLOW.md"
cp "$source_checkout/docs/FUTURE_RELEASES.md" "$packet/docs/FUTURE_RELEASES.md"
cp "$source_checkout/docs/PRICING_WORKFLOW_BUILD.md" "$packet/docs/PRICING_WORKFLOW_BUILD.md"
cp "$source_checkout/docs/plans/STACKED_MODELS.md" "$packet/docs/plans/STACKED_MODELS.md"
```

Check that the bundle contains the current `codex/stack-diagnostics-scope` commit
(`git rev-parse codex/stack-diagnostics-scope`) and the locally verified `v0.472` tag. Copy the packet to the other machine, clone the bundle into a
**new** directory with the stack branch selected, then copy the six guidance
files into that clone and inspect `git status` before work. Keep the packet's
directory structure; its docs are newer than the 30 September recovery bundle.
For example, with transferred packet and destination paths chosen by the
operator:

```bash
git clone --branch codex/stack-diagnostics-scope "$packet/easyglm.bundle" "$destination"
cp "$packet/AGENTS.md" "$destination/AGENTS.md"
cp "$packet/docs/HANDOVER.md" "$destination/docs/HANDOVER.md"
cp "$packet/docs/MODEL_FLOW.md" "$destination/docs/MODEL_FLOW.md"
cp "$packet/docs/FUTURE_RELEASES.md" "$destination/docs/FUTURE_RELEASES.md"
cp "$packet/docs/PRICING_WORKFLOW_BUILD.md" "$destination/docs/PRICING_WORKFLOW_BUILD.md"
mkdir -p "$destination/docs/plans"
cp "$packet/docs/plans/STACKED_MODELS.md" "$destination/docs/plans/STACKED_MODELS.md"
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
