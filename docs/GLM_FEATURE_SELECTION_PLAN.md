# GLM feature selection

Status: implemented and independently reviewed; local verification below.
Branch: `codex/glm-feature-selection`, based on v0.464.

## User workflow

An optional **Feature selection** panel on Variables, after roles, binning and
the training split have been specified. Candidates are predictors and unassigned
source columns. Generated derived-only columns are outside this screen because
they are not editable in the Variables role table. Target, weight, exposure,
offset, current premium, split, time, ID and
ignored fields are excluded. Display names may change; results retain source
column identifiers so they can update the correct table and JSON entries.

The screen uses the Variables draft through its existing validation path. It
must work before a model exists, so family, link, target/weight division and
regularisation settings must be explicit. Target, weight and offset come from
the role assignments. Existing custom cuts and default/per-variable bin counts
apply to the screening GLM.

## Approved fit topology

Fit **one candidate at a time**: each GLM contains one real candidate, its four
shuffled copies and one independent random variable. The user explicitly chose
this to screen for a lack of one-way effect. Correlated candidates may all pass;
conditional importance and a minimal deployable model remain later steps.

Separate planning and statistical critique are complete. The initial topology
question is resolved. This is a marginal screening module, not joint Boruta.

## Statistical contract

- Apply the canonical data preparation and split, then isolate training rows
  before discovering bins/levels, generating controls, fitting or scoring.
  Holdout outcomes never enter selection.
- Generate four independently shuffled training-column copies per candidate,
  including missing values. Use reproducible seeded randomness and temporary
  names that cannot collide with source columns.
- Clone the real variable's encoder for its shadows: identical cuts, levels,
  null handling, penalty weights and monotonic restrictions. Recomputing
  weighted categorical levels after shuffling would make the comparison unfair.
- The extra random predictor is independent seeded numeric noise with the
  project's default numeric bin count; explain this definition in the results.
- Use main effects only. Force five-fold cross-validation to choose alpha over
  the configured penalty path, then fit on all training rows. This selects the
  best tested penalty, not a claim of the best possible GLM.
- Calculate fixed-model permutation importance on training rows using mean
  weighted deviance increase, preserving the target/weight/offset convention.
  Report repeated-permutation mean and standard deviation, including negatives.
- Report the real importance, four shadow scores, random score, comparison
  threshold and margin. The threshold is the maximum of zero, that candidate's
  four shadow importances and its independent noise importance. The candidate
  must exceed it by a small numerical tolerance; ties and zero importance do
  not pass. No p-values or automatic significance claims.
- This is a Boruta-style training screening heuristic, not a reproduction of
  Boruta's iterative statistical acceptance/rejection procedure. In-sample
  importance and correlated-predictor limitations must be clear.
  Reference: [Boruta method and terminology](https://www.jstatsoft.org/article/view/v036i11).
  Bins and controls are prepared on the full training set; CV tunes the penalty,
  rather than providing an independent performance estimate for this screen.

## Implementation boundaries

- New `workflow/feature_selection.py`: GUI-independent algorithm, preparation,
  controls, fitting, importance and structured results. Reuse design/fit/loss
  primitives; do not change existing encoder or fitting semantics.
- New desktop request/job layer: immutable inputs, cancellable subprocess,
  bounded result storage, progress, generation/revision/fingerprint checks,
  descriptive failure messages and stale-result suppression.
- New Svelte panel: optional settings, candidate counts, Run/Cancel, searchable
  result table, real/control comparison and explicit review of role changes.
  Label results Signal detected, No signal detected, Skipped or Failed. Failed
  and skipped candidates never become no-signal recommendations. The user can
  select rows and stage Ignore or Make predictor actions through the existing
  Variables preview/apply flow; completing a scan changes no roles, model
  definitions or fitted runs.
- Preflight the expanded design cost. No silent candidate truncation or row
  sampling. Any sampling option must be explicit and training-only.
- No README edits, version bump, push or publication in this feature task.

## Acceptance checks

1. Changing holdout values cannot change any controls, bins, selected penalty,
   importance or selection result for a fixed training set and seed.
2. Candidates and operational-column exclusions are correct after renames,
   recodes, filters and type overrides.
3. Four deterministic shadows preserve each candidate's training values and
   dtype; encoder definitions match exactly apart from column name.
4. Weight, exposure convention, target division, offset, family/link and
   five-fold CV settings reach the fit and permutation loss correctly.
5. Strong planted signal can beat controls; ties, negatives, constants,
   all-null columns and unusable inputs have explicit outcomes.
6. The project is unchanged until reviewed application. Table and JSON stay
   synchronised, and existing model invalidation/cleanup handles role changes.
7. Cancellation terminates work; changed drafts/projects suppress late results;
   errors and excessive designs produce actionable messages.
8. Separate implementation and independent review agents, followed by focused
   Python/API/browser checks and a real preview of the complete workflow.

## Review record

Separate agents planned and critiqued the design, implemented the algorithm,
desktop jobs and UI, and reviewed the implementation. Review fixes cover source
column identity, cancellation, input validation and unsuccessful fits. Failed
or unconverged fits never produce a no-signal recommendation. The final
independent review found no remaining correctness blocker.

Focused workflow/API checks cover actual fits across all five supported
families, training-only decisions, custom cuts, encoder cloning, renames,
cancelled jobs, stale results and unchanged project settings.

Verification: 44 focused workflow/API tests, 18 browser tests covering feature
selection and existing Variables/binning/screening, and 14 JavaScript unit tests
passed. The broader Python run had 1,369 passes and nine socket-permission
failures; all nine passed when the launcher tests were rerun with local-server
access. The subsequently added four family smoke tests also passed. Black,
Ruff, mypy, Svelte check and the frontend build passed. A synthetic-data
preview is available on port 8826; no release was created.
