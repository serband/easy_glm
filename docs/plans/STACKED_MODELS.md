# GLMs, CatBoost models and rating files

**Implemented and checked on `codex/multiplicative-model-stack`, awaiting user review. Not yet released.**

The runnable example is [Stacking models and building rating files](../../examples/stacked_models.md). This document records the agreed design; use the worked example for the implemented public calls.

**The core requirement is any ordering or repetition of compatible
multiplicative components.** Fit a model, keep it fixed, then fit the next
component against the complete prediction so far. GLMs, CatBoost models,
interaction tables and catalogue lookups follow the same stacking rules.
Vehicle, occupation and postcode files are examples of components, not fixed
positions in a prescribed workflow.

One acceptance workflow is:

1. Fit the main GLM and inspect A/E and relativities.
2. Add two CatBoost interactions, each deployed as its own existing pair table.
3. Fit a vehicle component against the GLM and both accepted interaction tables.
4. Score the complete vehicle catalogue; review and accept the vehicle file.
5. Fit occupation against that whole model, including the accepted vehicle
   file. Score and review the complete occupation catalogue.
6. Inspect the complete model, request holdout validation, and export the
   component files and combined scorer.

The same structure must support, for example:

- GLM → CatBoost → GLM → CatBoost.
- CatBoost → GLM → interaction table → vehicle component.
- Vehicle component → occupation component → GLM → postcode component.
- A single supported component used on its own.

There is no architecture rule limiting how many components of a type may
appear or requiring all GLMs, interactions or rating files to sit together.

## What a component means

A component is one fitted part of the model: a GLM, an interaction table,
a CatBoost model, or a vehicle/occupation/postcode lookup file. A named model
contains all components accepted up to that point.

| Named model | New component | Held fixed while fitting it |
| --- | --- | --- |
| Main | GLM | Any original external offset |
| With first interaction | Pair table 1 | Main |
| With second interaction | Pair table 2 | Main + pair table 1 |
| With vehicle | Vehicle component | Main + both pair tables |
| With occupation | Occupation component | Main + both pair tables + accepted vehicle file |

“Any order” means any supported component can be fitted next. Changing the
fitting order can change the result. Correlated factors share explanatory
power: fitting vehicle before occupation gives vehicle the first opportunity
to explain that part of the risk. Reordering requires new fits from the first
changed position onwards.

Any supported component may start the sequence. If it only learns a correction
(such as a pair interaction), fit a training-only intercept baseline internally
and build on that, including any declared external offset. Do not require the
user to build a separate GLM merely to satisfy a GLM-first implementation.
Save the baseline with the resulting model and reproduce it during validation.

Earlier named models remain available for comparison. An append, refit, band
change or catalogue replacement must never change them.

## Fitting and scoring rules

Multiplicative compatibility means a positive, finite starting prediction and
positive, finite correction multipliers on the same response scale:

```text
complete unit prediction = starting prediction × correction 1 × correction 2 × ...
```

Each newly fitted component supplies its incremental correction, not another
complete prediction to multiply in. Two independently fitted full predictions
cannot simply be multiplied together and called an offset fit. Check units,
target definition and exposure treatment before appending a component. Being
positive alone is not sufficient; for example, odds and unit claim frequencies
are not interchangeable.

One stack shares its target definition, family, link, Tweedie power where
relevant, weights, exposure convention and training/holdout split. Each
component chooses its own predictors without changing that response contract.

The proposed first version uses a **log link**: Poisson, Tweedie with fixed power
strictly between 1 and 2, or Gaussian. Gaussian observations may be negative;
its fitted means remain positive. Other family/link combinations stay available
where already supported by the package, but are not automatically supported in
the new mixed stack.

Each stage fits the original modelling response with the log of the complete
preceding unit prediction as its offset. Users do not create residual columns or offset
arrays. This is not an ordinary fit to actual minus expected, or to actual
divided by expected with unchanged weights.

Internally, each component returns only its own link-scale contribution:

```text
total log prediction = original external offset
                     + first component
                     + second component
                     + ...

unit prediction = exp(total log prediction)
expected amount = unit prediction × exposure, when the response requires it
```

External offset and exposure enter exactly once. The first component owns its
starting level. A later GLM may have a correction intercept, displayed as that
component's level adjustment; it must not rewrite an earlier intercept. The
proposed default is to allow it. A correction is not automatically neutral to
the portfolio total, even without an intercept.

A standalone CatBoost starts from a training-only intercept under the chosen
objective, conditional on any external offset. Save only its own starting
level; the cumulative scorer already includes the external offset. If the
data do not admit a usable positive baseline, explain the failure rather than
inventing one. Reuse/generalize the existing Gaussian/log objective explicitly;
ordinary RMSE on a raw CatBoost score is not equivalent.

CatBoost supports a per-row training baseline. EasyGLM must still own extracting
the correction and adding the preceding model once. See
[CatBoost's baseline documentation](https://catboost.ai/docs/en/concepts/python-reference_pool#baseline)
and [regression objectives](https://catboost.ai/docs/en/concepts/loss-functions-regression).

For an existing tariff, retain the current session's explicit log-offset
column convention. For example, this setup assumes `LogExistingFrequency`
already contains the log of an annualised technical frequency:

```python
work = PricingSession(
    data, family="poisson", claims="Claims", exposure="Exposure",
    offset="LogExistingFrequency", split="traintest",
)
```

The same baseline must be available when scoring new data. A stored historical
quote is not a reproducible tariff model. Do not treat a claim amount or a
commercial premium as a frequency, or include exposure in this baseline again.
Direct import of arbitrary external estimators is outside the first version.

## Use the accepted scorer for every later fit

| Fitted component | Accepted scorer | Input to later fits |
| --- | --- | --- |
| Existing two-variable CatBoost interaction | Binned pair table | The table, including amendments |
| Vehicle, occupation or postcode CatBoost | Complete catalogue lookup | The accepted lookup, including amendments |
| General CatBoost model | Saved CatBoost model | Its incremental contribution |

General CatBoost enables standalone models and CatBoost → GLM. Exact scoring
requires its model artifact; an unrestricted tree model cannot be exported as
independent GLM rating tables.

Catalogue scoring differs from approximating a pair surface. If every required
attribute is determined by the key, evaluate the component at each catalogue
row and retain that exact value. No ICE averaging or bin approximation is
needed. Caps, rounding or manual amendments are explicit subsequent changes.

A/E, comparisons, residual searches, later offsets and exports all use the
accepted scorer. Retain the fitted teacher for inspection where useful; never
silently substitute it for a deployed table.

## Turning a model into a rating file

A catalogue is the full list of keys and attributes to score, such as vehicle
codes with make, model, power and body type. It is separate from the experience
data used to estimate effects.

For a vehicle-only file, each vehicle code must determine every input used by
the vehicle component. Driver age cannot affect a vehicle-only factor. Vehicle
age at policy inception is not determined by a fixed model code either, unless
age becomes part of the rating key. Choose a larger key or keep that effect
elsewhere.

Export the **component multiplier**, not the full model prediction. Scoring the
vehicle catalogue needs no invented claims, exposure, driver age or occupation.
The experience-data scorer looks up that multiplier and combines it with the
preceding model.

Proposed first-version rules:

- One declared key per catalogue. Preserve string identifiers and leading zeros;
  validate any explicitly requested normalization.
- One row per key. Reject duplicates, ambiguous joins and missing required
  columns. Never multiply or drop experience rows during a join.
- Before fitting, join the declared catalogue attributes onto experience by
  the key, with explicit coverage checks. If experience already has those
  attributes, require agreement. Conflicting historical/current values block
  this one-key fit: use a versioned key, change the selected attributes or use
  a different component design. Do not silently train on one set of values
  and deploy another.
- Score every supplied catalogue key, including keys absent from training.
  Show training support and unseen attribute levels. An unseen code is not
  evidence that its prediction is reliable.
- Missing attribute values may use the fitted missing-value handling. Missing
  lookup keys are different: default to an error, with a user-selected fallback
  available and its usage counted. Never silently return 1.0.
- Save catalogue version, contents, key rules and final values with the named
  model. Replacing the source dataframe must not change an old model.
- Retain the matching catalogue attributes for diagnostics such as A/E by
  occupation group. Fitting and diagnostic enrichment use the same pinned
  catalogue and row-preserving join.
- Historical attributes must represent information available for that
  observation. Do not silently replace them with today's catalogue values.
  Learn outcome-derived attributes within folds. Time-varying attributes
  require a versioned key or a different component design; one-key exports
  cannot represent two values for one key.

No automatic normalization is proposed. Export the accepted multiplier.
If the modeller requests a reference of 1.0, retain any compensating level
inside the same component so complete predictions stay unchanged. Use a stated
training/reference portfolio, not equal weights across catalogue entries.

For the first version, a standalone multiplier column includes the component's
entire level as well as its shape. This also applies when the catalogue is the
first model. Display normalization must not leave an unexported scalar behind.
Metadata records the decomposition, but is not needed to reconstruct a missing
factor in the one-column rating file.

Label the units explicitly: a later file is an incremental multiplier. A
catalogue used as the entire base model, without an external baseline, carries
the unit prediction (for example, annualised frequency). Do not present that
base rate as a relativity to an unspecified separate model.

Retain scoring precision by default. If deployment requires rounding, apply it
to a reviewed lookup before fitting later components. Display precision alone
must not change scoring values.

## Proposed interactive calls

These are API proposals, not runnable examples of the released package.
Existing GLM/interaction calls remain familiar. Fitting from the session starts
a model; fitting from a named model appends a component against its complete
prediction.

```python
main = work.fit_glm("Main", factors=["DriverAge", "BonusMalus", "Region"])
first_pair = main.fit_interaction("DriverAge", "Region", name="First pair")
second_pair = first_pair.fit_interaction(
    "BonusMalus", "Region", name="Two pairs",
)
```

Proposed `fit_catalogue` fits the listed attributes, scores the supplied master
data and returns a model using the resulting lookup. `vehicles` and
`occupations` are user-supplied dataframes; these column names are illustrative.

```python
vehicle = second_pair.fit_catalogue(
    "Vehicle",
    catalogue=vehicles,
    key="VehicleCode",
    factors=["Make", "Model", "Power", "BodyType"],
    method="catboost",
)
vehicle.compare(second_pair, cv=True)
vehicle.component("Vehicle").show()
vehicle.ae("DriverAge").show()
```

Pause to review support, the lookup and full-model A/E. The proposed edit calls
below illustrate a cap and deployment rounding; these limits are examples,
not recommended vehicle rates. Applying produces a new named model.

```python
review = vehicle.edit_rates(name="Reviewed vehicle")
review.cap_component("Vehicle", lower=0.70, upper=1.50)
review.round_component("Vehicle", decimals=4)
review.preview().show()
accepted_vehicle = review.apply(refit_later_components=False)
```

Occupation is fitted against that reviewed lookup:

```python
occupation = accepted_vehicle.fit_catalogue(
    "Occupation",
    catalogue=occupations,
    key="OccupationCode",
    factors=["OccupationGroup", "PhysicalWork", "DrivingForWork"],
    method="catboost",
)
occupation.ae("OccupationGroup").show()
occupation.compare(accepted_vehicle, cv=True)
occupation.export_lookup("Vehicle", "vehicle_factors.csv")
occupation.export_lookup("Occupation", "occupation_factors.csv")
occupation.to_excel("rating_tables.xlsx")
occupation.save("motor_model.easyglm")
```

Before accepting the final model, inspect its A/E across the earlier main
factors, both interaction dimensions, vehicle and occupation, plus aggregate
A/E and a relevant cross-segment. Good aggregate A/E alone is insufficient.
All these checks use the accepted full scorer; request holdout explicitly.

The reverse order uses the same idea:

```python
base = work.fit_catboost("CatBoost base", factors=["DriverAge", "BonusMalus"])
corrected = base.fit_glm("GLM correction", factors=["Region"])
```

A catalogue fit also works directly from `work` as the first component.
An interaction can follow either model type. A repeated factor is identified
by component: its later relativity is an additional effect, not its total
effect across the model.

## Review and amendments

Show an ordered model list in the workbench: component name, type, predictors
or key, baseline and fit status. Selecting one opens its tables or diagnostics;
keep full-model A/E and comparison available. Variable selectors stay searchable
and alphabetical.

Table/lookup edits and component-level adjustments use Preview then Apply.
A direct CatBoost model has no editable relativity cells: change its fitting
settings and refit, or add an explicit supported correction. Arbitrary editing
of its trees is not part of this work. Show the component change and its impact on full
predictions over a stated portfolio. Capping a vehicle multiplier, capping a
total rate change, restoring the prior portfolio total and calibrating A/E to
1 are separate decisions.

Changing an earlier component preserves the old named model and offers:

- **Keep later components fixed:** create the amended scorer without new fits.
  Label later fit evidence as based on the old preceding model. This is useful
  for deliberate pricing amendments.
- **Refit later components:** rebuild in order against the amended scorer.
  Show that correlated predictors may partly counteract the amendment.

Neither choice silently drops later components or prior amendments.
Reordering/deleting a fitted component or changing its recipe marks affected
fits for review/refit. Distinguish a usable deliberately amended scorer from a
freshly fitted sequence.

## Validation and fitting time

Use the same training split and folds for a comparison. Within each fold,
rebuild every data-learned preceding component on that fold's training rows,
then construct its deployed tables/lookups. Fit the next component against that
complete scorer; evaluate it on the fold's held-out rows. A preceding model
fitted on validation outcomes must never supply the validation offset.

Automatic bins, categorical processing and learned catalogue attributes follow
the same rule. Fixed user choices such as explicit cuts and the selected recipe
stay fixed. To evaluate screening as part of a selection procedure, repeat it
inside the fold too.

Replay accepted caps, rounding and key/cell amendments on each fold's fitted
tables using their saved rules. Never freeze a full-training learned lookup
inside a validation fold. If an amendment cannot be mapped to a fold's bands
or levels, report CV unavailable with the reason; do not omit the edit. Edits
chosen after inspecting all training outcomes can be fixed in selected-recipe
tuning, but do not produce independent CV evidence. Compare the shorter and
extended models on the same rows and the same fold-specific preceding scorer.

Routine CV guides tuning of the chosen recipe. If components, predictors or
settings were chosen using those results, label them as tuning results. After
selection, refit on training data and evaluate on untouched holdout only when
requested. Nested validation of the entire selection process is an additional
option, not a requirement for every interactive fit.

Cache compatible fold-specific preceding fits. An independently fitted external
scorer may remain fixed across folds; saving a model fitted on the evaluation
outcomes does not make it independent. Use time/group/geographic splits to
answer the intended question. New policies on known vehicle codes and wholly
unseen codes are different validation questions.

Give CatBoost per-component trial/time limits. Do not time out the GLM. Show
prefix reconstruction, tuning, final refit and catalogue scoring separately.
Time spent fitting an upstream GLM must not consume the next CatBoost
component's tuning allowance.

## Implementation structure

Retain `PricingSession` and named `PricingModel` objects. Replace the internal
assumption of one GLM plus pair stages with an ordered list of component recipes
and immutable fitted artifacts.

Each artifact records its stable component ID, inputs, recipe, preceding-model
identity, fitted results, accepted scorer and evaluation evidence. Repeated
factor names in different components remain distinct. A cumulative scorer
owns response semantics and sums contributions once.

Reuse existing GLM tables, pair fitting/conversion, review tools and fold-specific
fitting. Generalize the GLM-first runner, pair-only configuration, single-fit
metadata and persistence; do not overload `pair_stages` with unrelated models.
Check compatibility through each component's scoring and fitting contract,
not a hard-coded list of allowed model-order combinations. Adding a supported
component type must not require redesigning which types may precede or follow it.

Use JSON for recipes and a versioned fitted bundle. Include direct CatBoost
assets where required. Lookup-only stacks score without CatBoost or the original
master-file source. Export a combined scoring entry point and separate component
files. A workbook may describe a direct CatBoost component but cannot reproduce
it as independent tables. Large lookups need CSV/Parquet and explicit workbook
size handling.

Load old projects as their existing GLM followed by separate pair tables, with
unchanged predictions and amendments. Version new formats, bump workbench
persisted-run format where needed, and preserve existing public calls.

## Build order and acceptance

1. **Shared stack/scorer:** adapt GLM/pair models. Prove old-model prediction
   parity and immutable checkpoints before adding another estimator.
2. **General CatBoost and GLM corrections:** fit either first or next, preserve
   response semantics, save/reload, and test each proposed family/link against
   the no-change correction.
3. **Catalogue components:** validate joins, fit, score the full master file,
   use the lookup in later fits, inspect/amend and export. Begin with one key
   and CatBoost; GLM catalogue fitting can follow after the shared contract.
4. **Interactive workflow:** component-aware review, diagnostics, progress and
   downstream decisions in Python/workbench. Agree delivery order; neither
   interface should require manual offset management.
5. **Examples and acceptance:** execute GLM → two pairs → vehicle → occupation
   with review stops, plus CatBoost → GLM. Check exported models in a fresh
   process and obtain separate actuarial and adversarial reviews.

Required checks:

| Check | What must hold |
| --- | --- |
| Append/refit | Preceding named models unchanged |
| GLM → CatBoost → GLM; CatBoost → pair → GLM | No hidden GLM-first requirement |
| Repeated/interleaved GLMs, CatBoost, pairs and lookups | Same offset contract regardless of position or type |
| Correction-only component first | Internal fitted baseline retained in scoring, folds and export |
| Incompatible units or invalid multipliers | Clear rejection before fitting/scoring, never a silently changed objective |
| External offset plus several components | Offset included once |
| Same risk with exposures 0.5 and 2 | Unit score unchanged; amount scales once |
| Amend pair/vehicle table before next fit | Next offset uses amendment, not teacher |
| Cap/round vehicle, then fit occupation | Uses reviewed lookup; earlier-factor full A/E checked again |
| Two GLMs use DriverAge | Contributions, edits and exports remain distinct |
| Change validation outcomes for fixed candidate | Training artifacts unchanged |
| Full catalogue includes unseen codes | Every supplied key scored once; support reported |
| Duplicate/blank/leading-zero/unknown keys | Declared matching/fallback; no row multiplication |
| Catalogue or deployment rounding changes | Old checkpoint stable; dependent fit evidence invalidated |
| Reorder/delete a component | Correct later fits rebuilt or explicitly retained for review |
| Save/reload/independent exported scoring | Full/component scores match accepted values |
| Catalogue first; reference changed to 1.0 | Export includes full component level, no missing scalar |
| Varying external offset and nonuniform weights | Standalone baseline and later offsets remain correct |
| Edit cannot map onto a fold's automatic bands | CV unavailable with reason; amendment never dropped |
| Poisson/Tweedie zeros; signed Gaussian target | Correct objective, finite supported predictions |
| Same key has conflicting attributes | One-key fit rejected before training; conflict identified |
| A/E by a catalogue attribute | Uses the pinned catalogue; no missing or duplicated experience rows |

Measure runtime, memory, export size and cancellation with representative wide
vehicle and large postcode catalogues. Report actual sizes/timings; do not assume
a specialist with many inputs will be as quick as a two-variable interaction.

Build work uses separate planning, software critique, actuarial review,
implementation and adversarial validation. Internal review notes stay out of
Git; public examples and reproducible tests provide durable evidence.

## Settings for each real project

- What response/family is the first project using: frequency, severity, claims
  cost or another target?
- Is each file defined solely by its key and attributes, or does the required
  key include another rating dimension?
- Agree actual key columns, attributes,
  catalogue versions, output schema/precision and fallback treatment. A small
  schema/example is sufficient to prepare the workflow.

The first version supports both Python and the workbench. Its limits are a linear sequence, the three log-link families
above, one-key catalogues and user-supplied master data. Multi-key tables,
arbitrary third-party model imports, automatic hierarchical credibility,
geographic smoothing and vendor-specific deployment adapters are follow-ups.
Stack exports include model bundles, Python, Excel and catalogue files. The
existing HTML report export is not yet available for mixed stacks.
