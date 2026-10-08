# Roadmap

This separates completed work from proposed work. A place on this list does not
mean a feature has been built or assigned a release version.

## Built, awaiting release

- CatBoost interactions for Gaussian models with a log link. Other Gaussian
  links remain unsupported for these interactions.
- Searchable, alphabetically sorted variable lists for Explore, fitted A/E
  plots and rate-table plots.

## Built, awaiting review: stack GLMs, CatBoost models and rating files

Implemented in Python and the workbench, with software, statistical and actuarial checks completed. See [the worked example](../examples/stacked_models.md) and [the agreed design](plans/STACKED_MODELS.md). Not yet released.

- Allow any ordering or repetition of supported multiplicative components:
  GLMs, CatBoost models, interaction tables and rating-file lookups. The vehicle
  then occupation sequence is an example, not a required order.
- Fit each new component against the complete preceding prediction, keeping
  earlier components fixed. A correction-only first component gets an internal
  fitted baseline without requiring a separate user-built GLM.
- Combine main effects, separate interaction tables, and vehicle, occupation
  or postcode components in the order chosen by the modeller.
- Fit every new component against the complete preceding model, using the
  tables and files actually used for scoring, including applied amendments.
- Score complete vehicle, occupation and postcode catalogues into lookup
  files. Use those same files for later fitting, diagnostics and deployment.
- Preserve named models. Make changing an earlier component explicit, with a
  choice to keep or refit later components.
- Provide short Python calls and a workbench view showing the fitting order.
  Both interfaces are included in this build.
- Select variables independently for each stage from the starting data, including
  unassigned columns, without changing global roles. Search lists alphabetically
  and retain selected variables while filtering.
- Show named models in a visual flow. Extend a complete fitted model or create
  an alternative from the same parent, keeping earlier fits unchanged. Preserve
  those connections in project JSON and label historical parents explicitly.
- Keep the selected model's variables, settings and Fit action together. Save
  and fit in one click, reopen saved drafts, and revise fitted models as new named
  versions. Keep the model map available behind a disclosure.
- Test both GLM → CatBoost and CatBoost → GLM, full-stack validation, lookup
  coverage, save/reload and export parity.

This replaces the earlier staged-recipe and specialist-CatBoost proposals.
Jointly fitting groups of interactions is no longer the proposed next step:
existing pair interactions remain separately fitted and inspectable.
No v0.5 commitment is implied.

## Later modelling work

- **Per-stage numeric designs:** consider separate band overrides for GLM and
  interaction-table stages. The current flow has independent variable selections
  and fitting settings; numeric bands remain shared on Variables.
- **Hierarchical rating factors:** let sparse vehicle variants borrow strength
  from model/make, occupations from occupation groups, and postcodes from larger
  areas. Show support and fallback rules; export leaf-level lookup files
  alongside component detail.
- **Postcode smoothing:** assess geographic smoothing and local residual
  effects, with validation suited to new areas and future periods.
- **Tweedie power selection:** investigate an optional training-only
  profile-likelihood search. Keep manual selection and show the powers and
  comparison before accepting a choice.
- **Model selection:** assess the one-standard-error rule for the GLM penalty
  and a raw linear trend alongside binned step effects. Neither is an accepted
  statistical change yet.
- **Residual searches:** review family-specific variance and dispersion in
  Gamma/Tweedie and Poisson searches. Test ranking and calibration on known
  examples before changing the method.

## Further worked examples

The [interactive frequency walkthrough](../examples/pricing_walkthrough.md)
ships in v0.472. Extend that style with:

- Gamma severity, an expanded Swedish motorcycle Tweedie workflow,
  existing-premium offsets, and a genuine yes/no binomial example.
- Wide datasets, predictor screening, step versus linear effects, monotone
  constraints, and comparing candidate models.
- Extend the [stacking example](../examples/stacked_models.md) with real catalogue data.

Use short public calls, explicit modelling decisions and reviewable outputs.
Run each example from a fresh standard installation and check that exported
scoring reproduces the accepted model on new data.

## Consistency checks to scope

These are audits of remaining gaps, rather than claims that the whole feature
is missing:

- Existing split-column detection when launching the workbench from a
  dataframe; keep training and holdout labels clear. Python sessions already
  support an existing split and stable splits by ID.
- Holdout visibility and grouped/time-based validation across interfaces.
  Python's interactive workflow already keeps holdout evaluation explicit.
- Exported Python setup: check final roles, renames, recodes, derived columns
  and split settings together, rather than adding another export format.
- Leakage-review candidates: make absent, unassigned and ignored roles
  consistent with Variables; preserve reserved-column exclusions and make any
  inclusion of unassigned columns explicit.
- Invalid evaluation targets for Gamma and inverse Gaussian: report affected
  rows and agreed treatment without silently dropping them from a headline
  metric. Keep family/link explanations consistent across examples.

## Recently completed

See [the changelog](../CHANGELOG.md) for released details.

- **v0.472:** interactive pricing workflow; family-specific metrics; sampled
  importance; categorical/mixed associations; Gaussian screening fix;
  unassigned interaction variables; per-interaction tuning limits; Windows
  progress-update fix; CatBoost and Optuna in the standard installation.
- **v0.470–0.471:** one-way feature selection, separately fitted CatBoost
  interaction tables, automatic tuning, full-table diagnostics and exports,
  and removal of feature-selection temporary files.
- **v0.464:** shared numeric bin settings, custom cuts and Variables JSON.

Older entries marked only “implemented on the feature branch” have been removed
from the future-work list. Current behaviour and the changelog take precedence
over those old status notes.
