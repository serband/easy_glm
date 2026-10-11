# Model variations from one pricing book

Use this page when you want to compare different **orders** of model components.
Each named result is a separate checkpoint. A later component fits the original
response with the complete earlier score as its offset, so `GLM → CatBoost` and
`CatBoost → GLM` are not the same allocation of effect.

> **Scope.** Mixed stacks are current local work and are not in published v0.472.
> See [workflows.md](workflows.md) for the workflow hub. The packaged motor data
> is synthetic and the examples demonstrate API choices, not an approved tariff.

All fitted and searched models below use only the supplied training rows. Holdout
is not displayed unless you explicitly request it.

## Load one reproducible book

```python
import polars as pl

from easy_glm import PricingSession, load_motor_catalogue_example

data, _, _, _ = load_motor_catalogue_example(rows=900, seed=73)
```

## Standalone CatBoost with a trial budget

Set `tune=True` when you want Optuna to choose settings on training folds.
`tuning_budget="trials"` means the cap is the number of candidate trials, rather
than elapsed fitting time. The final selected tree is then fitted on all training
rows; holdout rows are not used for either step.

```python
tree_session = PricingSession(
    data,
    family="poisson",
    claims="Claims",
    exposure="Exposure",
    id="PolicyID",
    split="traintest",
)
tree_only = tree_session.fit_catboost(
    "Tuned tree model",
    factors=["DriverAge", "BonusMalus", "Region"],
    tune=True,
    tuning_budget="trials",
    trials=2,
    cv=3,
    seed=73,
    thread_count=2,
)
tree_only.summary()
tree_only.ae("DriverAge").show()
```

## GLM followed by a manual CatBoost correction

This is useful when you want a stable, inspectable main-effects GLM and a
separate flexible correction. `tune=False` is explicit: the supplied tree
settings are fitted once, with no Optuna search or fold selection.

```python
glm_first_session = PricingSession(
    data,
    family="poisson",
    claims="Claims",
    exposure="Exposure",
    id="PolicyID",
    split="traintest",
)
glm_first = glm_first_session.fit_glm(
    "GLM main effects",
    factors=["DriverAge", "Region"],
    cv=3,
    n_alphas=8,
)
glm_then_tree = glm_first.fit_catboost(
    "GLM then manual tree",
    factors=["BonusMalus"],
    tune=False,
    iterations=40,
    depth=3,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    seed=73,
    thread_count=2,
)
glm_then_tree.summary()
glm_then_tree.ae("BonusMalus").show()
```

The CatBoost component is a correction to the GLM score. Its contribution is not
a replacement for the GLM's driver-age or region rate tables.

## CatBoost followed by a GLM correction

A time budget is the other explicit tuning policy. `tuning_budget="time"` counts
CatBoost fitting time over the training folds. A candidate already running may
finish after the limit; preparation, scoring and the final full-training fit do
not count against it.

```python
tree_first_session = PricingSession(
    data,
    family="poisson",
    claims="Claims",
    exposure="Exposure",
    id="PolicyID",
    split="traintest",
)
tree_first = tree_first_session.fit_catboost(
    "Timed tree base",
    factors=["BonusMalus", "Region"],
    tune=True,
    tuning_budget="time",
    time_limit_minutes=0.05,
    cv=3,
    seed=73,
    thread_count=2,
)
tree_then_glm = tree_first.fit_glm(
    "Tree then age GLM",
    factors=["DriverAge"],
    cv=3,
    n_alphas=8,
)
tree_then_glm.summary()
tree_then_glm.ae("DriverAge").show()
```

Compare only models built from the same response and split. `cv=True` compares
them on shared training folds; request holdout separately when you are ready.

```python
tree_then_glm.compare(tree_only, cv=True)
tree_then_glm.compare(tree_only, subset="holdout")
```

## Gaussian with a log link

Mixed stacks support Gaussian models with a log link. This small example uses a
strictly positive annualised claims proxy to demonstrate the response convention;
it is **not** a claim-severity analysis. For severity work, use a defensible
severity target and review its distribution and weights first.

```python
gaussian_data = data.with_columns(
    (pl.col("Claims") / pl.col("Exposure") + 0.01).alias("AnnualisedClaims")
)
gaussian_session = PricingSession(
    gaussian_data,
    family="gaussian",
    target="AnnualisedClaims",
    weight="Exposure",
    id="PolicyID",
    split="traintest",
    link="log",
)
gaussian_log = gaussian_session.fit_glm(
    "Gaussian log main effects",
    factors=["DriverAge", "Region"],
    cv=3,
    n_alphas=8,
)
gaussian_pair = gaussian_log.fit_interaction(
    "DriverAge",
    "Region",
    name="Gaussian age and region",
    trials=2,
    prefix_trials=1,
)
gaussian_complete = gaussian_pair.fit_catboost(
    "Gaussian bonus-malus correction",
    factors=["BonusMalus"],
    tune=False,
    iterations=12,
    depth=3,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    seed=73,
    thread_count=2,
)
gaussian_complete.summary()
gaussian_complete.ae("DriverAge").show()
```

The interaction is fitted after the Gaussian main effects and the manual
CatBoost is then fitted from that complete score. It therefore demonstrates the
same ordered-offset rule under a supported Gaussian log response. Gamma and
binomial GLMs need response definitions appropriate to their outcome; see
[model_families.md](model_families.md) for real-data examples rather than choosing
a family merely to make an API example run.

## Revise a fitted table without overwriting it

A rate review creates another named checkpoint. This one-row revision is only a
mechanical example; inspect the preview before changing a real portfolio.

```python
main_rate_row = glm_then_tree.component("GLM main effects").show().table.row(
    0,
    named=True,
)
review = glm_then_tree.edit_rates("GLM and tree revised")
review.set_relativity(
    "DriverAge",
    level=main_rate_row["label"],
    value=1.05,
    component="GLM main effects",
)
review.preview().show()
fixed_suffix = review.apply(refit_later_components=False)
fixed_suffix.summary()
```

`fixed_suffix` preserves the later tree as a historical suffix. To refit that
suffix against the amended GLM instead, create a fresh review from the unamended
checkpoint and choose `True` explicitly:

```python
refit_review = glm_then_tree.edit_rates("GLM and tree refitted")
refit_review.set_relativity(
    "DriverAge",
    level=main_rate_row["label"],
    value=1.05,
    component="GLM main effects",
)
refit_review.preview().show()
refitted_suffix = refit_review.apply(refit_later_components=True)
refitted_suffix.summary()
```

These are different modelling decisions. Neither mutates `glm_then_tree`.
