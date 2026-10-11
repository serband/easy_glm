# Model variations from one pricing book

Use this page when the same data could support more than one modelling order.
For example, you may want a conventional GLM first and a flexible correction
afterwards, or want to see what happens when the flexible model comes first.
Each fit below starts with the same claim outcome and training split. A model
added later is fitted to what the earlier models have not explained, so
`GLM → CatBoost` and `CatBoost → GLM` answer different questions.

> Mixed stacks are unreleased work; see [the current status](../docs/HANDOVER.md).

Fitting and tuning use training rows. The last comparison asks for holdout
results separately.

## Load one reproducible book

This uses synthetic motor data so the results are repeatable.

```python
import polars as pl

from easy_glm import PricingSession, load_motor_catalogue_example

data, _, _, _ = load_motor_catalogue_example(rows=900, seed=73)
```

## Standalone CatBoost with a trial budget

Start with CatBoost on its own. `tune=True` asks Optuna to try settings on
training folds. `trials=2` keeps this example short; use a larger number for
real work. Once it has chosen settings, EasyGLM fits the selected model on the
training rows. Holdout rows are left alone.

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

Here the GLM carries driver age and region. The CatBoost model is then used for
bonus-malus only. `tune=False` means: fit one model with the settings shown,
without an Optuna search. This is useful when you have already agreed the tree
settings or are trying a quick candidate.

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

The CatBoost result is an extra multiplier on top of the GLM. The driver-age and
region GLM tables remain the GLM tables; CatBoost has not refitted them.

## CatBoost followed by a GLM correction

This time CatBoost takes bonus-malus and region first, then a GLM is fitted for
driver age on the remaining signal. `tuning_budget="time"` limits the time spent
trying CatBoost settings on training folds. A trial already under way can finish;
the limit does not include preparing the data, scoring candidates or the final
fit on all training rows.

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

Compare models only when they use the same response and split. The first call
compares them on shared training folds. The second is the independent holdout
check; use it after deciding which candidates are worth testing.

```python
tree_then_glm.compare(tree_only, cv=True)
tree_then_glm.compare(tree_only, subset="holdout")
```

## Gaussian with a log link

Gaussian models can be part of a stack when they use a log link. The log link is
what lets the stages multiply together. This block creates a positive annualised
claims proxy simply to show the mechanics. It is not a recommended severity
model; choose a severity target, population and weights that match the question
you are trying to answer.

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

The interaction is fitted after the Gaussian GLM, and CatBoost is fitted after
both. The result is still a sequence of multipliers. For examples of a Gamma or
binomial response defined from actual insurance outcomes, see
[model_families.md](model_families.md).

## Revise a fitted table without overwriting it

Suppose the driver-age table needs a manual change. `edit_rates` starts a draft;
the fitted model is not changed until `apply`. The preview shows the effect on
the whole score, including the later tree.

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

`fixed_suffix` keeps the later tree as it was fitted. That may be appropriate
when you are applying a final rate change and do not want to rebuild later files.
If you want the tree to be fitted again after the GLM amendment, start again from
the unamended model and use `True`:

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

Both results sit alongside `glm_then_tree`, so you can compare them before
choosing one.
