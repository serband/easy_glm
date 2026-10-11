# Examples and workflows

Start with [Building a pricing model](pricing_walkthrough.md): fit a small GLM,
check A/E, add missing factors and interactions, amend rates and export to Excel.
For the browser version, use the [Workbench walkthrough](workbench_walkthrough.md).

Use the guides below for other model orders, full vehicle, occupation and postcode
files, and scoring new data.

| What you want to do | Worked example |
| --- | --- |
| Fit a small GLM, inspect A/E and find missing factors | [Building a pricing model](pricing_walkthrough.md#fit-the-first-glm) |
| Control automatic bins or supply your own numeric cuts | [Choose the bins](pricing_walkthrough.md#set-how-numeric-variables-are-binned) |
| Compare numeric bands with smooth curves and control their tails | [Numeric curves](numeric_curves.md) |
| Add two interaction tables in order | [Fit and review interactions](pricing_walkthrough.md#fit-the-first-interaction) |
| Set up shared roles and different variables for each GUI model | [Workbench walkthrough](workbench_walkthrough.md) |
| Build vehicle, occupation and postcode files after a GLM and its interactions | [Stacking models and building rating files](stacked_models.md) |
| Score every entry in a catalogue, including codes with no training rows | [Full-file scoring and exports](stacked_models.md#export-reopen-and-score-an-unseen-catalogue-code) |
| Score a renewal extract, an individual rating file or a refreshed master | [Score new data and export rating files](new_data_scoring.md) |
| Fit CatBoost alone, GLM then CatBoost, or CatBoost then GLM | [Other model workflows](model_variations.md) |
| Model pure premium, paid severity or claim occurrence | [Model families](model_families.md) |
| Choose Optuna trials, a tuning-time limit or manual CatBoost settings | [CatBoost fitting controls](model_variations.md#standalone-catboost-with-a-trial-budget) |
| Change bands, keep earlier models and compare candidates | [Revised numeric bands](pricing_walkthrough.md#fit-the-first-glm) |
| Understand, save and run `Project.from_dict` recipes | [Save, edit and run a model recipe](project_workflow.md) |
| Take a fitted GUI workflow into Python, or score without refitting | [Project and scorer exports](project_workflow.md#export-the-accepted-setup-and-the-fitted-scorer) |
| Give an LLM the package's current conventions | [Usage guide](../docs/LLM_USAGE_GUIDE.md) |

The pricing walkthrough uses the published v0.472 API. Staged models, numeric
curves and the new workbench screens are [unreleased](../docs/HANDOVER.md).

## Which API should I use?

| What you want to do | Use |
| --- | --- |
| Build and review models one step at a time | `PricingSession` sets up the data; each fit returns a `PricingModel` to inspect, amend or build on. |
| Run or edit the workbench's saved settings in Python | `Project` reads the setup; `prepare` prepares the data and `run_model` fits a named model. |
| Apply fitted rates to data | Load the saved model or use its exported Python scorer. |

Save both the project setup and fitted model if you want to return to the
modelling work and keep the existing rates.

## Read the model chain

For the rating-file example, the fitting sequence is:

```mermaid
flowchart LR
    A[GLM main effects] --> B[First interaction table]
    B --> C[Second interaction table]
    C --> D[Vehicle file]
    D --> E[Reviewed vehicle rates]
    E --> F[Occupation file]
    F --> G[Postcode file]
```

In the workbench, add interaction tables inside the GLM after reviewing its
main effects. Each later fit uses the combined prediction from everything
before it as its offset. Occupation therefore corrects the GLM, both interaction
tables and the reviewed vehicle file.

A/E after Vehicle checks that combined prediction against actual claims.
The Vehicle file alone contains multipliers to apply to the preceding score.

## How to work through the examples

Run a page's blocks in order in the same notebook or session. Inspect each fit
before deciding what to change. Named models remain available for comparison,
and commented blocks show alternatives you can try.

Use training data for model development and the explicit holdout steps for
the final check. `cv=True` compares models on held-back training folds.
