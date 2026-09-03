# easy_glm

Build an insurance GLM, turn it into rating tables, review it and score a
portfolio. `easy_glm` is for pricing teams that want a regularised model and a
portable table-based scorer without losing the link between the two.

Start with a frequency model. The recommended route is:

1. Fit a small, intelligible model with `EasyGLM.fit`.
2. Use the workflow API when you need deliberate design choices, validation
   and reproducible artefacts.
3. Review a saved `.easyglm` scorer or open the workbench.

The public French motor sample used below is included in a source checkout, so
the examples run offline there. If you installed from PyPI, replace the sample
path with your own Polars frame. Every Python block on this page is tested in
order.

## Install

```bash
pip install easy_glm
# Optional browser workbench and relativity editor
pip install "easy_glm[ui]"
```

## Start here: a first frequency model

### The data you need

Use a Polars `DataFrame` with one policy period per row. It needs:

- a non-negative claim count;
- positive exposure, used both as the frequency denominator and the credibility
  weight;
- predictor columns; and
- a train/holdout flag: `1` for fitting and `0` for validation.

The bundled example calls the count `ClaimNb` and time on risk `Exposure`.
With `divide_target_by_weight=True`, the model fits claim frequency:
`ClaimNb / Exposure`.

This first model uses four familiar motor variables. `EasyGLM.fit` chooses
banded factors for numeric variables and treats text variables as
categoricals. It is the recommended default while you are learning the book.

```python
from pathlib import Path

import numpy as np
import polars as pl

import easy_glm

DATA = Path("tests/fixtures/french_motor_50k.parquet")
df = pl.read_parquet(DATA)
rng = np.random.default_rng(42)
df = df.with_columns(
    pl.Series("traintest", rng.random(len(df)) < 0.7, dtype=pl.Int64)
)

PREDICTORS = ["DrivAge", "Region", "BonusMalus", "Density"]
model = easy_glm.EasyGLM.fit(
    data=df,
    target="ClaimNb",
    model_type="Poisson",
    predictors=PREDICTORS,
    weight_col="Exposure",
    train_test_col="traintest",
    divide_target_by_weight=True,
    alpha=0.001,  # fixed penalty for a quick iteration; use cv=5 for production selection
)
print(model)
print(f"Base rate: {model.rate_model.base_rate:.5f}")
```

The result contains a base rate and one table per factor. The relativity is a
multiplier on the base rate; `1.00` is the selected base band or level. For a
frequency model, expected claims for a policy period are **exposure × base
rate × every applicable relativity**. One table value is not a price by itself.

```python
print(model.relativities["BonusMalus"].select("label", "relativity", "exposure"))

holdout = df.filter(pl.col("traintest") == 0)
expected = model.rate_model.predict(holdout)
ae = holdout["ClaimNb"].sum() / expected.sum()
print(f"Holdout A/E: {ae:.3f}")
```

**Read this first.** Overall A/E compares observed and expected claims.
Values near 1 are a calibration check, not evidence that every factor is
well modelled. Review results by factor before changing a price.

Save the scorer when you want to review or deploy it. A `.easyglm` file is
portable JSON; it contains the tables and does not need `glum` to score.

```python
model.rate_model.to_json("my_model.easyglm")
print("Wrote my_model.easyglm")
```

Rare, unseen and missing categorical values score through the table's
**Other / Unknown** row. Check that row when you review a new portfolio: it is
the model's safe fallback, not evidence that the new value has its own fitted
relativity.

## Next: a model you can explain and validate

Use the lower-level pipeline when you need to control how factors are built.
It separates three decisions:

| Decision | Recommended starting point | Why it matters |
|---|---|---|
| Factor form | Numeric variables as bands; text as categoricals | Gives tables that are easy to inspect |
| Penalty | `cv=5` for a model you expect to use | Chooses the amount of shrinkage from the data |
| Shape | Add monotonicity only where the business relationship warrants it | Prevents an implausible fitted curve |

Here `Density` is a smooth piecewise-linear curve, while `BonusMalus` is
constrained to rise. Build the specification on training data only: that
prevents holdout information affecting knots or category levels.

```python
from easy_glm import DesignSpec, fit_glm, rate_tables, to_rate_model

train = df.filter(pl.col("traintest") == 1)
spec = DesignSpec.from_data(
    train,
    PREDICTORS,
    weight_col="Exposure",
    linear=["Density"],
)
fit = fit_glm(
    train,
    spec,
    target="ClaimNb",
    family="poisson",
    weight_col="Exposure",
    divide_target_by_weight=True,
    alpha=0.001,  # tutorial shortcut; use cv=5 when selecting a production penalty
    monotone={"BonusMalus": "increasing"},
)
tables = rate_tables(fit)
rate_model = to_rate_model(fit, exposure_col="Exposure")
print(tables["Density"].select("label", "relativity", "exposure").head(4))
```

Validate on the holdout. Inspect the spread of A/E by band or level, not a
second total across bins: the bins already add back to overall A/E.

```python
from easy_glm.workflow import ae_by_variable

actual = holdout["ClaimNb"].to_numpy()
expected = rate_model.predict(holdout)
weight = holdout["Exposure"].to_numpy()
by_band = ae_by_variable(holdout, "BonusMalus", actual, expected, weight)
print(by_band.select("label", "exposure", "ae").head(5))
print(f"A/E range: {by_band['ae'].min():.2f}–{by_band['ae'].max():.2f}")
```

**Watch-out.** Tables reproduce the fitted GLM exactly before manual edits.
If you cap, smooth or round a relativity, total expected claims can move. Use
the review tools to measure that change, then rebalance the base rate if the
commercial objective is to keep the portfolio total unchanged.

For a complete, commented version of this workflow, run:

```bash
python examples/advanced_pipeline.py
```

## Review a fitted model

Keep fitting and review separate. The review example reads the scorer written
above rather than fitting a fresh model. It prints factor-level A/E, shows how
to make and assess a table change, and retains a snapshot for comparison.

```bash
python examples/exploring_fit.py my_model.easyglm
```

To score a new portfolio or map different source column names to the saved
model, use the scorer example:

```bash
python examples/scoring_editor.py my_model.easyglm
```

The scoring promise is simple: the unedited `RateModel` gives the same
per-unit predictions as the fitted GLM, including nulls and unseen categories.
You can check it whenever you need an audit trail:

```python
per_unit_model = to_rate_model(fit, exposure_col="Exposure")
assert np.allclose(
    per_unit_model.predict(holdout, exposure_col=None),
    fit.predict(holdout),
    rtol=1e-10,
)
print("Table scorer agrees with the fitted model.")
```

## Use the workbench

The workbench is the visual route through the same workflow: set data roles,
choose factor designs, fit, inspect diagnostics, review tables and export a
scorer. Start in the browser, or run
`python examples/easy_glm_demo.py` to create a ready-to-open project file.

```bash
easy-glm-workbench
```

This opens an empty workbench. To open an existing project, pass its file:

```bash
easy-glm-workbench project.easyglm-project.json
# From a source checkout:
python -m easy_glm.app project.easyglm-project.json
```

For a browser-based review of an existing scorer, launch the editor from a
fitted model. It opens a working copy; the original model is left unchanged.

```python skip-test
model.rate_model.launch_editor(data=df, port=8501)
```

## Specialist recipes

Choose these when the business problem calls for them, not as part of the
first pass through the package.

| Need | Recipe | Key caveat |
|---|---|---|
| Rate change from current premium | `examples/rate_change.py` | The premium becomes an offset; solve the base rate for the target loss ratio |
| Lapse or conversion | `examples/lapse_model.py` | Tables are odds relativities and predictions are probabilities, not amounts |
| Large portfolio | `examples/large_book.py --rows 1000000` | The compact design is selected automatically for large books |
| Interactions | `fit_two_stage` and [the interaction guide](docs/checks/a-interactions.md) | Main tables remain fixed; cells are adjustments on top |
| Command line and reproducible exports | `easy-glm run project.json --out artefacts/` | Artefact commands refit from the project data |

## Where to go next

- [Intermediate workflow example](examples/advanced_pipeline.py): design,
  validation, tables and a portable scorer.
- [Review example](examples/exploring_fit.py): inspect a saved model and test
  a table adjustment.
- [Scoring example](examples/scoring_editor.py): score new business and map
  source columns.
- [Workbench project example](examples/easy_glm_demo.py): create a project
  file, then open it in the browser.
- [Rate-change recipe](examples/rate_change.py), [lapse recipe](examples/lapse_model.py)
  and [large-book recipe](examples/large_book.py).
- [Workbench plan](docs/WORKBENCH_PLAN.md) and [technical checks](docs/checks/)
  for implementation detail and evidence behind the guarantees.

## Command line

```bash
easy-glm validate project.json
easy-glm run project.json --out artefacts/
easy-glm export project.json --script --excel --report
```

## Development

```bash
black . && ruff check . && mypy src/easy_glm/core src/easy_glm/workflow --ignore-missing-imports && pytest -q
```

MIT licensed. See [LICENSE](LICENSE).
