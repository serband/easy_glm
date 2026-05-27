"""Step-by-step pipeline (advanced): blueprint → prep → fit_lasso_glm → RateModel.

Use this when you need control between stages. Most users should prefer
``examples/basic_usage.py`` (``EasyGLM.fit``).

Run from project root:
    PYTHONPATH=src python examples/advanced_pipeline.py
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import easy_glm
from easy_glm.engine import RateModel

raw = easy_glm.load_external_dataframe()
raw = raw.with_columns(
    (pl.Series(np.random.rand(raw.height)) < 0.7).cast(pl.Int8).alias("traintest")
)

predictors = ["VehAge", "Region", "VehGas", "DrivAge", "BonusMalus", "Density"]

blueprint = easy_glm.generate_blueprint(raw)

prepped = easy_glm.prepare_data(
    df=raw,
    modelling_variables=predictors,
    additional_columns=["Exposure", "ClaimNb", "traintest"],
    formats=blueprint,
    traintest_column="traintest",
    table_name="cars",
)

model = easy_glm.fit_lasso_glm(
    dataframe=prepped,
    target="ClaimNb",
    model_type="Poisson",
    weight_col="Exposure",
    train_test_col="traintest",
    use_cv=True,
    divide_target_by_weight=True,
)

vehage_tbl = easy_glm.ratetable(
    model=model,
    dataset=raw,
    col_name="VehAge",
    levels=blueprint["VehAge"],
    prepare=lambda d: easy_glm.prepare_data(
        df=d,
        modelling_variables=predictors,
        formats=blueprint,
        table_name="line_prepped",
    ),
    random_seed=42,
)
print(vehage_tbl.head())

all_tables = easy_glm.generate_all_ratetables(
    model=model,
    dataset=raw,
    predictor_variables=predictors,
    blueprint=blueprint,
)
print(f"Generated {len(all_tables)} rate tables: {list(all_tables)}")

rm = RateModel.from_rate_tables(
    all_tables=all_tables,
    blueprint=blueprint,
    base_rate=0.1,
    model_type="poisson",
    target="ClaimNb",
    weight_col="Exposure",
    exposure_col="Exposure",
    train_test_col="traintest",
)
rm.to_json("french_motor.easyglm")
print("Exported model to french_motor.easyglm")

loaded = RateModel.from_json("french_motor.easyglm")

sample = raw.head(3)
print(f"Predictions (with exposure): {loaded.predict(sample)}")

test_data = raw.filter(pl.col("traintest") == 0)
test_preds = loaded.predict(test_data)
print(f"Overall Test A/E: {test_data['ClaimNb'].sum() / test_preds.sum():.4f}")

if __name__ == "__main__":
    # Optional matplotlib A/E grid (same as former basic_usage.py)
    scored = test_data.with_columns(pred=pl.Series("pred", test_preds))
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="white")
    axes = axes.flatten()
    for i, var in enumerate(predictors):
        bins = blueprint[var]
        is_num = bins and isinstance(bins[0], int | float)
        ae = (
            scored.with_columns(
                grouping=(
                    pl.col(var).qcut(10, allow_duplicates=True)
                    if is_num
                    else pl.col(var).cast(pl.Utf8)
                ),
                _val=pl.col(var) if is_num else pl.lit(None),
            )
            .group_by("grouping")
            .agg(
                actual=pl.col("ClaimNb").sum() / pl.col("Exposure").sum(),
                expected=pl.col("pred").sum() / pl.col("Exposure").sum(),
                exposure=pl.col("Exposure").sum(),
                mid=pl.col("_val").mean() if is_num else pl.lit(None),
            )
            .sort("grouping")
        ).to_pandas()
        x = range(len(ae))
        ax = axes[i]
        labels = ae["mid"].round(1) if is_num else ae["grouping"]
        ax.bar(
            x,
            ae["exposure"] / ae["exposure"].sum(),
            color="#c0c0c0",
            alpha=0.6,
            zorder=1,
        )
        ax.plot(x, ae["actual"], "o-", color="#1f77b4", linewidth=2, zorder=3)
        ax.plot(x, ae["expected"], "s--", color="#ff7f0e", linewidth=2, zorder=3)
        ax.set_title(var, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Actual vs Expected — test set", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.show()
