"""End-to-end easy_glm demo: build a model and launch the relativity editor.

Uses the bundled French motor insurance dataset (same as basic_usage.py).

Usage:
    python examples/easy_glm_demo.py            # full run, opens UI
    python examples/easy_glm_demo.py --no-ui    # build & save only
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import polars as pl

from easy_glm import (
    fit_lasso_glm,
    generate_blueprint,
    load_external_dataframe,
    prepare_data,
)
from easy_glm.engine import RateModel

BASE_RATE = 0.05
PREDICTORS = [
    "VehAge",
    "Region",
    "VehGas",
    "DrivAge",
    "BonusMalus",
    "Density",
]


def pipeline(df: pl.DataFrame) -> tuple:
    print("1. Generating blueprint...")
    bp = generate_blueprint(df)
    print(f"   {len(bp)} columns processed")
    for var in PREDICTORS:
        vals = bp[var]
        print(f"   {var}: {type(vals[0]).__name__}, {len(vals)} levels")

    print("2. Preparing data...")
    additional = ["Exposure", "ClaimNb", "traintest"]
    prepped = prepare_data(
        df=df,
        modelling_variables=PREDICTORS,
        additional_columns=additional,
        formats=bp,
        traintest_column="traintest",
        table_name="cars",
    )
    print(f"   Prepared shape: {prepped.shape}")

    print("3. Fitting LASSO GLM (Poisson)...")
    model = fit_lasso_glm(
        dataframe=prepped,
        target="ClaimNb",
        model_type="Poisson",
        weight_col="Exposure",
        train_test_col="traintest",
        divide_target_by_weight=True,
    )
    print(f"   {len(model.coef_)} coefficients, intercept={model.intercept_:.4f}")
    return bp, model


def build_rate_model(model, df: pl.DataFrame, bp: dict) -> RateModel:
    print("4. Creating RateModel (from_glm_model)...")
    rm = RateModel.from_glm_model(
        model=model,
        dataset=df,
        blueprint=bp,
        base_rate=BASE_RATE,
        model_type="poisson",
        target="ClaimNb",
        weight_col="Exposure",
        train_test_col="traintest",
        predictor_variables=PREDICTORS,
    )

    print(f"   Variables: {list(rm.variables.keys())}")
    print(
        f"   Metadata: type={rm.metadata.model_type}, "
        f"target={rm.metadata.target}, "
        f"weight={rm.metadata.weight_col}"
    )
    print(f"   Snapshots: {rm.current_version}")

    print("5. Testing predictions...")
    sample = df.select(PREDICTORS).head(3)
    preds = rm.predict(sample)
    print(f"   {sample}")
    print(f"   Predictions: {preds}")
    return rm


def main():
    parser = argparse.ArgumentParser(description="easy_glm end-to-end demo")
    parser.add_argument("--no-ui", action="store_true", help="Build and save only")
    parser.add_argument("--port", type=int, default=8501, help="UI port")
    parser.add_argument(
        "--sample", type=int, default=10000, help="Rows to sample (0 = all)"
    )
    parser.add_argument("--save", type=str, default=None, help="Path to save the model")
    args, _ = parser.parse_known_args()

    try:
        print("0. Loading French motor insurance dataset...")
        df = load_external_dataframe()
        if args.sample and args.sample < df.height:
            df = df.sample(n=args.sample, seed=42)
        df = df.with_columns(
            (pl.Series(np.random.rand(df.height)) < 0.7)
            .cast(pl.Int8)
            .alias("traintest")
        )
        print(f"   {df.height} rows, {len(df.columns)} columns")

        bp, model = pipeline(df)
        rm = build_rate_model(model, df, bp)

        save_path = args.save or "model.easyglm"
        print(f"6. Saving to {save_path}...")
        rm.to_json(save_path)
        print(f"   Saved! ({save_path})")

        if not args.no_ui:
            print(f"\n7. Launching UI editor on http://localhost:{args.port} ...")
            rm.launch_editor(data=df, port=args.port)
        else:
            print("\n✓ Done. To launch the UI, run:")
            print(f"  rm = RateModel.from_json('{save_path}')")
            print("  rm.launch_editor(data=df)")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        print(f"\n✗ Error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
