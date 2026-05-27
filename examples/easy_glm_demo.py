"""End-to-end demo: EasyGLM.fit and optional relativity editor.

Usage:
    PYTHONPATH=src python examples/easy_glm_demo.py
    PYTHONPATH=src python examples/easy_glm_demo.py --no-ui
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import polars as pl

from easy_glm import EasyGLM, load_external_dataframe

BASE_RATE = 0.05
PREDICTORS = [
    "VehAge",
    "Region",
    "VehGas",
    "DrivAge",
    "BonusMalus",
    "Density",
]


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
        print("Loading French motor insurance dataset...")
        df = load_external_dataframe()
        if args.sample and args.sample < df.height:
            df = df.sample(n=args.sample, seed=42)
        df = df.with_columns(
            (pl.Series(np.random.rand(df.height)) < 0.7)
            .cast(pl.Int8)
            .alias("traintest")
        )
        print(f"  {df.height} rows")

        print("Fitting EasyGLM pipeline...")
        eglm = EasyGLM.fit(
            data=df,
            target="ClaimNb",
            model_type="Poisson",
            predictors=PREDICTORS,
            weight_col="Exposure",
            train_test_col="traintest",
            divide_target_by_weight=True,
            use_cv=True,
            base_rate=BASE_RATE,
        )
        print(f"  Variables: {list(eglm.rate_model.variables.keys())}")
        print(f"  Intercept: {eglm.model.intercept_:.4f}")

        sample = df.select(PREDICTORS).head(3)
        print(f"  Sample preds: {eglm.rate_model.predict(sample)}")

        save_path = args.save or "model.easyglm"
        print(f"Saving to {save_path}...")
        eglm.rate_model.to_json(save_path)

        if not args.no_ui:
            print(f"Launching editor on http://localhost:{args.port} ...")
            eglm.launch_editor(data=df, port=args.port)
        else:
            print("Done. Reload with RateModel.from_json(...) to score or edit.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
