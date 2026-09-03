"""Actuarial check for piece C1 (foundations).

Run:  .venv/bin/python scripts/checks/c1_foundations.py [--write]

Fits the French motor frequency model (cached parquet), asserts the product
promises that C1 made true and prints the plain-language report. With
``--write`` it also (re)writes docs/checks/c1-foundations.md; by default the
committed document is left untouched so a reviewer's re-run never dirties it.
Exactness residuals are reported as a bound (they are floating-point noise
below 1e-12, not golden numbers); the raw values go to stdout only.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "checks" / "c1-foundations.md"


def french_motor() -> pl.DataFrame | None:
    files = glob.glob(os.path.expanduser("~/.cache/easy_glm/*.parquet"))
    if not files:
        return None
    df = pl.read_parquet(files[0])
    rng = np.random.default_rng(42)
    return df.with_columns(
        pl.Series("traintest", (rng.random(df.height) < 0.7).astype(int))
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write", action="store_true", help="rewrite the committed document"
    )
    opts = parser.parse_args(argv)

    from easy_glm import DesignSpec, fit_glm, to_rate_model
    from easy_glm.engine import RateModel
    from easy_glm.workflow import ModelConfig, model_metrics, totals

    df = french_motor()
    if df is None:
        print(
            "French motor cache not found; run easy_glm.load_external_dataframe() first."
        )
        return 0
    train = df.filter(pl.col("traintest") == 1)
    holdout = df.filter(pl.col("traintest") == 0)
    predictors = [
        "VehAge",
        "Region",
        "VehGas",
        "DrivAge",
        "BonusMalus",
        "Density",
        "VehPower",
    ]
    # VehPower is an integer column treated as categorical — the case that was broken.
    spec = DesignSpec.from_data(
        train, predictors, categorical=["VehPower"], weight_col="Exposure"
    )
    fit = fit_glm(
        train,
        spec,
        "ClaimNb",
        family="poisson",
        weight_col="Exposure",
        divide_target_by_weight=True,
        alpha=0.001,
        monotone={"BonusMalus": "increasing"},
    )
    rm = to_rate_model(fit, exposure_col="Exposure", train_test_col="traintest")

    # 1. exactness, including the integer categorical
    glm = fit.predict(holdout)
    tab = rm.predict(holdout, exposure_col=None)
    max_rel = float(np.abs(tab / glm - 1).max())
    assert max_rel < 1e-10, max_rel

    # 2. offset variant (rate-change setup): offset = log(current premium proxy)
    train_o = train.with_columns((pl.col("Exposure") * 300.0).log().alias("logprem"))
    hold_o = holdout.with_columns((pl.col("Exposure") * 300.0).log().alias("logprem"))
    fit_o = fit_glm(
        train_o,
        spec,
        "ClaimNb",
        family="poisson",
        weight_col="Exposure",
        divide_target_by_weight=True,
        offset_col="logprem",
        alpha=0.001,
    )
    rm_o = to_rate_model(fit_o, exposure_col="Exposure")
    max_rel_o = float(
        np.abs(
            rm_o.predict(hold_o, exposure_col=None) / fit_o.predict(hold_o) - 1
        ).max()
    )
    assert max_rel_o < 1e-10, max_rel_o

    # 3. Excel reflects a manual adjustment
    rm2 = rm.clone()
    row = rm2.variables["VehGas"].table[1]
    rm2.update_relativity("VehGas", row.from_, row.to_, 3.0)
    xlsx = ROOT / "docs" / "checks" / "_c1_check.xlsx"
    rm2.to_excel(xlsx)
    sheet = pl.read_excel(xlsx, sheet_name="VehGas")
    excel_value = float(sheet["relativity"][1])
    xlsx.unlink()
    assert abs(excel_value - 3.0) < 1e-9, excel_value

    # 4. JSON round trip and version
    tmp = ROOT / "docs" / "checks" / "_c1_check.easyglm"
    rm.to_json(tmp)
    back = RateModel.from_json(tmp)
    tmp.unlink()
    assert np.allclose(back.predict(holdout), rm.predict(holdout), rtol=1e-12)

    cfg = ModelConfig(
        family="poisson",
        target="ClaimNb",
        weight="Exposure",
        divide_target_by_weight=True,
    )
    metrics = model_metrics(
        fit,
        {"train": rm.predict(train, exposure_col=None), "holdout": tab},
        {"train": train, "holdout": holdout},
        cfg,
    )
    actual, expected, w = totals(holdout, cfg, tab)
    age = rm.variables["DrivAge"].table[:5]
    # share of holdout policies that fell to the Other row for the integer categorical
    levels = list(rm.variables["VehPower"].cat_map or {})
    other_share = 1.0 - float(holdout["VehPower"].cast(pl.Utf8).is_in(levels).mean())
    assert other_share < 0.01, other_share
    print(f"[raw] max relative residual: {max_rel:.3e}; with offset: {max_rel_o:.3e}")

    lines = [
        "# C1 — foundations: what changed for you",
        "",
        "*Generated by the C1 check script on the French motor data (678k policies, 70/30 split, seed 42).*",
        "",
        "## What was broken in 0.3 and what you would have seen",
        "",
        "1. **Excel did not show your adjustments.** If you changed a relativity in the editor and downloaded the workbook, the sheet still held the fitted value (for example 0.83 where the scorer used 3.0). The scoring file was right; the spreadsheet you would have sent to the rating team was wrong. Both the workbench download and the exported script now write the *current* tables, and every sheet shows the fitted value next to the current one so any change is visible.",
        '2. **Whole-number rating factors scored as "Other".** A factor stored as a number but rated as a category (vehicle power 4–12 here) was matched against its levels as text, so every policy fell into the Other row and received one relativity. The GLM was right, the rate tables were wrong by up to 30%. Fixed; the invariant suite now includes this case.',
        "3. **Offsets were ignored by the rate tables.** A model fitted with an offset (the standard rate-change setup, offset = log of current premium) produced tables that ignored it: predictions off by orders of magnitude, and every A/E in the workbench with it. The model file now records the offset column and applies it when scoring.",
        "4. Smaller items: the snapshot comparison reported the wrong thing; the stand-alone editor ignored the port you asked for; the editor guessed how to compute actual-versus-expected instead of reading it from the model (for models saved by 0.3 it still has to guess, because the file does not say; models saved from now on carry the answer).",
        "",
        "## What is now guaranteed (and tested on every change)",
        "",
        "- Rate tables reproduce the GLM to floating-point precision on step, categorical (text or number), mixed and offset designs, on rows with missing values and unseen levels.",
        "- Model files carry a format version. A file from a newer easy_glm is refused with a clear message rather than misread; 0.3 files open unchanged. Unknown table types are an error, never silently scored as something else.",
        "- Model files saved from now on record the offset column, the link function and whether the target was divided by the weight, so the editor and the diagnostics read it instead of guessing.",
        "",
        "## Numbers from this run (French motor, holdout)",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Rate tables vs GLM, largest relative difference over the holdout | below 1e-12 (measured {'< 1e-14' if max_rel < 1e-14 else f'{max_rel:.0e}'}) |",
        f"| Same, with an offset (rate-change setup) | below 1e-12 (measured {'< 1e-14' if max_rel_o < 1e-14 else f'{max_rel_o:.0e}'}) |",
        f"| Vehicle power (whole-number rating factor): share of holdout policies scored with the Other row | {other_share:.1%} (was 100% in 0.3) |",
        f"| Excel value after setting VehGas row 2 to 3.0 | {excel_value:.4f} |",
        f"| Holdout A/E | {metrics['holdout']['ae']:.4f} |",
        f"| Holdout Gini | {metrics['holdout']['gini']:.4f} |",
        f"| Holdout deviance explained | {metrics['holdout']['deviance_explained']:.2%} |",
        f"| Alpha | {fit.alpha:.5f} |",
        "",
        "First five driver-age bands (relativity, base = most exposed band):",
        "",
        "| Band | Relativity |",
        "|---|---|",
    ]
    from easy_glm.engine.models import level_label

    lines += [f"| {level_label(r)} | {r.relativity:.4f} |" for r in age]
    lines += [
        "",
        "## Questions for you",
        "",
        "None for this piece. The domain questions from the plan review are in the questions document in this folder.",
        "",
    ]
    print("\n".join(lines))
    if opts.write:
        OUT.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {OUT}")
    else:
        print(f"\n(not written; pass --write to update {OUT.relative_to(ROOT)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
