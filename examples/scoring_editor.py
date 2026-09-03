"""Score new business from a saved model and prepare it for review.

First create ``my_model.easyglm`` with ``python examples/basic_usage.py``.
Then run:

    python examples/scoring_editor.py my_model.easyglm

This example does not fit a model. A saved scorer is the hand-off point
between modelling, pricing review and portfolio scoring.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from easy_glm.engine import RateModel


def score(model_path: Path) -> RateModel:
    """Load a scorer, score a small new-business file and save a review copy."""
    rate_model = RateModel.from_json(model_path)
    new_business = pl.DataFrame(
        {
            "DrivAge": [35, 52, 22],
            "Region": ["Ile-de-France", "Bretagne", "Nord-Pas-de-Calais"],
            "BonusMalus": [50, 68, 90],
            "Density": [2000, 500, 8000],
        }
    )

    predictions = rate_model.predict(new_business, exposure_col=None)
    print(f"Loaded {model_path}: {len(rate_model.variables)} rating variables")
    print("Per-unit predicted frequency or cost:")
    for risk, prediction in enumerate(predictions, start=1):
        print(f"  Risk {risk}: {prediction:.6f}")

    # A mapping makes a saved scorer usable when a downstream file has
    # different but unambiguous field names.
    renamed_data = new_business.rename(
        {
            "DrivAge": "driver_age",
            "Region": "region_code",
            "BonusMalus": "bonus_malus",
            "Density": "population_density",
        }
    )
    rate_model.column_mapping = {
        "driver_age": "DrivAge",
        "region_code": "Region",
        "bonus_malus": "BonusMalus",
        "population_density": "Density",
    }
    assert (rate_model.predict(renamed_data, exposure_col=None) == predictions).all()
    rate_model.create_snapshot("Source columns mapped for review")
    out = Path("review_copy.easyglm")
    rate_model.to_json(out)
    print(f"Wrote {out} with the source-column mapping and a named snapshot.")
    return rate_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=Path("my_model.easyglm"),
        help="saved .easyglm model (default: my_model.easyglm)",
    )
    parser.add_argument(
        "--open-editor",
        action="store_true",
        help="open the saved scorer in the browser editor after scoring",
    )
    args = parser.parse_args()
    if not args.model.exists():
        print(
            f"No saved model at {args.model}. Run 'python examples/basic_usage.py' "
            "first, then pass the .easyglm file to this scoring script."
        )
        return
    rate_model = score(args.model)
    if args.open_editor:
        rate_model.launch_editor()
        print("Opened the editor. Supply data there to calculate A/E.")
    else:
        print(
            "Open the editor with: python examples/scoring_editor.py my_model.easyglm --open-editor"
        )


if __name__ == "__main__":
    main()
