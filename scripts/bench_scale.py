"""Scale benchmark for piece G: how big a book fits, how long it takes.

Fits the same synthetic motor book (about 190 design columns) at several row
counts, once on the **compact** design matrix and once on the **dense** one,
and records for each run the wall-clock time, the peak resident memory of the
process and the bytes the design matrix actually holds — checked against the
arithmetic ``DesignSpec.expected_design_bytes`` predicts.

Every run happens in its **own subprocess**, so one run's peak memory can never
be blamed on another's and a run that is killed by the operating system is
reported rather than taking the whole benchmark with it.

Run from the repository root::

    PYTHONPATH=src .venv/bin/python scripts/bench_scale.py
    PYTHONPATH=src .venv/bin/python scripts/bench_scale.py --sizes 200000,1000000
    PYTHONPATH=src .venv/bin/python scripts/bench_scale.py --sizes 5000000 \
        --representations sparse --json results.json

``--check-budget BYTES`` makes the process exit non-zero if any run's peak
resident memory is above the budget; that is how the ``-m slow`` test in
``tests/test_scale.py`` asserts the 5M-row promise.

**Read the peak-memory numbers with one caveat**: on macOS ``ru_maxrss`` is
deflated when the memory compressor is active, so a run that is quietly paging
can look cheaper than it is. The fit time per row is the honest paging
detector: it should barely move between sizes.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow a plain `python scripts/bench_scale.py`
    sys.path.insert(0, str(ROOT / "src"))

from easy_glm import DesignSpec, fit_two_stage  # noqa: E402
from easy_glm.core.design import design_bytes, quantile_knots  # noqa: E402

DEFAULT_SIZES = (200_000, 1_000_000, 5_000_000)
#: The plan's promise: 5M rows and about 200 columns inside three gigabytes.
BUDGET_5M_BYTES = 3 * 1024**3
#: Knots per step variable — chosen so the design lands near 200 columns.
STEP_KNOTS = {
    "DrivAge": 20,
    "VehAge": 15,
    "BonusMalus": 25,
    "Density": 20,
    "Mileage": 18,
    "VehValue": 22,
}
LINEAR_VAR = "Premium"
LINEAR_KNOTS = 5


def peak_rss_bytes() -> int:
    """Peak resident set size of this process, in bytes on every platform."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def synthetic_book(n: int, seed: int = 0) -> pl.DataFrame:
    """A motor book with the shape a real one has: skewed numerics, a long
    tail of categorical levels, about one per cent nulls in two columns."""
    rng = np.random.default_rng(seed)

    def levels(count: int, prefix: str) -> np.ndarray:
        weights = 1.0 / np.arange(1, count + 1) ** 1.2
        weights /= weights.sum()
        names = np.array([f"{prefix}{i:02d}" for i in range(count)])
        return names[rng.choice(count, n, p=weights)]

    driv_age = np.clip(rng.gamma(9.0, 4.5, n) + 18.0, 18, 95)
    veh_age = np.clip(rng.gamma(2.0, 3.0, n), 0, 40)
    bonus = np.clip(50 + rng.gamma(2.0, 12.0, n), 50, 230)
    density = np.exp(rng.normal(6.0, 1.6, n))
    mileage = np.clip(rng.gamma(3.0, 4000.0, n), 0, 120_000)
    veh_value = np.clip(rng.gamma(4.0, 4000.0, n), 500, 200_000)
    premium = np.clip(rng.gamma(5.0, 60.0, n), 40, 5_000)
    exposure = np.clip(rng.beta(4.0, 2.0, n), 0.02, 1.0)

    frame = pl.DataFrame(
        {
            "DrivAge": driv_age,
            "VehAge": veh_age,
            "BonusMalus": bonus,
            "Density": density,
            "Mileage": mileage,
            "VehValue": veh_value,
            LINEAR_VAR: premium,
            "Region": levels(22, "R"),
            "VehBrand": levels(11, "B"),
            "Area": levels(6, "A"),
            "VehGas": np.where(rng.random(n) < 0.55, "Regular", "Diesel"),
            "Exposure": exposure,
        }
    )
    log_mu = (
        -2.6
        + 0.012 * (bonus - 50)
        - 0.02 * np.clip(driv_age - 18, 0, 30)
        + 0.03 * np.clip(veh_age, 0, 12)
        + 0.10 * np.log1p(density) / 3.0
        + 0.00002 * mileage
    )
    claims = rng.poisson(np.exp(log_mu) * exposure).astype(np.float64)
    frame = frame.with_columns(pl.Series("ClaimNb", claims))
    # ~1 % nulls in one numeric and one categorical, as a real extract has
    null_numeric = rng.random(n) < 0.01
    null_category = rng.random(n) < 0.01
    return frame.with_columns(
        pl.when(pl.Series(null_numeric))
        .then(None)
        .otherwise(pl.col("VehAge"))
        .alias("VehAge"),
        pl.when(pl.Series(null_category))
        .then(None)
        .otherwise(pl.col("VehBrand"))
        .alias("VehBrand"),
    )


def book_spec(data: pl.DataFrame) -> DesignSpec:
    """The design the benchmark fits: six step terms (one with nulls), one
    piecewise-linear term, four categoricals and one interaction — about 190
    columns, the size of a real personal-lines model."""
    knots = {var: quantile_knots(data[var], count) for var, count in STEP_KNOTS.items()}
    knots[LINEAR_VAR] = quantile_knots(data[LINEAR_VAR], LINEAR_KNOTS)
    return DesignSpec.from_data(
        data,
        [*STEP_KNOTS, LINEAR_VAR, "Region", "VehBrand", "Area", "VehGas"],
        knots=knots,
        linear=[LINEAR_VAR],
        min_level_share=0.002,
        interactions=[("DrivAge", "Region")],
        min_cell_exposure=0.004,
    )


def run_one(n: int, representation: str, alpha: float, seed: int) -> dict:
    """One benchmark point, meant to be the only thing this process ever does."""
    sparse = representation == "sparse"
    t0 = time.perf_counter()
    data = synthetic_book(n, seed=seed)
    t_data = time.perf_counter() - t0

    t0 = time.perf_counter()
    spec = book_spec(data)
    t_spec = time.perf_counter() - t0
    rss_after_load = peak_rss_bytes()

    t0 = time.perf_counter()
    design = spec.build(data, sparse=sparse)
    t_build = time.perf_counter() - t0
    measured_bytes = design_bytes(design)
    expected_bytes = spec.expected_design_bytes(n) if sparse else design.nbytes
    n_columns = int(design.shape[1])
    del design

    t0 = time.perf_counter()
    fit = fit_two_stage(
        data,
        spec,
        "ClaimNb",
        family="poisson",
        weight_col="Exposure",
        divide_target_by_weight=True,
        alpha=alpha,
        sparse=sparse,
    )
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    predictions = fit.predict(data)
    t_score = time.perf_counter() - t0

    return {
        "rows": n,
        "representation": representation,
        "columns": n_columns,
        "non_zero_coefficients": int((fit.coef != 0).sum()),
        "mean_prediction": float(predictions.mean()),
        "design_bytes": measured_bytes,
        "expected_design_bytes": int(expected_bytes),
        "seconds_data": t_data,
        "seconds_spec": t_spec,
        "seconds_build": t_build,
        "seconds_fit": t_fit,
        "seconds_score": t_score,
        "microseconds_fit_per_row": 1e6 * t_fit / n,
        "peak_rss_bytes": peak_rss_bytes(),
        "rss_after_load_bytes": rss_after_load,
    }


def _gb(value: float) -> str:
    return f"{value / 1024**3:.2f} GB"


def _mb(value: float) -> str:
    return f"{value / 1024**2:.0f} MB"


def print_table(records: list[dict]) -> None:
    header = (
        f"{'rows':>10}  {'design':>8}  {'cols':>5}  {'build':>7}  {'fit':>8}  "
        f"{'score':>7}  {'design bytes':>13}  {'formula':>13}  {'peak RSS':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        if r.get("failed"):
            print(
                f"{r['rows']:>10,}  {r['representation']:>8}  "
                f"FAILED ({r['failed']})"
            )
            continue
        print(
            f"{r['rows']:>10,}  {r['representation']:>8}  {r['columns']:>5}  "
            f"{r['seconds_build']:>6.1f}s  {r['seconds_fit']:>7.1f}s  "
            f"{r['seconds_score']:>6.1f}s  {_mb(r['design_bytes']):>13}  "
            f"{_mb(r['expected_design_bytes']):>13}  {_gb(r['peak_rss_bytes']):>9}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        default=",".join(str(s) for s in DEFAULT_SIZES),
        help="comma-separated row counts",
    )
    parser.add_argument(
        "--representations",
        default="sparse,dense",
        help="comma-separated: sparse, dense (dense is skipped above 1M rows)",
    )
    parser.add_argument("--alpha", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None, help="write the records")
    parser.add_argument(
        "--check-budget",
        type=int,
        default=None,
        help="fail if any run's peak RSS is above this many bytes",
    )
    parser.add_argument("--run", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--representation", default="sparse", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.run is not None:  # child mode: one point, JSON on stdout
        record = run_one(args.run, args.representation, args.alpha, args.seed)
        print("JSON " + json.dumps(record))
        return 0

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    representations = [r.strip() for r in args.representations.split(",") if r.strip()]
    records: list[dict] = []
    for n in sizes:
        for representation in representations:
            if representation == "dense" and n > 1_000_000:
                print(
                    f"skipping dense at {n:,} rows: {8 * n * 190 / 1024**3:.1f} GB of "
                    "design alone, which is what piece G exists to avoid",
                    file=sys.stderr,
                )
                continue
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--run",
                str(n),
                "--representation",
                representation,
                "--alpha",
                str(args.alpha),
                "--seed",
                str(args.seed),
            ]
            print(f"running {n:,} rows, {representation} ...", file=sys.stderr)
            completed = subprocess.run(command, capture_output=True, text=True)
            line = next(
                (ln for ln in completed.stdout.splitlines() if ln.startswith("JSON ")),
                None,
            )
            if line is None:
                records.append(
                    {
                        "rows": n,
                        "representation": representation,
                        "failed": f"exit {completed.returncode}",
                        "stderr": completed.stderr[-2000:],
                    }
                )
                print(completed.stderr[-2000:], file=sys.stderr)
                continue
            records.append(json.loads(line[len("JSON ") :]))

    print_table(records)
    if args.json is not None:
        args.json.write_text(json.dumps(records, indent=2))

    status = 0
    for record in records:
        if record.get("failed"):
            status = 1
            continue
        if record["representation"] == "sparse":
            expected = record["expected_design_bytes"]
            if record["design_bytes"] != expected:
                print(
                    f"design bytes {record['design_bytes']:,} != formula "
                    f"{expected:,} at {record['rows']:,} rows",
                    file=sys.stderr,
                )
                status = 1
        if (
            args.check_budget is not None
            and record["peak_rss_bytes"] > args.check_budget
        ):
            print(
                f"peak RSS {_gb(record['peak_rss_bytes'])} above the budget "
                f"{_gb(args.check_budget)} at {record['rows']:,} rows "
                f"({record['representation']})",
                file=sys.stderr,
            )
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
