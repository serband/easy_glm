"""Benchmark demo: compare easy_glm against statsmodels and CatBoost.

Generates synthetic datasets and runs a single-call benchmark.
Install optional deps for the full comparison:

    pip install easy_glm[benchmark]
    python examples/benchmark_demo.py
"""

from easy_glm.benchmarking import run_benchmarks

# ---------------------------------------------------------------------------
# All the heavy lifting lives in the benchmarking package.
# Tune n_rows / seed here; everything else is automatic.
# ---------------------------------------------------------------------------

results = run_benchmarks(n_rows=5_000, seed=42)

# results is a Polars DataFrame — slice, filter, or write out as needed.
print("\nTop 5 by lowest deviance:")
print(
    results.sort("Deviance")
    .select("Dataset", "Method", "Deviance", "RMSE", "FitTime_s")
    .head(5)
)
