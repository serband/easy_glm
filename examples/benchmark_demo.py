"""Benchmark demo: compare easy_glm against statsmodels and CatBoost.

Generates synthetic datasets (Poisson, Gamma, Gaussian, Binomial)
and benchmarks easy_glm (+/- CV) against statsmodels and CatBoost.

Usage:
    python examples/benchmark_demo.py
"""

import time

import numpy as np
import pandas as pd
import polars as pl

from easy_glm.core.model import fit_lasso_glm, predict_with_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_ROWS = 5_000
SEED = 42
N_NUMERIC = 30
N_CATEGORICAL = 20
N_PREDICTIVE = 15

# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

NUM_NAMES = [f"num_{i}" for i in range(N_NUMERIC)]
CAT_NAMES = [f"cat_{i}" for i in range(N_CATEGORICAL)]
CAT_POOL = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def _make_predictors(rng, n_rows):
    data = {}
    for i, name in enumerate(NUM_NAMES):
        k = i % 4
        if k == 0:
            data[name] = rng.uniform(-2, 2, size=n_rows)
        elif k == 1:
            data[name] = rng.normal(0, 1, size=n_rows)
        elif k == 2:
            data[name] = rng.lognormal(0, 0.5, size=n_rows)
        else:
            data[name] = rng.uniform(0, 10, size=n_rows)
    for name in CAT_NAMES:
        n_levels = rng.integers(3, 9)
        levels = CAT_POOL[:n_levels]
        probs = rng.dirichlet(np.ones(n_levels))
        data[name] = rng.choice(levels, size=n_rows, p=probs).tolist()
    return pl.DataFrame(data)


def _make_coefficients(rng):
    coeffs = {}
    shuffled_num = list(NUM_NAMES)
    rng.shuffle(shuffled_num)
    pred_num = set(shuffled_num[:N_PREDICTIVE])
    for name in NUM_NAMES:
        coeffs[name] = float(rng.uniform(-1.5, 1.5)) if name in pred_num else 0.0
    shuffled_cat = list(CAT_NAMES)
    rng.shuffle(shuffled_cat)
    pred_cat = set(shuffled_cat[:N_PREDICTIVE])
    for name in CAT_NAMES:
        coeffs[name] = float(rng.uniform(-1.5, 1.5)) if name in pred_cat else 0.0
    return coeffs


def _encode_cats(df):
    encoded = []
    df = df.clone()
    for name in CAT_NAMES:
        levels = sorted(df[name].unique().to_list())
        if len(levels) <= 2:
            continue
        for lvl in levels[1:]:
            col = f"{name}_{lvl}"
            df = df.with_columns((pl.col(name) == lvl).cast(pl.Int64).alias(col))
            encoded.append(col)
    return df, encoded


def _linear_predictor(df, coeffs, encoded):
    eta = np.zeros(df.height, dtype=np.float64)
    for name in NUM_NAMES:
        eta += df[name].to_numpy() * coeffs[name]
    for name in encoded:
        base = name.rsplit("_", 1)[0]
        eta += df[name].to_numpy() * coeffs.get(base, 0.0)
    return eta


def _make_dataset(df, exposure, response):
    out = df.select(NUM_NAMES + CAT_NAMES).clone()
    return out.with_columns(
        pl.Series("Exposure", exposure),
        pl.Series("Response", response),
        pl.Series("traintest", (rng.random(N_ROWS) < 0.75).astype(np.int8)),
    )


rng = np.random.default_rng(SEED)
df = _make_predictors(rng, N_ROWS)
df_encoded, encoded_cols = _encode_cats(df)
coeffs = _make_coefficients(rng)
eta = _linear_predictor(df_encoded, coeffs, encoded_cols)
exposure = np.exp(rng.normal(0, 0.3, size=N_ROWS))

# Generate response for each family
poisson_data = _make_dataset(
    df, exposure, rng.poisson(np.clip(exposure * np.exp(-1.8 + eta), 0, 50))
)
gamma_data = _make_dataset(
    df, exposure, rng.gamma(1.5, np.clip(exposure * np.exp(0.5 + eta), 0.1, None) / 1.5)
)
gaussian_data = _make_dataset(df, exposure, rng.normal(2.0 + eta, 2.5))
binomial_data = _make_dataset(
    df,
    exposure,
    rng.binomial(1, np.clip(1.0 / (1.0 + np.exp(0.3 - eta)), 0.001, 0.999)),
)

DATASETS = {
    "poisson": poisson_data,
    "gamma": gamma_data,
    "gaussian": gaussian_data,
    "binomial": binomial_data,
}

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _train_test_split(pdf):
    pred_cols = [
        c for c in pdf.columns if c not in {"Response", "Exposure", "traintest"}
    ]
    num_cols = [c for c in pred_cols if c.startswith("num_")]
    cat_cols = [c for c in pred_cols if c.startswith("cat_")]
    x = pdf[num_cols].copy()
    for c in cat_cols:
        dummies = pd.get_dummies(pdf[c], prefix=c, drop_first=True).astype(np.float64)
        x = pd.concat([x, dummies], axis=1)
    y = pdf["Response"].to_numpy()
    train = pdf["traintest"].to_numpy() == 1
    return (
        x.iloc[train].reset_index(drop=True),
        y[train],
        x.iloc[~train].reset_index(drop=True),
        y[~train],
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _poisson_deviance(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-9, None)
    y_safe = np.clip(y_true, 1e-9, None)
    dev = y_safe * np.log(y_safe / y_pred) - (y_safe - y_pred)
    return float(np.mean(np.where(y_true == 0, y_pred, dev)) * 2)


def _gamma_deviance(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-9, None)
    return float(np.mean((y_true - y_pred) / y_pred - np.log(y_true / y_pred)) * 2)


_deviance = {
    "poisson": _poisson_deviance,
    "gamma": _gamma_deviance,
    "gaussian": lambda yt, yp: float(np.mean((yt - yp) ** 2)),
    "binomial": lambda yt, yp: float(
        -np.mean(
            yt * np.log(np.clip(yp, 1e-15, 1 - 1e-15))
            + (1 - yt) * np.log(np.clip(1 - yp, 1e-15, 1 - 1e-15))
        )
        * 2
    ),
}

FAMILY_LABEL = {
    "poisson": "Poisson",
    "gamma": "Gamma",
    "gaussian": "Gaussian",
    "binomial": "Binomial",
}

# ---------------------------------------------------------------------------
# Check optional deps
# ---------------------------------------------------------------------------

has_sm, has_cb = False, False
try:
    import statsmodels.api as sm

    has_sm = True
except ImportError:
    pass
try:
    from catboost import CatBoostClassifier, CatBoostRegressor

    has_cb = True
except ImportError:
    pass

print("=" * 60)
print("  easy_glm Benchmark Demo")
parts = ["easy_glm (+/- CV)"]
if has_sm:
    parts.append("statsmodels")
if has_cb:
    parts.append("CatBoost")
print(f"  Comparing: {' vs '.join(parts)}")
print("=" * 60)

if not has_sm and not has_cb:
    print("\n  Tip: pip install statsmodels catboost (for full comparison)\n")

print(f"\nGenerating {N_ROWS:,}-row synthetic datasets...")
for name, d in DATASETS.items():
    print(f"  {name:10s} — {d.height:,} rows, {len(d.columns)} cols")

# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

all_results = []

for family_key, dataset in DATASETS.items():
    fam_label = FAMILY_LABEL[family_key]
    pdf = dataset.to_pandas()
    X_train, y_train, X_test, y_test = _train_test_split(pdf)

    print(
        f"\n  {family_key} ({X_train.shape[1]} features, {X_train.shape[0]} train / {X_test.shape[0]} test)"
    )

    # --- easy_glm (no CV) ---
    print("    easy_glm (no CV)...", end=" ", flush=True)
    try:
        df_glm = pl.DataFrame(X_train).with_columns(
            pl.Series("Response", y_train),
            pl.Series("traintest", np.ones(len(y_train), dtype=np.int8)),
            pl.Series("Exposure", np.ones(len(y_train))),
        )
        t0 = time.perf_counter()
        model = fit_lasso_glm(
            df_glm, "Response", "traintest", fam_label, "Exposure", use_cv=False
        )
        ft = time.perf_counter() - t0
        t0 = time.perf_counter()
        yp = predict_with_model(model, pl.DataFrame(X_test))
        pt = time.perf_counter() - t0
        if family_key == "binomial":
            yp = np.clip(yp, 1e-15, 1 - 1e-15)
        if family_key == "gamma":
            yp = np.clip(yp, 1e-9, None)
        d = _deviance[family_key](y_test, yp)
        all_results.append(
            {
                "Dataset": family_key,
                "Method": "easy_glm (no CV)",
                "Deviance": round(d, 4),
                "RMSE": round(float(np.sqrt(np.mean((y_test - yp) ** 2))), 4),
                "MAE": round(float(np.mean(np.abs(y_test - yp))), 4),
                "FitTime_s": round(ft, 3),
                "PredTime_s": round(pt, 4),
                "NParams": len(model.coef_),
            }
        )
        print("done")
    except Exception as exc:
        print(f"FAILED ({exc})")

    # --- easy_glm (CV) ---
    print("    easy_glm (CV)...   ", end=" ", flush=True)
    try:
        t0 = time.perf_counter()
        model_cv = fit_lasso_glm(
            df_glm, "Response", "traintest", fam_label, "Exposure", use_cv=True
        )
        ft = time.perf_counter() - t0
        t0 = time.perf_counter()
        yp = predict_with_model(model_cv, pl.DataFrame(X_test))
        pt = time.perf_counter() - t0
        if family_key == "binomial":
            yp = np.clip(yp, 1e-15, 1 - 1e-15)
        if family_key == "gamma":
            yp = np.clip(yp, 1e-9, None)
        d = _deviance[family_key](y_test, yp)
        all_results.append(
            {
                "Dataset": family_key,
                "Method": "easy_glm (CV)",
                "Deviance": round(d, 4),
                "RMSE": round(float(np.sqrt(np.mean((y_test - yp) ** 2))), 4),
                "MAE": round(float(np.mean(np.abs(y_test - yp))), 4),
                "FitTime_s": round(ft, 3),
                "PredTime_s": round(pt, 4),
                "NParams": len(model_cv.coef_),
            }
        )
        print("done")
    except Exception as exc:
        print(f"FAILED ({exc})")

    # --- statsmodels ---
    s = "skipped" if not has_sm else "..."
    print(f"    statsmodels......... {s}", end=" " if has_sm else "\n", flush=True)
    if has_sm:
        try:
            sm_fams = {
                "poisson": sm.families.Poisson(),
                "gamma": sm.families.Gamma(sm.families.links.Log()),
                "gaussian": sm.families.Gaussian(),
                "binomial": sm.families.Binomial(),
            }
            t0 = time.perf_counter()
            xd = sm.add_constant(X_train.to_numpy(dtype=np.float64))
            res = sm.GLM(y_train, xd, family=sm_fams[family_key]).fit()
            ft = time.perf_counter() - t0
            t0 = time.perf_counter()
            yp = res.predict(sm.add_constant(X_test.to_numpy(dtype=np.float64)))
            pt = time.perf_counter() - t0
            if family_key == "binomial":
                yp = np.clip(yp, 1e-15, 1 - 1e-15)
            if family_key == "gamma":
                yp = np.clip(yp, 1e-9, None)
            d = _deviance[family_key](y_test, yp)
            all_results.append(
                {
                    "Dataset": family_key,
                    "Method": "statsmodels",
                    "Deviance": round(d, 4),
                    "RMSE": round(float(np.sqrt(np.mean((y_test - yp) ** 2))), 4),
                    "MAE": round(float(np.mean(np.abs(y_test - yp))), 4),
                    "FitTime_s": round(ft, 3),
                    "PredTime_s": round(pt, 4),
                    "NParams": len(res.params),
                }
            )
            print("done")
        except Exception as exc:
            print(f"FAILED ({exc})")

    # --- catboost ---
    s = "skipped" if not has_cb else "..."
    print(f"    catboost............ {s}", end=" " if has_cb else "\n", flush=True)
    if has_cb:
        try:
            if family_key == "binomial":
                t0 = time.perf_counter()
                cb = CatBoostClassifier(
                    iterations=500,
                    learning_rate=0.1,
                    depth=6,
                    random_seed=42,
                    verbose=False,
                    allow_writing_files=False,
                )
                cb.fit(X_train, y_train.astype(int))
                ft = time.perf_counter() - t0
                t0 = time.perf_counter()
                yp = cb.predict_proba(X_test)[:, 1]
                pt = time.perf_counter() - t0
                n_params = cb.tree_count_
            else:
                yt = (
                    np.clip(y_train, 1e-6, None)
                    if family_key == "gamma"
                    else y_train.copy()
                )
                t0 = time.perf_counter()
                cb = CatBoostRegressor(
                    loss_function={
                        "poisson": "Poisson",
                        "gamma": "RMSE",
                        "gaussian": "RMSE",
                    }[family_key],
                    iterations=500,
                    learning_rate=0.1,
                    depth=6,
                    random_seed=42,
                    verbose=False,
                    allow_writing_files=False,
                )
                cb.fit(X_train, yt)
                ft = time.perf_counter() - t0
                t0 = time.perf_counter()
                yp = cb.predict(X_test)
                pt = time.perf_counter() - t0
                n_params = cb.tree_count_
            if family_key == "binomial":
                yp = np.clip(yp, 1e-15, 1 - 1e-15)
            if family_key == "gamma":
                yp = np.clip(yp, 1e-9, None)
            d = _deviance[family_key](y_test, yp)
            all_results.append(
                {
                    "Dataset": family_key,
                    "Method": "catboost",
                    "Deviance": round(d, 4),
                    "RMSE": round(float(np.sqrt(np.mean((y_test - yp) ** 2))), 4),
                    "MAE": round(float(np.mean(np.abs(y_test - yp))), 4),
                    "FitTime_s": round(ft, 3),
                    "PredTime_s": round(pt, 4),
                    "NParams": n_params,
                }
            )
            print("done")
        except Exception as exc:
            print(f"FAILED ({exc})")

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

cols = {
    "Dataset": 10,
    "Method": 18,
    "Deviance": 12,
    "RMSE": 12,
    "MAE": 12,
    "FitTime_s": 10,
    "PredTime_s": 10,
    "NParams": 10,
}
header = "  ".join(f"{k:<{v}}" for k, v in cols.items())
sep = "  ".join("-" * v for v in cols.values())
bar = "=" * len(header)

print(f"\n{bar}")
print(f"  Benchmark Results — {N_ROWS:,} rows, seed={SEED}")
print(bar)
print(header)
print(sep)

for r in all_results:
    line = "  ".join(
        (
            str(r.get(k, "FAIL"))[: v - 1].ljust(v)
            if r.get(k) is not None
            else "FAIL".ljust(v)
        )
        for k, v in cols.items()
    )
    print(line)

print(bar)
print(f"\n{len(all_results)} benchmarks completed.")
