from __future__ import annotations

import numpy as np
import polars as pl

_NUMERIC_COUNT = 30
_CATEGORICAL_COUNT = 20
_NUMERIC_PREDICTIVE = 10
_CATEGORICAL_PREDICTIVE = 5


def _numeric_names() -> list[str]:
    return [f"num_{i}" for i in range(_NUMERIC_COUNT)]


def _categorical_names() -> list[str]:
    return [f"cat_{i}" for i in range(_CATEGORICAL_COUNT)]


def _generate_predictors(
    rng: np.random.Generator,
    n_rows: int,
) -> tuple[pl.DataFrame, list[str], list[str]]:
    num_names = _numeric_names()
    cat_names = _categorical_names()
    data: dict[str, np.ndarray | list[str]] = {}

    for i, name in enumerate(num_names):
        distribution_type = i % 4
        if distribution_type == 0:
            data[name] = rng.uniform(-2, 2, size=n_rows)
        elif distribution_type == 1:
            data[name] = rng.normal(0, 1, size=n_rows)
        elif distribution_type == 2:
            data[name] = rng.lognormal(0, 0.5, size=n_rows)
        else:
            data[name] = rng.uniform(0, 10, size=n_rows)

    cat_levels_pool = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    for _i, name in enumerate(cat_names):
        num_levels = rng.integers(3, 9)
        levels = cat_levels_pool[:num_levels]
        probs = rng.dirichlet(np.ones(num_levels))
        labels = rng.choice(levels, size=n_rows, p=probs)
        data[name] = labels.tolist()

    df = pl.DataFrame(data)
    return df, num_names, cat_names


def _build_coefficients(
    num_names: list[str],
    cat_names: list[str],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    coeffs: dict[str, np.ndarray] = {}
    all_num = list(num_names)
    rng.shuffle(all_num)
    predictive_num = set(all_num[:_NUMERIC_PREDICTIVE])
    for name in num_names:
        if name in predictive_num:
            coeffs[name] = np.array(rng.uniform(-1.5, 1.5, size=1))
        else:
            coeffs[name] = np.array([0.0])

    all_cat = list(cat_names)
    rng.shuffle(all_cat)
    predictive_cat = set(all_cat[:_CATEGORICAL_PREDICTIVE])
    for name in cat_names:
        if name in predictive_cat:
            coeffs[name] = rng.uniform(-1.5, 1.5, size=1)
        else:
            coeffs[name] = np.array([0.0])

    return coeffs


def _encode_categoricals(
    df: pl.DataFrame,
    cat_names: list[str],
) -> tuple[pl.DataFrame, list[str]]:
    encoded_cols: list[str] = []
    df = df.clone()
    for name in cat_names:
        levels = sorted(df[name].unique().to_list())
        n_levels = len(levels)
        if n_levels <= 2:
            continue
        for lvl in levels[1:]:
            encoded_name = f"{name}_{lvl}"
            df = df.with_columns(
                (pl.col(name) == lvl).cast(pl.Int64).alias(encoded_name)
            )
            encoded_cols.append(encoded_name)
    return df, encoded_cols


def _compute_eta(
    df: pl.DataFrame,
    num_names: list[str],
    encoded_cat_names: list[str],
    coeffs: dict[str, np.ndarray],
) -> np.ndarray:
    eta = np.zeros(df.height, dtype=np.float64)
    for name in num_names:
        if name in coeffs:
            eta += df[name].to_numpy() * coeffs[name][0]
    for name in encoded_cat_names:
        if name.startswith("cat_"):
            base = name.rsplit("_", 1)[0]
        else:
            base = name
        if base in coeffs:
            eta += df[name].to_numpy() * coeffs[base][0]
    return eta


def generate_poisson_dataset(
    n_rows: int = 200_000,
    seed: int = 42,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    df, num_names, cat_names = _generate_predictors(rng, n_rows)
    df, encoded_cat_names = _encode_categoricals(df, cat_names)
    coeffs = _build_coefficients(num_names, cat_names, rng)
    coef_intercept = -1.8
    eta = _compute_eta(df, num_names, encoded_cat_names, coeffs)
    exposure = np.exp(rng.normal(0, 0.3, size=n_rows))
    mu = exposure * np.exp(coef_intercept + eta)
    mu = np.clip(mu, 0, 50)
    response = rng.poisson(mu)
    train_test = (rng.random(n_rows) < 0.75).astype(np.int8)
    return _assemble_dataframe(df, num_names, cat_names, exposure, response, train_test)


def generate_gamma_dataset(
    n_rows: int = 200_000,
    seed: int = 42,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    df, num_names, cat_names = _generate_predictors(rng, n_rows)
    df, encoded_cat_names = _encode_categoricals(df, cat_names)
    coeffs = _build_coefficients(num_names, cat_names, rng)
    coef_intercept = 0.5
    eta = _compute_eta(df, num_names, encoded_cat_names, coeffs)
    exposure = np.exp(rng.normal(0, 0.3, size=n_rows))
    mu = exposure * np.exp(coef_intercept + eta)
    mu = np.clip(mu, 0.1, None)
    phi = 1.5
    shape = phi
    scale = mu / phi
    response = rng.gamma(shape, scale)
    train_test = (rng.random(n_rows) < 0.75).astype(np.int8)
    return _assemble_dataframe(df, num_names, cat_names, exposure, response, train_test)


def generate_gaussian_dataset(
    n_rows: int = 200_000,
    seed: int = 42,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    df, num_names, cat_names = _generate_predictors(rng, n_rows)
    df, encoded_cat_names = _encode_categoricals(df, cat_names)
    coeffs = _build_coefficients(num_names, cat_names, rng)
    coef_intercept = 2.0
    eta = _compute_eta(df, num_names, encoded_cat_names, coeffs)
    exposure = np.exp(rng.normal(0, 0.3, size=n_rows))
    sigma = 2.5
    response = rng.normal(coef_intercept + eta, sigma)
    train_test = (rng.random(n_rows) < 0.75).astype(np.int8)
    return _assemble_dataframe(df, num_names, cat_names, exposure, response, train_test)


def generate_binomial_dataset(
    n_rows: int = 200_000,
    seed: int = 42,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    df, num_names, cat_names = _generate_predictors(rng, n_rows)
    df, encoded_cat_names = _encode_categoricals(df, cat_names)
    coeffs = _build_coefficients(num_names, cat_names, rng)
    coef_intercept = -0.3
    eta = _compute_eta(df, num_names, encoded_cat_names, coeffs)
    p = 1.0 / (1.0 + np.exp(-(coef_intercept + eta)))
    p = np.clip(p, 0.001, 0.999)
    exposure = np.exp(rng.normal(0, 0.3, size=n_rows))
    response = rng.binomial(1, p)
    train_test = (rng.random(n_rows) < 0.75).astype(np.int8)
    return _assemble_dataframe(df, num_names, cat_names, exposure, response, train_test)


def _assemble_dataframe(
    predictor_df: pl.DataFrame,
    num_names: list[str],
    cat_names: list[str],
    exposure: np.ndarray,
    response: np.ndarray,
    train_test: np.ndarray,
) -> pl.DataFrame:
    predictor_cols = num_names + cat_names
    out = predictor_df.select(predictor_cols).clone()
    out = out.with_columns(
        pl.Series("Exposure", exposure),
        pl.Series("Response", response),
        pl.Series("traintest", train_test),
    )
    return out


def generate_all_datasets(
    seed: int = 42,
    n_rows: int = 200_000,
) -> dict[str, pl.DataFrame]:
    return {
        "poisson": generate_poisson_dataset(n_rows=n_rows, seed=seed),
        "gamma": generate_gamma_dataset(n_rows=n_rows, seed=seed),
        "gaussian": generate_gaussian_dataset(n_rows=n_rows, seed=seed),
        "binomial": generate_binomial_dataset(n_rows=n_rows, seed=seed),
    }
