from __future__ import annotations

import numpy as np


def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 1e-9, None)
    y_true_safe = np.clip(y_true, 1e-9, None)
    dev = y_true_safe * np.log(y_true_safe / y_pred) - (y_true_safe - y_pred)
    dev = np.where(y_true == 0, y_pred, dev)
    return float(np.mean(dev) * 2)


def gamma_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 1e-9, None)
    dev = (y_true - y_pred) / y_pred - np.log(y_true / y_pred)
    return float(np.mean(dev) * 2)


def gaussian_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def binomial_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    dev = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    return float(-np.mean(dev) * 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


DEV_FUNCTIONS: dict[str, callable] = {
    "poisson": poisson_deviance,
    "gamma": gamma_deviance,
    "gaussian": gaussian_deviance,
    "binomial": binomial_deviance,
}


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    family: str,
) -> dict[str, float]:
    dev_fn = DEV_FUNCTIONS[family.lower()]
    return {
        "deviance": dev_fn(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
    }
