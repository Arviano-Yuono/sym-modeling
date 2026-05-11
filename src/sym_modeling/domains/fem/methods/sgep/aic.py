from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    rss: float
    rmse: float
    aic: float
    aicc: float
    num_samples: int
    num_parameters: int


def residual_sum_squares(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    error = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sum(np.square(error)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    error = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    if error.size == 0:
        return float("inf")
    return float(np.sqrt(np.mean(np.square(error))))


def aic(rss: float, num_samples: int, num_parameters: int) -> float:
    if num_samples <= 0:
        return float("inf")
    safe_rss = max(float(rss), np.finfo(float).tiny)
    return float(2 * num_parameters + num_samples * np.log(safe_rss / num_samples))


def aicc(rss: float, num_samples: int, num_parameters: int) -> float:
    base = aic(rss, num_samples, num_parameters)
    if not np.isfinite(base):
        return float("inf")
    if num_samples <= num_parameters + 1:
        return float("inf")
    correction = (2 * num_parameters * (num_parameters + 1)) / (
        num_samples - num_parameters - 1
    )
    return float(base + correction)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_parameters: int,
) -> RegressionMetrics:
    sample_count = int(np.asarray(y_true).size)
    rss_value = residual_sum_squares(y_true, y_pred)
    return RegressionMetrics(
        rss=rss_value,
        rmse=rmse(y_true, y_pred),
        aic=aic(rss_value, sample_count, num_parameters),
        aicc=aicc(rss_value, sample_count, num_parameters),
        num_samples=sample_count,
        num_parameters=int(num_parameters),
    )
