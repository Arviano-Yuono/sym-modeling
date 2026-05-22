from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sym_modeling.common.regression import SparseRegressionModel


@dataclass(frozen=True)
class RegressionMetrics:
    rss: float
    rmse: float
    aic: float
    aicc: float
    num_samples: int
    num_parameters: int


@dataclass(frozen=True)
class SparseFitResult:
    theta: np.ndarray
    prediction: np.ndarray
    active_mask: np.ndarray
    metrics: RegressionMetrics
    column_scales: np.ndarray


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
    if not np.isfinite(base) or num_samples <= num_parameters + 1:
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


def _column_scales(X: np.ndarray) -> np.ndarray:
    scales = np.linalg.norm(X, axis=0) / max(np.sqrt(max(X.shape[0], 1)), 1.0)
    scales[~np.isfinite(scales)] = 0.0
    scales[scales < 1e-14] = 0.0
    return scales


def _least_squares(X: np.ndarray, y: np.ndarray, ridge_alpha: float = 0.0) -> np.ndarray:
    if X.shape[1] == 0:
        return np.zeros(0, dtype=float)
    if ridge_alpha > 0.0:
        lhs = X.T @ X + ridge_alpha * np.eye(X.shape[1])
        rhs = X.T @ y
        try:
            return np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _stlsq(
    X_scaled: np.ndarray,
    y: np.ndarray,
    threshold: float,
    ridge_alpha: float,
    max_iter: int,
) -> np.ndarray:
    coef = _least_squares(X_scaled, y, ridge_alpha)
    active = np.abs(coef) >= threshold
    for _ in range(max_iter):
        if not np.any(active):
            return np.zeros_like(coef)
        next_coef = np.zeros_like(coef)
        next_coef[active] = _least_squares(X_scaled[:, active], y, ridge_alpha)
        next_active = np.abs(next_coef) >= threshold
        if np.array_equal(active, next_active):
            return next_coef
        active = next_active
        coef = next_coef
    return coef


def fit_sparse_regression(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "stlsq",
    alpha: float = 1e-8,
    threshold: float = 1e-8,
    refit: bool = True,
    max_iter: int = 10,
) -> SparseFitResult:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have incompatible sample counts.")

    if X.shape[1] == 0:
        prediction = np.zeros_like(y)
        metrics = regression_metrics(y, prediction, num_parameters=0)
        return SparseFitResult(
            theta=np.zeros(0, dtype=float),
            prediction=prediction,
            active_mask=np.zeros(0, dtype=bool),
            metrics=metrics,
            column_scales=np.zeros(0, dtype=float),
        )

    scales = _column_scales(X)
    usable = scales > 0.0
    X_scaled = np.zeros_like(X)
    X_scaled[:, usable] = X[:, usable] / scales[usable]

    coef_scaled = np.zeros(X.shape[1], dtype=float)
    if np.any(usable):
        method_key = method.lower()
        if method_key == "stlsq":
            coef_scaled[usable] = _stlsq(
                X_scaled[:, usable],
                y,
                threshold=threshold,
                ridge_alpha=alpha,
                max_iter=max_iter,
            )
        elif method_key in {"lasso", "ridge", "elasticnet"}:
            kwargs = {"alpha": alpha, "fit_intercept": False}
            if method_key in {"lasso", "elasticnet"}:
                kwargs["max_iter"] = max(1000, 200 * max_iter)
            model = SparseRegressionModel(model_type=method_key, **kwargs).fit(
                X_scaled[:, usable],
                y,
            )
            coef_scaled[usable] = np.asarray(model.model.coef_, dtype=float).reshape(-1)
        elif method_key == "ridge_threshold":
            model = SparseRegressionModel(
                model_type="ridge",
                alpha=alpha,
                fit_intercept=False,
            ).fit(X_scaled[:, usable], y)
            coef_scaled[usable] = np.asarray(model.model.coef_, dtype=float).reshape(-1)
        else:
            raise ValueError("Unsupported sparse regression method: %s" % method)

    theta = np.zeros_like(coef_scaled)
    theta[usable] = coef_scaled[usable] / scales[usable]
    active = np.abs(theta) >= threshold

    if refit and np.any(active):
        theta_refit = np.zeros_like(theta)
        theta_refit[active] = _least_squares(X[:, active], y, ridge_alpha=0.0)
        theta = theta_refit
        active = np.abs(theta) >= threshold

    prediction = X @ theta
    metrics = regression_metrics(y, prediction, num_parameters=int(np.count_nonzero(active)))
    return SparseFitResult(
        theta=theta,
        prediction=prediction,
        active_mask=active,
        metrics=metrics,
        column_scales=scales,
    )
