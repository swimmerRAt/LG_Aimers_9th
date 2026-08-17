"""Utilities for time-aware non-negative probability stacking."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.lg_aimers.metrics import brier_score


def apply_logit_shift(probability, shift: float) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-9, 1.0 - 1e-9)
    logit = np.log(probability / (1.0 - probability))
    return 1.0 / (1.0 + np.exp(-(logit + float(shift))))


def normalized_fold_brier(truth, prediction) -> float:
    truth = np.asarray(truth, dtype=float)
    event_rate = float(truth.mean())
    reference = event_rate * (1.0 - event_rate)
    if reference <= 0.0:
        raise ValueError("fold must contain both target classes")
    return brier_score(truth, prediction) / reference


def robust_stack_objective(
    truth,
    prediction_matrix,
    seasons,
    weights,
    final_logit_shift: float,
    season_weights=(0.4, 0.6),
    stability_penalty: float = 0.10,
) -> float:
    truth = np.asarray(truth, dtype=float)
    matrix = np.asarray(prediction_matrix, dtype=float)
    seasons = np.asarray(seasons, dtype=int)
    weights = np.asarray(weights, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(truth):
        raise ValueError("prediction matrix must have one row per target")
    if matrix.shape[1] != len(weights):
        raise ValueError("stack weight count must match prediction columns")
    if (weights < 0.0).any() or not np.isclose(weights.sum(), 1.0, atol=1e-7):
        raise ValueError("stack weights must be non-negative and sum to one")
    unique_seasons = sorted(int(value) for value in np.unique(seasons))
    fold_weights = np.asarray(season_weights[-len(unique_seasons):], dtype=float)
    if len(fold_weights) != len(unique_seasons):
        raise ValueError("not enough season weights")
    fold_weights /= fold_weights.sum()
    shifted = apply_logit_shift(matrix @ weights, final_logit_shift)
    losses = np.asarray(
        [
            normalized_fold_brier(truth[seasons == season], shifted[seasons == season])
            for season in unique_seasons
        ],
        dtype=float,
    )
    mean = float(np.dot(fold_weights, losses))
    variance = float(np.dot(fold_weights, np.square(losses - mean)))
    return mean + float(stability_penalty) * np.sqrt(variance)


def constrained_stack_weights(
    official_weight: float,
    catboost_fraction_of_remainder: float,
    minimum_official_weight: float = 0.5,
) -> np.ndarray:
    """Map two bounded Optuna parameters to valid three-branch stack weights."""
    official = float(official_weight)
    catboost_fraction = float(catboost_fraction_of_remainder)
    minimum = float(minimum_official_weight)
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("minimum_official_weight must be inside [0, 1]")
    if not minimum <= official <= 1.0:
        raise ValueError("official_weight must satisfy the configured minimum")
    if not 0.0 <= catboost_fraction <= 1.0:
        raise ValueError("catboost_fraction_of_remainder must be inside [0, 1]")
    remainder = 1.0 - official
    weights = np.asarray(
        [
            official,
            remainder * catboost_fraction,
            remainder * (1.0 - catboost_fraction),
        ],
        dtype=float,
    )
    weights /= weights.sum()
    return weights


def select_simplex_stack_weights(
    truth,
    prediction_matrix,
    seasons,
    final_logit_shift: float,
    season_weights=(0.4, 0.6),
    stability_penalty: float = 0.10,
) -> tuple[np.ndarray, dict]:
    """Select robust non-negative weights, explicitly considering pure models."""
    matrix = np.asarray(prediction_matrix, dtype=float)
    n_models = matrix.shape[1]

    def evaluate(values) -> float:
        clipped = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
        clipped /= clipped.sum()
        return robust_stack_objective(
            truth,
            matrix,
            seasons,
            clipped,
            final_logit_shift,
            season_weights,
            stability_penalty,
        )

    starts = [np.full(n_models, 1.0 / n_models)]
    starts.extend(np.eye(n_models))
    solutions = []
    for start in starts:
        result = minimize(
            evaluate,
            start,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_models,
            constraints={"type": "eq", "fun": lambda values: float(np.sum(values) - 1.0)},
            options={"ftol": 1e-14, "maxiter": 1000},
        )
        if result.success:
            values = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
            values /= values.sum()
            solutions.append((evaluate(values), values, result))
    for start in np.eye(n_models):
        solutions.append((evaluate(start), start.copy(), None))
    if not solutions:
        raise RuntimeError("stack optimizer produced no valid solution")
    objective, weights, result = min(solutions, key=lambda item: item[0])
    diagnostics = {
        "optimizer": "multi_start_SLSQP",
        "objective": float(objective),
        "success": result is None or bool(result.success),
        "message": "simplex endpoint" if result is None else str(result.message),
        "iterations": 0 if result is None else int(result.nit),
    }
    return weights, diagnostics


def error_correlation_long(
    truth,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Return signed-error and squared-error correlations for every model pair."""
    truth = np.asarray(truth, dtype=float)
    names = list(predictions)
    signed = pd.DataFrame(
        {name: np.asarray(predictions[name], dtype=float) - truth for name in names}
    ).corr()
    squared = pd.DataFrame(
        {
            name: np.square(np.asarray(predictions[name], dtype=float) - truth)
            for name in names
        }
    ).corr()
    rows = []
    for left in names:
        for right in names:
            rows.append(
                {
                    "model_left": left,
                    "model_right": right,
                    "signed_error_correlation": float(signed.loc[left, right]),
                    "squared_error_correlation": float(squared.loc[left, right]),
                }
            )
    return pd.DataFrame(rows)
