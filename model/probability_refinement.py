"""Leakage-safe probability refinements for temporal OOF predictions."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def _validated_inputs(probability, target, sample_weight=None):
    probability = np.asarray(probability, dtype=float)
    target = np.asarray(target, dtype=float)
    if probability.ndim != 1 or probability.shape != target.shape or probability.size == 0:
        raise ValueError("probability and target must be non-empty equal-length vectors")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("probability must be finite and inside [0, 1]")
    if not np.isin(target, [0.0, 1.0]).all():
        raise ValueError("target must contain only 0 and 1")
    if sample_weight is None:
        weight = np.ones(len(target), dtype=float)
    else:
        weight = np.asarray(sample_weight, dtype=float)
        if weight.shape != target.shape or not np.isfinite(weight).all() or (weight < 0).any():
            raise ValueError("sample_weight must be finite, non-negative, and match target")
        if weight.sum() <= 0:
            raise ValueError("sample_weight must have a positive sum")
    return probability, target, weight


def _logit(probability):
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(value):
    value = np.asarray(value, dtype=float)
    positive = value >= 0
    result = np.empty_like(value)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def fitted_logit_intercept(probability, target, sample_weight=None) -> float:
    """Find the intercept that aligns weighted predicted and observed event rates."""
    probability, target, weight = _validated_inputs(probability, target, sample_weight)
    logits = _logit(probability)
    observed = float(np.average(target, weights=weight))
    if observed <= 0.0:
        return -20.0
    if observed >= 1.0:
        return 20.0

    def difference(intercept):
        return float(np.average(_sigmoid(logits + intercept), weights=weight) - observed)

    return float(brentq(difference, -20.0, 20.0))


class GameTypeLogitAdjuster:
    """Shrink game-type intercepts toward the overall OOF intercept."""

    def __init__(self, strength: float = 0.10, shrinkage: float = 100_000.0):
        self.strength = strength
        self.shrinkage = shrinkage

    def fit(self, probability, target, groups, sample_weight=None):
        probability, target, weight = _validated_inputs(
            probability, target, sample_weight
        )
        groups = np.asarray(groups, dtype=object)
        if groups.shape != target.shape:
            raise ValueError("groups must match target")
        if not 0.0 <= float(self.strength) <= 1.0:
            raise ValueError("strength must be inside [0, 1]")
        if float(self.shrinkage) < 0.0:
            raise ValueError("shrinkage must be non-negative")

        global_raw = fitted_logit_intercept(probability, target, weight)
        self.global_intercept_ = float(self.strength) * global_raw
        self.group_intercepts_ = {}
        for group in np.unique(groups):
            mask = groups == group
            group_raw = fitted_logit_intercept(
                probability[mask], target[mask], weight[mask]
            )
            effective_n = float(weight[mask].sum())
            reliability = effective_n / (effective_n + float(self.shrinkage))
            shrunk = global_raw + reliability * (group_raw - global_raw)
            self.group_intercepts_[group] = float(self.strength) * shrunk
        return self

    def transform(self, probability, groups):
        if not hasattr(self, "group_intercepts_"):
            raise ValueError("GameTypeLogitAdjuster must be fitted before transform")
        probability = np.asarray(probability, dtype=float)
        groups = np.asarray(groups, dtype=object)
        if probability.shape != groups.shape:
            raise ValueError("probability and groups must have equal shapes")
        intercept = np.asarray(
            [self.group_intercepts_.get(group, self.global_intercept_) for group in groups],
            dtype=float,
        )
        return _sigmoid(_logit(probability) + intercept)


class LogitInterceptCalibrator:
    """Conservatively correct the overall probability level on the logit scale."""

    def __init__(self, strength: float = 0.25):
        self.strength = strength

    def fit(self, probability, target, sample_weight=None):
        if not 0.0 <= float(self.strength) <= 1.0:
            raise ValueError("strength must be inside [0, 1]")
        raw = fitted_logit_intercept(probability, target, sample_weight)
        self.intercept_ = float(self.strength) * raw
        return self

    def transform(self, probability):
        if not hasattr(self, "intercept_"):
            raise ValueError("LogitInterceptCalibrator must be fitted before transform")
        return _sigmoid(_logit(probability) + self.intercept_)
