"""Competition metrics with strict probability validation."""

from __future__ import annotations

import numpy as np


def _validated_arrays(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=float)
    prob = np.asarray(y_prob, dtype=float)
    if truth.shape != prob.shape:
        raise ValueError(f"shape mismatch: y_true={truth.shape}, y_prob={prob.shape}")
    if truth.ndim != 1 or truth.size == 0:
        raise ValueError("y_true and y_prob must be non-empty 1-D arrays")
    if not np.isfinite(truth).all() or not np.isfinite(prob).all():
        raise ValueError("targets and probabilities must all be finite")
    if not np.isin(truth, [0.0, 1.0]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if ((prob < 0.0) | (prob > 1.0)).any():
        raise ValueError("y_prob must be inside [0, 1]")
    return truth, prob


def brier_score(y_true, y_prob) -> float:
    truth, prob = _validated_arrays(y_true, y_prob)
    return float(np.mean((prob - truth) ** 2))


def competition_score(y_true, y_prob) -> float:
    """Return the published Brier skill score on a labeled validation set."""
    truth, prob = _validated_arrays(y_true, y_prob)
    event_rate = float(truth.mean())
    reference = event_rate * (1.0 - event_rate)
    if reference == 0.0:
        raise ValueError("competition score is undefined when validation has one class")
    return max(0.0, 100000.0 * (1.0 - brier_score(truth, prob) / reference))


def paired_brier_comparison(y_true, baseline_prob, candidate_prob) -> dict[str, float]:
    """Compare candidate-minus-baseline squared error on the same rows."""
    truth, baseline = _validated_arrays(y_true, baseline_prob)
    _, candidate = _validated_arrays(y_true, candidate_prob)
    difference = np.square(candidate - truth) - np.square(baseline - truth)
    standard_error = float(difference.std(ddof=1) / np.sqrt(len(difference)))
    mean = float(difference.mean())
    return {
        "paired_brier_delta": mean,
        "paired_standard_error": standard_error,
        "paired_ci95_low": mean - 1.96 * standard_error,
        "paired_ci95_high": mean + 1.96 * standard_error,
    }
