#!/usr/bin/env python3
"""Fit a quadratic score curve to submitted global logit shifts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def fit_quadratic_vertex(shifts, scores) -> dict[str, float]:
    shifts = np.asarray(shifts, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if shifts.ndim != 1 or scores.shape != shifts.shape or len(shifts) < 3:
        raise ValueError("at least three equal-length shift and score values are required")
    if not np.isfinite(shifts).all() or not np.isfinite(scores).all():
        raise ValueError("shift and score values must be finite")
    if len(np.unique(shifts)) < 3:
        raise ValueError("at least three distinct shifts are required")
    quadratic, linear, intercept = np.polyfit(shifts, scores, 2)
    if quadratic >= 0.0:
        raise ValueError("fitted score curve has no finite maximum")
    vertex = -linear / (2.0 * quadratic)
    estimated_score = np.polyval([quadratic, linear, intercept], vertex)
    return {
        "quadratic": float(quadratic),
        "linear": float(linear),
        "intercept": float(intercept),
        "optimal_logit_shift": float(vertex),
        "estimated_optimal_score": float(estimated_score),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("artifacts/leaderboard_logit_shift/scores.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/leaderboard_logit_shift/curve_fit.json"),
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.scores)
    submitted = frame.dropna(subset=["official_score"])
    result = fit_quadratic_vertex(
        submitted["logit_shift"], submitted["official_score"]
    )
    result["submitted_points"] = submitted[
        ["logit_shift", "official_score"]
    ].to_dict(orient="records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
