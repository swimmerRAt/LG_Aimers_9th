#!/usr/bin/env python3
"""Create a submission model candidate with one additional global logit shift."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def logit(probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-9, 1.0 - 1e-9)
    return np.log(probability / (1.0 - probability))


def positive_probability(model, features: pd.DataFrame) -> np.ndarray:
    classes = np.asarray(model.classes_)
    position = np.flatnonzero(classes == 1)
    if len(position) != 1:
        raise ValueError("model must expose positive class 1 exactly once")
    return np.asarray(model.predict_proba(features)[:, int(position[0])], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("model/final_model.pkl"))
    parser.add_argument("--shift", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    args = parser.parse_args()

    if not np.isfinite(args.shift) or abs(args.shift) > 1.0:
        raise ValueError("shift must be finite and no larger than 1.0 in magnitude")
    source = joblib.load(args.source)
    if not isinstance(source, dict) or "model" not in source or "feature_columns" not in source:
        raise ValueError("source must be a model artifact dictionary")
    base_model = source["model"]
    calibrator = getattr(base_model, "calibrator", None)
    if calibrator is None or not hasattr(calibrator, "intercept_"):
        raise ValueError("source model does not expose a fitted global logit calibrator")

    candidate = copy.deepcopy(source)
    candidate_model = candidate["model"]
    original_intercept = float(candidate_model.calibrator.intercept_)
    candidate_model.calibrator.intercept_ = original_intercept + float(args.shift)
    candidate["selected_model"] = (
        f"{source.get('selected_model', type(base_model).__name__)}_logit_shift"
    )
    candidate["leaderboard_logit_shift"] = float(args.shift)
    candidate["source_model"] = str(args.source)
    candidate["source_calibration_intercept"] = original_intercept
    candidate["candidate_calibration_intercept"] = float(
        candidate_model.calibrator.intercept_
    )

    test = pd.read_csv(args.test)
    feature_columns = list(candidate["feature_columns"])
    features = test.loc[:, feature_columns]
    base_probability = positive_probability(base_model, features)
    candidate_probability = positive_probability(candidate_model, features)
    observed_shift = logit(candidate_probability) - logit(base_probability)
    np.testing.assert_allclose(observed_shift, args.shift, atol=1e-10, rtol=0.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate, args.output, compress=3)
    loaded = joblib.load(args.output)
    reloaded_probability = positive_probability(loaded["model"], features)
    np.testing.assert_allclose(reloaded_probability, candidate_probability, atol=1e-12)
    print(
        f"saved {args.output} | shift={args.shift:+.6f} | "
        f"base_mean={base_probability.mean():.8f} | "
        f"candidate_mean={candidate_probability.mean():.8f}"
    )


if __name__ == "__main__":
    main()
