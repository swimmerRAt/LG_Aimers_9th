#!/usr/bin/env python3
"""Train and validate the optimized, submission-safe probability ensemble."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from model.ensemble import OptimizedBaseballEnsemble
from src.lg_aimers import ID_COL, TARGET_COL
from src.lg_aimers.metrics import brier_score, competition_score


# Removed after 2024 permutation diagnostics or because an integer identifier has
# no meaningful numeric ordering. These are fixed before looking at evaluation data.
EXCLUDED_FEATURES = {
    "pitcher_id",
    "batter_id",
    "score_diff_pitcher_team",
    "away_win_expectancy",
    "run_bot_before",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "game_month",
    "asof_pitcher_middle_rate",
    "score_diff_home",
    "num_runners_on",
    "run_top_before",
    "run_total_before",
    "asof_pitcher_pitchmix_n",
}


def positive_probability(model, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(frame)[:, 1], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--hist-weight", type=float, default=0.45)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/optimized_ensemble_2024"))
    parser.add_argument("--model-output", type=Path, default=Path("model/final_model.pkl"))
    args = parser.parse_args()

    test_columns = pd.read_csv(args.data_dir / "test.csv", nrows=0).columns.tolist()
    feature_columns = [
        column for column in test_columns
        if column != ID_COL and column not in EXCLUDED_FEATURES
    ]
    train = pd.read_csv(
        args.data_dir / "train.csv",
        usecols=[ID_COL, TARGET_COL, *feature_columns],
    )
    train_mask = train["season"] < args.validation_season
    validation_mask = train["season"] == args.validation_season
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("forward validation split is empty")

    validation_model = OptimizedBaseballEnsemble(
        hist_weight=args.hist_weight,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )
    started = time.perf_counter()
    validation_model.fit(
        train.loc[train_mask, feature_columns],
        train.loc[train_mask, TARGET_COL],
    )
    fit_seconds = time.perf_counter() - started
    infer_started = time.perf_counter()
    predictions = positive_probability(
        validation_model,
        train.loc[validation_mask, feature_columns],
    )
    inference_seconds = time.perf_counter() - infer_started
    truth = train.loc[validation_mask, TARGET_COL].to_numpy()
    brier = brier_score(truth, predictions)
    score = competition_score(truth, predictions)
    slope, intercept = np.linalg.lstsq(
        np.column_stack([predictions, np.ones(len(predictions))]),
        truth,
        rcond=None,
    )[0]
    calibrated = np.clip(slope * predictions + intercept, 0.0, 1.0)
    metrics = {
        "model": "optimized_ensemble",
        "validation_season": args.validation_season,
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "validation_target_rate": float(truth.mean()),
        "prediction_mean": float(predictions.mean()),
        "brier": brier,
        "competition_score": score,
        "diagnostic_optimal_slope": float(slope),
        "diagnostic_optimal_intercept": float(intercept),
        "diagnostic_calibrated_brier": brier_score(truth, calibrated),
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
    }
    print(json.dumps(metrics, ensure_ascii=False), flush=True)

    final_model = OptimizedBaseballEnsemble(
        hist_weight=args.hist_weight,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )
    final_started = time.perf_counter()
    final_model.fit(train[feature_columns], train[TARGET_COL])
    final_fit_seconds = time.perf_counter() - final_started
    versions = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    artifact = {
        "model": final_model,
        "feature_columns": feature_columns,
        "positive_class": 1,
        "selected_model": "optimized_ensemble",
        "validation_seasons": [args.validation_season],
        "full_train_rows": len(train),
        "full_train_target_rate": float(train[TARGET_COL].mean()),
        "random_state": args.random_state,
        "versions": versions,
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_output, compress=3)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(args.artifact_dir / "metrics.csv", index=False)
    pd.DataFrame({
        ID_COL: train.loc[validation_mask, ID_COL].to_numpy(),
        TARGET_COL: truth,
        "prediction": predictions,
    }).to_csv(args.artifact_dir / "oof_predictions.csv", index=False)
    (args.artifact_dir / "run_summary.json").write_text(
        json.dumps({
            **metrics,
            "hist_weight": args.hist_weight,
            "extra_weight": 1.0 - args.hist_weight,
            "n_estimators": args.n_estimators,
            "feature_columns": feature_columns,
            "excluded_features": sorted(EXCLUDED_FEATURES),
            "final_fit_seconds": final_fit_seconds,
            "model_output": str(args.model_output),
            "versions": versions,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved {args.model_output} | full fit={final_fit_seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
