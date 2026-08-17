#!/usr/bin/env python3
"""Compare the current and Optuna candidate on untouched forward seasons."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss

from optimize_hyperparameters import competition_score, make_preprocessor
from train_optimized import EXCLUDED_FEATURES


ID_COL = "row_id"
TARGET_COL = "control_success"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--params-path",
        type=Path,
        default=Path("artifacts/optuna_2024/best_params.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/optuna_2024/temporal_validation.csv"),
    )
    parser.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tuned = json.loads(args.params_path.read_text(encoding="utf-8"))
    hist_params = tuned["histgb"]["params"]
    tuned_extra_params = dict(tuned["extra_trees"]["params"])
    tuned_extra_params.pop("n_estimators", None)
    n_estimators = int(tuned["extra_trees"]["params"]["n_estimators"])
    tuned_raw_weight = float(tuned["raw_ensemble"]["hist_weight"])
    calibrated_params = tuned["calibrated_ensemble"]["params"]

    test_columns = pd.read_csv(args.data_dir / "test.csv", nrows=0).columns.tolist()
    feature_columns = [
        column
        for column in test_columns
        if column != ID_COL and column not in EXCLUDED_FEATURES
    ]
    train = pd.read_csv(
        args.data_dir / "train.csv",
        usecols=[TARGET_COL, *feature_columns],
    )

    current_extra_params = {
        "max_depth": 16,
        "min_samples_leaf": 100,
        "max_features": 0.8,
        "criterion": "gini",
    }
    rows: list[dict] = []
    for season in args.seasons:
        train_mask = train["season"] < season
        validation_mask = train["season"] == season
        y_train = train.loc[train_mask, TARGET_COL].to_numpy()
        y_validation = train.loc[validation_mask, TARGET_COL].to_numpy()
        if not len(y_train) or not len(y_validation):
            raise ValueError(f"empty forward split for season {season}")

        preprocessor = make_preprocessor(feature_columns)
        X_train = preprocessor.fit_transform(
            train.loc[train_mask, feature_columns], y_train
        )
        X_validation = preprocessor.transform(
            train.loc[validation_mask, feature_columns]
        )

        started = time.perf_counter()
        hist_model = HistGradientBoostingClassifier(
            **hist_params,
            early_stopping=False,
            random_state=args.seed,
        ).fit(X_train, y_train)
        hist_prediction = hist_model.predict_proba(X_validation)[:, 1]

        current_extra = ExtraTreesClassifier(
            **current_extra_params,
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=args.seed,
        ).fit(X_train, y_train)
        current_extra_prediction = current_extra.predict_proba(X_validation)[:, 1]

        tuned_extra = ExtraTreesClassifier(
            **tuned_extra_params,
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=args.seed,
        ).fit(X_train, y_train)
        tuned_extra_prediction = tuned_extra.predict_proba(X_validation)[:, 1]
        fit_seconds = time.perf_counter() - started

        predictions = {
            "current_raw": 0.45 * hist_prediction + 0.55 * current_extra_prediction,
            "optuna_raw": (
                tuned_raw_weight * hist_prediction
                + (1.0 - tuned_raw_weight) * tuned_extra_prediction
            ),
        }
        calibrated_raw = (
            float(calibrated_params["hist_weight"]) * hist_prediction
            + (1.0 - float(calibrated_params["hist_weight"]))
            * tuned_extra_prediction
        )
        predictions["optuna_2024_calibrated"] = np.clip(
            float(calibrated_params["calibration_slope"]) * calibrated_raw
            + float(calibrated_params["calibration_intercept"]),
            0.0,
            1.0,
        )

        for name, prediction in predictions.items():
            rows.append({
                "validation_season": season,
                "candidate": name,
                "train_rows": int(train_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "target_rate": float(y_validation.mean()),
                "prediction_mean": float(prediction.mean()),
                "brier": float(brier_score_loss(y_validation, prediction)),
                "competition_score": competition_score(y_validation, prediction),
                "shared_fit_seconds": fit_seconds,
            })
        print(
            pd.DataFrame(rows).query("validation_season == @season").to_string(index=False),
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
