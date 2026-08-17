#!/usr/bin/env python3
"""Optuna search for forward-validation Brier score.

Optuna is a local tuning dependency only. The saved DACON inference artifact still
uses only scikit-learn, pandas, NumPy, and joblib.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from train_optimized import EXCLUDED_FEATURES


ID_COL = "row_id"
TARGET_COL = "control_success"
CATEGORICAL = ("top_bottom", "game_type", "base_state")


def competition_score(y_true: np.ndarray, prediction: np.ndarray) -> float:
    rate = float(np.mean(y_true))
    reference = rate * (1.0 - rate)
    return max(0.0, 100000.0 * (1.0 - brier_score_loss(y_true, prediction) / reference))


def trial_frame(study: optuna.Study) -> pd.DataFrame:
    frame = study.trials_dataframe(
        attrs=("number", "value", "params", "state", "duration", "user_attrs")
    )
    return frame.sort_values("number").reset_index(drop=True)


def completed_trials(study: optuna.Study) -> int:
    return sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)


def remaining_trials(study: optuna.Study, requested_total: int) -> int:
    return max(0, int(requested_total) - completed_trials(study))


def make_study(name: str, storage: str, seed: int) -> optuna.Study:
    return optuna.create_study(
        study_name=name,
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
    )


def make_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    categorical = [column for column in CATEGORICAL if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]
    return ColumnTransformer([
        (
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "encode",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
            ]),
            categorical,
        ),
        ("num", SimpleImputer(strategy="median"), numeric),
    ])


def hist_params_from_trial(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.09, log=True),
        "max_iter": trial.suggest_int("max_iter", 150, 500, step=50),
        "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes", [7, 15, 23, 31, 47, 63]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 80, 1000, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.1, 30.0, log=True),
        "max_bins": trial.suggest_categorical("max_bins", [127, 255]),
    }


def extra_params_from_trial(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 10, 22),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 40, 400, log=True),
        "max_features": trial.suggest_float("max_features", 0.4, 1.0, step=0.1),
        "criterion": trial.suggest_categorical("criterion", ["gini", "log_loss"]),
    }


def fit_hist(params: dict, X_train, y_train, X_validation, seed: int) -> tuple[object, np.ndarray, float]:
    model = HistGradientBoostingClassifier(
        **params,
        early_stopping=False,
        random_state=seed,
    )
    started = time.perf_counter()
    model.fit(X_train, y_train)
    prediction = model.predict_proba(X_validation)[:, 1]
    return model, prediction, time.perf_counter() - started


def fit_extra(
    params: dict,
    X_train,
    y_train,
    X_validation,
    n_estimators: int,
    seed: int,
) -> tuple[object, np.ndarray, float]:
    model = ExtraTreesClassifier(
        **params,
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=seed,
    )
    started = time.perf_counter()
    model.fit(X_train, y_train)
    prediction = model.predict_proba(X_validation)[:, 1]
    return model, prediction, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/optuna_2024"))
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--hist-trials", type=int, default=20)
    parser.add_argument("--extra-trials", type=int, default=10)
    parser.add_argument("--ensemble-trials", type=int, default=200)
    parser.add_argument("--extra-search-estimators", type=int, default=60)
    parser.add_argument("--extra-final-estimators", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

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
    y_train = train.loc[train_mask, TARGET_COL].to_numpy()
    y_validation = train.loc[validation_mask, TARGET_COL].to_numpy()
    validation_ids = train.loc[validation_mask, ID_COL].to_numpy()

    preprocessor = make_preprocessor(feature_columns)
    preprocess_started = time.perf_counter()
    X_train = preprocessor.fit_transform(train.loc[train_mask, feature_columns], y_train)
    X_validation = preprocessor.transform(train.loc[validation_mask, feature_columns])
    preprocess_seconds = time.perf_counter() - preprocess_started
    print(
        json.dumps({
            "train_shape": X_train.shape,
            "validation_shape": X_validation.shape,
            "validation_target_rate": float(y_validation.mean()),
            "preprocess_seconds": preprocess_seconds,
        }),
        flush=True,
    )
    del train
    gc.collect()

    database = (args.artifact_dir / "optuna.db").resolve()
    storage = f"sqlite:///{database}"

    hist_study = make_study(
        f"histgb_forward_{args.validation_season}", storage, args.seed
    )
    if not hist_study.trials:
        hist_study.enqueue_trial({
            "learning_rate": 0.04,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 200,
            "l2_regularization": 5.0,
            "max_bins": 255,
        })

    def hist_objective(trial: optuna.Trial) -> float:
        params = hist_params_from_trial(trial)
        model, prediction, seconds = fit_hist(
            params, X_train, y_train, X_validation, args.seed
        )
        brier = brier_score_loss(y_validation, prediction)
        trial.set_user_attr("competition_score", competition_score(y_validation, prediction))
        trial.set_user_attr("prediction_mean", float(prediction.mean()))
        trial.set_user_attr("fit_seconds", seconds)
        del model, prediction
        gc.collect()
        return brier

    todo = remaining_trials(hist_study, args.hist_trials)
    if todo:
        hist_study.optimize(hist_objective, n_trials=todo, gc_after_trial=True)
    trial_frame(hist_study).to_csv(args.artifact_dir / "histgb_trials.csv", index=False)
    print("best HistGB", hist_study.best_value, hist_study.best_params, flush=True)

    extra_study = make_study(
        f"extra_trees_forward_{args.validation_season}", storage, args.seed + 1
    )
    if not extra_study.trials:
        extra_study.enqueue_trial({
            "max_depth": 16,
            "min_samples_leaf": 100,
            "max_features": 0.8,
            "criterion": "gini",
        })

    def extra_objective(trial: optuna.Trial) -> float:
        params = extra_params_from_trial(trial)
        model, prediction, seconds = fit_extra(
            params,
            X_train,
            y_train,
            X_validation,
            args.extra_search_estimators,
            args.seed,
        )
        brier = brier_score_loss(y_validation, prediction)
        trial.set_user_attr("competition_score", competition_score(y_validation, prediction))
        trial.set_user_attr("prediction_mean", float(prediction.mean()))
        trial.set_user_attr("fit_seconds", seconds)
        del model, prediction
        gc.collect()
        return brier

    todo = remaining_trials(extra_study, args.extra_trials)
    if todo:
        extra_study.optimize(extra_objective, n_trials=todo, gc_after_trial=True)
    trial_frame(extra_study).to_csv(args.artifact_dir / "extra_trees_trials.csv", index=False)
    print("best ExtraTrees", extra_study.best_value, extra_study.best_params, flush=True)

    best_hist_model, hist_prediction, hist_seconds = fit_hist(
        hist_study.best_params, X_train, y_train, X_validation, args.seed
    )
    best_extra_model, extra_prediction, extra_seconds = fit_extra(
        extra_study.best_params,
        X_train,
        y_train,
        X_validation,
        args.extra_final_estimators,
        args.seed,
    )

    # A resumed ExtraTrees/HistGB search can change the winning base models.
    # Isolate blend trials by their base-model signature so objectives from
    # different prediction arrays are never mixed in the same study.
    base_signature = hashlib.sha256(
        json.dumps(
            {
                "histgb": hist_study.best_params,
                "extra_trees": extra_study.best_params,
                "extra_final_estimators": args.extra_final_estimators,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:10]
    ensemble_study = make_study(
        f"ensemble_calibration_forward_{args.validation_season}_{base_signature}",
        storage,
        args.seed + 2,
    )
    if not ensemble_study.trials:
        ensemble_study.enqueue_trial({
            "hist_weight": 0.45,
            "calibration_slope": 1.0,
            "calibration_intercept": 0.0,
        })

    def ensemble_objective(trial: optuna.Trial) -> float:
        hist_weight = trial.suggest_float("hist_weight", 0.0, 1.0)
        slope = trial.suggest_float("calibration_slope", 0.85, 1.25)
        intercept = trial.suggest_float("calibration_intercept", -0.10, 0.05)
        raw = hist_weight * hist_prediction + (1.0 - hist_weight) * extra_prediction
        prediction = np.clip(slope * raw + intercept, 0.0, 1.0)
        brier = brier_score_loss(y_validation, prediction)
        trial.set_user_attr("competition_score", competition_score(y_validation, prediction))
        trial.set_user_attr("prediction_mean", float(prediction.mean()))
        return brier

    todo = remaining_trials(ensemble_study, args.ensemble_trials)
    if todo:
        ensemble_study.optimize(ensemble_objective, n_trials=todo, gc_after_trial=False)
    trial_frame(ensemble_study).to_csv(args.artifact_dir / "ensemble_trials.csv", index=False)

    # Also optimize the raw blend alone so calibration gains remain explicit.
    raw_weights = np.linspace(0.0, 1.0, 1001)
    raw_briers = np.asarray([
        brier_score_loss(
            y_validation,
            weight * hist_prediction + (1.0 - weight) * extra_prediction,
        )
        for weight in raw_weights
    ])
    raw_index = int(np.argmin(raw_briers))
    raw_weight = float(raw_weights[raw_index])
    raw_prediction = raw_weight * hist_prediction + (1.0 - raw_weight) * extra_prediction

    best_ensemble_params = ensemble_study.best_params
    calibrated_raw = (
        best_ensemble_params["hist_weight"] * hist_prediction
        + (1.0 - best_ensemble_params["hist_weight"]) * extra_prediction
    )
    calibrated_prediction = np.clip(
        best_ensemble_params["calibration_slope"] * calibrated_raw
        + best_ensemble_params["calibration_intercept"],
        0.0,
        1.0,
    )

    metrics = {
        "validation_season": args.validation_season,
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_validation)),
        "validation_target_rate": float(y_validation.mean()),
        "histgb": {
            "params": hist_study.best_params,
            "brier": brier_score_loss(y_validation, hist_prediction),
            "competition_score": competition_score(y_validation, hist_prediction),
            "prediction_mean": float(hist_prediction.mean()),
            "fit_seconds": hist_seconds,
        },
        "extra_trees": {
            "params": {**extra_study.best_params, "n_estimators": args.extra_final_estimators},
            "brier": brier_score_loss(y_validation, extra_prediction),
            "competition_score": competition_score(y_validation, extra_prediction),
            "prediction_mean": float(extra_prediction.mean()),
            "fit_seconds": extra_seconds,
        },
        "raw_ensemble": {
            "hist_weight": raw_weight,
            "extra_weight": 1.0 - raw_weight,
            "brier": float(raw_briers[raw_index]),
            "competition_score": competition_score(y_validation, raw_prediction),
            "prediction_mean": float(raw_prediction.mean()),
        },
        "calibrated_ensemble": {
            "params": best_ensemble_params,
            "brier": brier_score_loss(y_validation, calibrated_prediction),
            "competition_score": competition_score(y_validation, calibrated_prediction),
            "prediction_mean": float(calibrated_prediction.mean()),
        },
        "search": {
            "hist_trials": completed_trials(hist_study),
            "extra_trials": completed_trials(extra_study),
            "ensemble_trials": completed_trials(ensemble_study),
            "extra_search_estimators": args.extra_search_estimators,
            "preprocess_seconds": preprocess_seconds,
            "seed": args.seed,
            "storage": str(database),
        },
        "caveat": "Parameters are optimized on the 2024 holdout and require 2022/2023 forward validation before promotion.",
    }
    (args.artifact_dir / "best_params.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame({
        ID_COL: validation_ids,
        TARGET_COL: y_validation,
        "histgb_prediction": hist_prediction,
        "extra_prediction": extra_prediction,
        "raw_ensemble_prediction": raw_prediction,
        "calibrated_ensemble_prediction": calibrated_prediction,
    }).to_csv(args.artifact_dir / "oof_predictions.csv", index=False)
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "hist_model": best_hist_model,
            "extra_model": best_extra_model,
            "feature_columns": feature_columns,
            "raw_hist_weight": raw_weight,
            "calibrated_params": best_ensemble_params,
        },
        args.artifact_dir / "validation_models.pkl",
        compress=3,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
