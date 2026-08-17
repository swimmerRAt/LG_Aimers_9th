#!/usr/bin/env python3
"""Leakage-resistant Optuna tuning with a one-shot temporal outer holdout.

Workflow
--------
1. ``tune`` uses only the configured inner forward seasons (2021-2023 by
   default). It minimizes a recency-weighted normalized Brier loss plus a
   cross-season stability penalty.
2. ``evaluate-outer`` freezes the selected parameters and evaluates 2024 once.
   Creating the outer lock permanently prevents adding more trials to the same
   artifact directory after the holdout has been inspected.

Optuna is a local tuning dependency. It is not required by the submission ZIP.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss

from optimize_hyperparameters import (
    completed_trials,
    extra_params_from_trial,
    hist_params_from_trial,
    make_preprocessor,
    remaining_trials,
    trial_frame,
)
from src.lg_aimers.metrics import competition_score
from src.lg_aimers.validation import make_season_forward_splits
from train_optimized import EXCLUDED_FEATURES


ID_COL = "row_id"
TARGET_COL = "control_success"

CURRENT_HIST_PARAMS = {
    "learning_rate": 0.04,
    "max_iter": 300,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 200,
    "l2_regularization": 5.0,
    "max_bins": 255,
}
CURRENT_EXTRA_PARAMS = {
    "max_depth": 16,
    "min_samples_leaf": 100,
    "max_features": 0.8,
    "criterion": "gini",
}
CURRENT_HIST_WEIGHT = 0.45


@dataclass
class PreparedFold:
    season: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    target_rate: float
    reference_brier: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: object, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def normalized_brier(y_true: np.ndarray, prediction: np.ndarray) -> float:
    target_rate = float(np.mean(y_true))
    reference = target_rate * (1.0 - target_rate)
    if reference <= 0.0:
        raise ValueError("normalized Brier is undefined for a one-class fold")
    return float(brier_score_loss(y_true, prediction) / reference)


def robust_objective(
    normalized_losses: list[float] | np.ndarray,
    weights: list[float] | np.ndarray,
    stability_penalty: float,
) -> tuple[float, float, float]:
    losses = np.asarray(normalized_losses, dtype=float)
    fold_weights = np.asarray(weights, dtype=float)
    if losses.ndim != 1 or not len(losses):
        raise ValueError("normalized_losses must be a non-empty 1-D sequence")
    if losses.shape != fold_weights.shape:
        raise ValueError("loss and weight shapes must match")
    if not np.isfinite(losses).all() or not np.isfinite(fold_weights).all():
        raise ValueError("losses and weights must be finite")
    if (fold_weights <= 0).any():
        raise ValueError("all fold weights must be positive")
    if stability_penalty < 0:
        raise ValueError("stability_penalty must be non-negative")
    fold_weights = fold_weights / fold_weights.sum()
    weighted_mean = float(np.sum(fold_weights * losses))
    weighted_std = float(
        np.sqrt(np.sum(fold_weights * np.square(losses - weighted_mean)))
    )
    objective = weighted_mean + float(stability_penalty) * weighted_std
    return objective, weighted_mean, weighted_std


def normalized_weights(values: list[float], seasons: list[int]) -> list[float]:
    if len(values) != len(seasons):
        raise ValueError(
            f"--inner-weights needs {len(seasons)} values for seasons {seasons}"
        )
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all() or (array <= 0).any():
        raise ValueError("inner weights must be finite and positive")
    return (array / array.sum()).tolist()


def load_training_frame(data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    test_columns = pd.read_csv(data_dir / "test.csv", nrows=0).columns.tolist()
    feature_columns = [
        column
        for column in test_columns
        if column != ID_COL and column not in EXCLUDED_FEATURES
    ]
    train = pd.read_csv(
        data_dir / "train.csv",
        usecols=[TARGET_COL, *feature_columns],
    )
    return train, feature_columns


def prepare_folds(
    train: pd.DataFrame,
    feature_columns: list[str],
    seasons: list[int],
) -> list[PreparedFold]:
    prepared: list[PreparedFold] = []
    for split in make_season_forward_splits(train, seasons):
        y_train = train.iloc[split.train_index][TARGET_COL].to_numpy()
        y_validation = train.iloc[split.validation_index][TARGET_COL].to_numpy()
        preprocessor = make_preprocessor(feature_columns)
        started = time.perf_counter()
        X_train = preprocessor.fit_transform(
            train.iloc[split.train_index][feature_columns], y_train
        )
        X_validation = preprocessor.transform(
            train.iloc[split.validation_index][feature_columns]
        )
        rate = float(y_validation.mean())
        reference = rate * (1.0 - rate)
        print(
            json.dumps(
                {
                    "prepared_season": split.validation_season,
                    "train_rows": len(split.train_index),
                    "validation_rows": len(split.validation_index),
                    "target_rate": rate,
                    "seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
        prepared.append(
            PreparedFold(
                season=split.validation_season,
                X_train=X_train,
                y_train=y_train,
                X_validation=X_validation,
                y_validation=y_validation,
                target_rate=rate,
                reference_brier=reference,
            )
        )
    return prepared


def make_study(name: str, storage: str, seed: int) -> optuna.Study:
    return optuna.create_study(
        study_name=name,
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
    )


def fit_hist_prediction(
    params: dict, fold: PreparedFold, seed: int
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    model = HistGradientBoostingClassifier(
        **params,
        early_stopping=False,
        random_state=seed,
    ).fit(fold.X_train, fold.y_train)
    prediction = model.predict_proba(fold.X_validation)[:, 1]
    seconds = time.perf_counter() - started
    del model
    gc.collect()
    return prediction, seconds


def fit_extra_prediction(
    params: dict,
    fold: PreparedFold,
    n_estimators: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    model = ExtraTreesClassifier(
        **params,
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=seed,
    ).fit(fold.X_train, fold.y_train)
    prediction = model.predict_proba(fold.X_validation)[:, 1]
    seconds = time.perf_counter() - started
    del model
    gc.collect()
    return prediction, seconds


def evaluate_component(
    folds: list[PreparedFold],
    weights: list[float],
    stability_penalty: float,
    predictor: Callable[[PreparedFold], tuple[np.ndarray, float]],
) -> tuple[float, list[float], list[float], list[float]]:
    normalized_losses: list[float] = []
    briers: list[float] = []
    fit_seconds: list[float] = []
    for fold in folds:
        prediction, seconds = predictor(fold)
        brier = float(brier_score_loss(fold.y_validation, prediction))
        briers.append(brier)
        normalized_losses.append(brier / fold.reference_brier)
        fit_seconds.append(seconds)
        del prediction
    objective, _, _ = robust_objective(
        normalized_losses, weights, stability_penalty
    )
    return objective, normalized_losses, briers, fit_seconds


def set_fold_attrs(
    trial: optuna.Trial,
    seasons: list[int],
    normalized_losses: list[float],
    briers: list[float],
    fit_seconds: list[float],
    weights: list[float],
    stability_penalty: float,
) -> None:
    objective, weighted_mean, weighted_std = robust_objective(
        normalized_losses, weights, stability_penalty
    )
    trial.set_user_attr("robust_objective", objective)
    trial.set_user_attr("weighted_mean_normalized_brier", weighted_mean)
    trial.set_user_attr("weighted_std_normalized_brier", weighted_std)
    for season, loss, brier, seconds in zip(
        seasons, normalized_losses, briers, fit_seconds
    ):
        trial.set_user_attr(f"season_{season}_normalized_brier", loss)
        trial.set_user_attr(f"season_{season}_brier", brier)
        trial.set_user_attr(f"season_{season}_fit_seconds", seconds)


def prediction_map(
    folds: list[PreparedFold],
    predictor: Callable[[PreparedFold], tuple[np.ndarray, float]],
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for fold in folds:
        result[fold.season], _ = predictor(fold)
    return result


def ensemble_fold_metrics(
    folds: list[PreparedFold],
    hist_predictions: dict[int, np.ndarray],
    extra_predictions: dict[int, np.ndarray],
    hist_weight: float,
) -> list[dict]:
    rows: list[dict] = []
    for fold in folds:
        prediction = (
            hist_weight * hist_predictions[fold.season]
            + (1.0 - hist_weight) * extra_predictions[fold.season]
        )
        brier = float(brier_score_loss(fold.y_validation, prediction))
        rows.append(
            {
                "validation_season": fold.season,
                "validation_rows": len(fold.y_validation),
                "target_rate": fold.target_rate,
                "prediction_mean": float(prediction.mean()),
                "brier": brier,
                "normalized_brier": brier / fold.reference_brier,
                "competition_score": competition_score(
                    fold.y_validation, prediction
                ),
            }
        )
    return rows


def objective_from_metric_rows(
    rows: list[dict], weights: list[float], stability_penalty: float
) -> dict:
    objective, weighted_mean, weighted_std = robust_objective(
        [row["normalized_brier"] for row in rows],
        weights,
        stability_penalty,
    )
    return {
        "objective": objective,
        "weighted_mean_normalized_brier": weighted_mean,
        "weighted_std_normalized_brier": weighted_std,
    }


def selection_signature(selection: dict) -> str:
    signature_payload = {
        "config": selection["config"],
        "feature_columns": selection["feature_columns"],
        "histgb_params": selection["histgb_params"],
        "extra_trees_params": selection["extra_trees_params"],
        "hist_weight": selection["hist_weight"],
    }
    return canonical_hash(signature_payload, length=24)


def assert_tuning_unlocked(artifact_dir: Path) -> None:
    lock = artifact_dir / "outer_lock.json"
    result = artifact_dir / "outer_evaluation.json"
    if lock.exists() or result.exists():
        raise RuntimeError(
            "outer holdout is already locked/evaluated; no more trials may be "
            "added in this artifact directory. Start a new pre-registered "
            "experiment directory instead."
        )


def tune(args: argparse.Namespace) -> None:
    assert_tuning_unlocked(args.artifact_dir)
    seasons = [int(season) for season in args.inner_seasons]
    if sorted(seasons) != seasons or len(set(seasons)) != len(seasons):
        raise ValueError("inner seasons must be unique and increasing")
    if args.outer_season in seasons:
        raise ValueError("outer season must not appear in inner seasons")
    if max(seasons) >= args.outer_season:
        raise ValueError("all inner seasons must precede the outer season")
    weights = normalized_weights(args.inner_weights, seasons)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "inner_seasons": seasons,
        "inner_weights": weights,
        "outer_season": int(args.outer_season),
        "stability_penalty": float(args.stability_penalty),
        "n_estimators": int(args.n_estimators),
        "seed": int(args.seed),
        "feature_policy": "test columns minus EXCLUDED_FEATURES",
        "calibration": "disabled",
    }
    train, feature_columns = load_training_frame(args.data_dir)
    config["feature_columns"] = feature_columns
    config["excluded_features"] = sorted(EXCLUDED_FEATURES)
    config_signature = canonical_hash(config)
    # Keep the outer season outside every object reachable by an Optuna
    # objective. The CSV is shared, but its 2024 rows and labels are discarded
    # before fold preparation or study creation.
    train = train.loc[train["season"] < args.outer_season].reset_index(drop=True)
    folds = prepare_folds(train, feature_columns, seasons)

    database = (args.artifact_dir / "optuna.db").resolve()
    storage = f"sqlite:///{database}"

    hist_study = make_study(
        f"robust_histgb_{config_signature}", storage, args.seed
    )
    if not hist_study.trials:
        hist_study.enqueue_trial(CURRENT_HIST_PARAMS)

    def hist_objective(trial: optuna.Trial) -> float:
        params = hist_params_from_trial(trial)
        objective, losses, briers, seconds = evaluate_component(
            folds,
            weights,
            args.stability_penalty,
            lambda fold: fit_hist_prediction(params, fold, args.seed),
        )
        set_fold_attrs(
            trial,
            seasons,
            losses,
            briers,
            seconds,
            weights,
            args.stability_penalty,
        )
        return objective

    todo = remaining_trials(hist_study, args.hist_trials)
    if todo:
        hist_study.optimize(hist_objective, n_trials=todo, gc_after_trial=True)
    trial_frame(hist_study).to_csv(
        args.artifact_dir / "histgb_trials.csv", index=False
    )

    extra_study = make_study(
        f"robust_extra_trees_{config_signature}", storage, args.seed + 1
    )
    if not extra_study.trials:
        extra_study.enqueue_trial(CURRENT_EXTRA_PARAMS)

    def extra_objective(trial: optuna.Trial) -> float:
        params = extra_params_from_trial(trial)
        objective, losses, briers, seconds = evaluate_component(
            folds,
            weights,
            args.stability_penalty,
            lambda fold: fit_extra_prediction(
                params, fold, args.n_estimators, args.seed
            ),
        )
        set_fold_attrs(
            trial,
            seasons,
            losses,
            briers,
            seconds,
            weights,
            args.stability_penalty,
        )
        return objective

    todo = remaining_trials(extra_study, args.extra_trials)
    if todo:
        extra_study.optimize(extra_objective, n_trials=todo, gc_after_trial=True)
    trial_frame(extra_study).to_csv(
        args.artifact_dir / "extra_trees_trials.csv", index=False
    )

    best_hist_predictions = prediction_map(
        folds,
        lambda fold: fit_hist_prediction(hist_study.best_params, fold, args.seed),
    )
    best_extra_predictions = prediction_map(
        folds,
        lambda fold: fit_extra_prediction(
            extra_study.best_params, fold, args.n_estimators, args.seed
        ),
    )
    base_model_signature = canonical_hash(
        {
            "config": config,
            "histgb": hist_study.best_params,
            "extra_trees": extra_study.best_params,
        }
    )
    blend_study = make_study(
        f"robust_blend_{base_model_signature}", storage, args.seed + 2
    )
    if not blend_study.trials:
        blend_study.enqueue_trial({"hist_weight": CURRENT_HIST_WEIGHT})

    def blend_objective(trial: optuna.Trial) -> float:
        hist_weight = trial.suggest_float("hist_weight", 0.0, 1.0)
        rows = ensemble_fold_metrics(
            folds,
            best_hist_predictions,
            best_extra_predictions,
            hist_weight,
        )
        summary = objective_from_metric_rows(
            rows, weights, args.stability_penalty
        )
        trial.set_user_attr(
            "weighted_mean_normalized_brier",
            summary["weighted_mean_normalized_brier"],
        )
        trial.set_user_attr(
            "weighted_std_normalized_brier",
            summary["weighted_std_normalized_brier"],
        )
        for row in rows:
            trial.set_user_attr(
                f"season_{row['validation_season']}_brier", row["brier"]
            )
        return summary["objective"]

    todo = remaining_trials(blend_study, args.blend_trials)
    if todo:
        blend_study.optimize(blend_objective, n_trials=todo)
    trial_frame(blend_study).to_csv(
        args.artifact_dir / "blend_trials.csv", index=False
    )

    candidate_rows = ensemble_fold_metrics(
        folds,
        best_hist_predictions,
        best_extra_predictions,
        float(blend_study.best_params["hist_weight"]),
    )
    candidate_summary = objective_from_metric_rows(
        candidate_rows, weights, args.stability_penalty
    )

    if hist_study.best_params == CURRENT_HIST_PARAMS:
        current_hist_predictions = best_hist_predictions
    else:
        current_hist_predictions = prediction_map(
            folds,
            lambda fold: fit_hist_prediction(CURRENT_HIST_PARAMS, fold, args.seed),
        )
    if extra_study.best_params == CURRENT_EXTRA_PARAMS:
        current_extra_predictions = best_extra_predictions
    else:
        current_extra_predictions = prediction_map(
            folds,
            lambda fold: fit_extra_prediction(
                CURRENT_EXTRA_PARAMS, fold, args.n_estimators, args.seed
            ),
        )
    baseline_rows = ensemble_fold_metrics(
        folds,
        current_hist_predictions,
        current_extra_predictions,
        CURRENT_HIST_WEIGHT,
    )
    baseline_summary = objective_from_metric_rows(
        baseline_rows, weights, args.stability_penalty
    )
    metrics = pd.DataFrame(
        [
            {"candidate": "current_baseline", **row}
            for row in baseline_rows
        ]
        + [{"candidate": "robust_optuna", **row} for row in candidate_rows]
    )
    metrics.to_csv(args.artifact_dir / "inner_fold_metrics.csv", index=False)

    selection = {
        "status": "frozen_pending_outer",
        "created_at": utc_now(),
        "config": config,
        "feature_columns": feature_columns,
        "histgb_params": hist_study.best_params,
        "extra_trees_params": {
            **extra_study.best_params,
            "n_estimators": args.n_estimators,
        },
        "hist_weight": float(blend_study.best_params["hist_weight"]),
        "extra_weight": 1.0 - float(blend_study.best_params["hist_weight"]),
        "inner_candidate": candidate_summary,
        "inner_baseline": baseline_summary,
        "studies": {
            "histgb": {
                "name": hist_study.study_name,
                "completed_trials": completed_trials(hist_study),
                "best_value": hist_study.best_value,
            },
            "extra_trees": {
                "name": extra_study.study_name,
                "completed_trials": completed_trials(extra_study),
                "best_value": extra_study.best_value,
            },
            "blend": {
                "name": blend_study.study_name,
                "completed_trials": completed_trials(blend_study),
                "best_value": blend_study.best_value,
            },
        },
    }
    selection["selection_signature"] = selection_signature(selection)
    (args.artifact_dir / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)


def fit_outer_models(
    train: pd.DataFrame,
    feature_columns: list[str],
    outer_season: int,
    hist_params: dict,
    extra_params: dict,
    n_estimators: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    train_mask = train["season"] < outer_season
    validation_mask = train["season"] == outer_season
    if not train_mask.any() or not validation_mask.any():
        raise ValueError(f"empty outer split for season {outer_season}")
    y_train = train.loc[train_mask, TARGET_COL].to_numpy()
    y_validation = train.loc[validation_mask, TARGET_COL].to_numpy()
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
        random_state=seed,
    ).fit(X_train, y_train)
    extra_model = ExtraTreesClassifier(
        **extra_params,
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=seed,
    ).fit(X_train, y_train)
    hist_prediction = hist_model.predict_proba(X_validation)[:, 1]
    extra_prediction = extra_model.predict_proba(X_validation)[:, 1]
    seconds = time.perf_counter() - started
    return y_validation, hist_prediction, extra_prediction, seconds


def metric_record(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "rows": int(len(y_true)),
        "target_rate": float(y_true.mean()),
        "prediction_mean": float(prediction.mean()),
        "brier": float(brier_score_loss(y_true, prediction)),
        "normalized_brier": normalized_brier(y_true, prediction),
        "competition_score": competition_score(y_true, prediction),
    }


def evaluate_outer(args: argparse.Namespace) -> None:
    selection_path = args.artifact_dir / "selection.json"
    result_path = args.artifact_dir / "outer_evaluation.json"
    lock_path = args.artifact_dir / "outer_lock.json"
    if result_path.exists():
        print(result_path.read_text(encoding="utf-8"), flush=True)
        print("outer result already exists; reused without reevaluation", flush=True)
        return
    if not selection_path.exists():
        raise FileNotFoundError("run the tune command before evaluating the outer holdout")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    expected_signature = selection_signature(selection)
    if selection.get("selection_signature") != expected_signature:
        raise ValueError("selection.json signature mismatch")

    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("selection_signature") != expected_signature:
            raise RuntimeError("outer holdout is locked to a different selection")
    else:
        lock_path.write_text(
            json.dumps(
                {
                    "locked_at": utc_now(),
                    "selection_signature": expected_signature,
                    "reason": "one-shot outer holdout evaluation started",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    config = selection["config"]
    outer_season = int(config["outer_season"])
    train, feature_columns = load_training_frame(args.data_dir)
    if feature_columns != selection["feature_columns"]:
        raise ValueError("feature columns changed after tuning")
    tuned_extra_params = dict(selection["extra_trees_params"])
    n_estimators = int(tuned_extra_params.pop("n_estimators"))

    y_true, tuned_hist, tuned_extra, tuned_seconds = fit_outer_models(
        train,
        feature_columns,
        outer_season,
        selection["histgb_params"],
        tuned_extra_params,
        n_estimators,
        int(config["seed"]),
    )
    candidate_prediction = (
        float(selection["hist_weight"]) * tuned_hist
        + float(selection["extra_weight"]) * tuned_extra
    )

    if (
        selection["histgb_params"] == CURRENT_HIST_PARAMS
        and tuned_extra_params == CURRENT_EXTRA_PARAMS
    ):
        baseline_prediction = (
            CURRENT_HIST_WEIGHT * tuned_hist
            + (1.0 - CURRENT_HIST_WEIGHT) * tuned_extra
        )
        baseline_seconds = 0.0
    else:
        _, baseline_hist, baseline_extra, baseline_seconds = fit_outer_models(
            train,
            feature_columns,
            outer_season,
            CURRENT_HIST_PARAMS,
            CURRENT_EXTRA_PARAMS,
            n_estimators,
            int(config["seed"]),
        )
        baseline_prediction = (
            CURRENT_HIST_WEIGHT * baseline_hist
            + (1.0 - CURRENT_HIST_WEIGHT) * baseline_extra
        )

    baseline_metric = metric_record(y_true, baseline_prediction)
    candidate_metric = metric_record(y_true, candidate_prediction)
    paired_delta = np.square(candidate_prediction - y_true) - np.square(
        baseline_prediction - y_true
    )
    delta_brier = float(paired_delta.mean())
    delta_se = float(paired_delta.std(ddof=1) / math.sqrt(len(paired_delta)))
    ci_low = delta_brier - 1.96 * delta_se
    ci_high = delta_brier + 1.96 * delta_se
    inner_improved = (
        selection["inner_candidate"]["objective"]
        < selection["inner_baseline"]["objective"]
    )
    approved = bool(
        inner_improved
        and delta_brier <= -float(args.min_brier_improvement)
        and ci_high < 0.0
    )
    result = {
        "status": "approved_for_final_training" if approved else "rejected_keep_baseline",
        "evaluated_at": utc_now(),
        "selection_signature": expected_signature,
        "outer_season": outer_season,
        "baseline": baseline_metric,
        "robust_optuna": candidate_metric,
        "paired_comparison": {
            "candidate_minus_baseline_brier": delta_brier,
            "standard_error": delta_se,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "minimum_required_improvement": float(args.min_brier_improvement),
        },
        "gates": {
            "inner_robust_objective_improved": inner_improved,
            "minimum_effect_met": delta_brier
            <= -float(args.min_brier_improvement),
            "paired_ci_excludes_zero": ci_high < 0.0,
        },
        "fit_seconds": {
            "candidate": tuned_seconds,
            "baseline_additional": baseline_seconds,
        },
        "policy": (
            "The outer result is immutable in this artifact directory. "
            "Failed candidates require a new pre-registered experiment and "
            "must not be retuned against this 2024 result."
        ),
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    tune_parser = subparsers.add_parser("tune")
    tune_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    tune_parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/optuna_robust")
    )
    tune_parser.add_argument(
        "--inner-seasons", type=int, nargs="+", default=[2021, 2022, 2023]
    )
    tune_parser.add_argument(
        "--inner-weights", type=float, nargs="+", default=[0.15, 0.30, 0.55]
    )
    tune_parser.add_argument("--outer-season", type=int, default=2024)
    tune_parser.add_argument("--stability-penalty", type=float, default=0.25)
    tune_parser.add_argument("--hist-trials", type=int, default=20)
    tune_parser.add_argument("--extra-trials", type=int, default=20)
    tune_parser.add_argument("--blend-trials", type=int, default=100)
    tune_parser.add_argument("--n-estimators", type=int, default=160)
    tune_parser.add_argument("--seed", type=int, default=42)
    tune_parser.set_defaults(func=tune)

    outer_parser = subparsers.add_parser("evaluate-outer")
    outer_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    outer_parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/optuna_robust")
    )
    outer_parser.add_argument(
        "--min-brier-improvement", type=float, default=0.00002
    )
    outer_parser.set_defaults(func=evaluate_outer)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
