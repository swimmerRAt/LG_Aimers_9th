#!/usr/bin/env python3
"""Compare XGBoost with the current ensemble on the 2024 forward holdout."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost

from model.xgboost_model import XGBoostProbabilityModel
from script import render_feature_importance_svg
from src.lg_aimers import ID_COL, TARGET_COL
from src.lg_aimers.metrics import brier_score, competition_score, paired_brier_comparison


def load_baseline(summary_path: Path, metrics_path: Path) -> tuple[list[str], dict]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metric_frame = pd.read_csv(metrics_path)
    if len(metric_frame) != 1:
        raise ValueError(f"expected one baseline metric row in {metrics_path}")
    source = metric_frame.iloc[0]
    metric = {
        "model": "current_histgb_extra_trees",
        "validation_rows": int(source["validation_rows"]),
        "target_rate": float(source["validation_target_rate"]),
        "prediction_mean": float(source["prediction_mean"]),
        "brier": float(source["brier"]),
        "competition_score": float(source["competition_score"]),
        "best_iterations": np.nan,
        "fit_seconds": float(source["fit_seconds"]),
        "inference_seconds": float(source["inference_seconds"]),
    }
    return list(summary["feature_columns"]), metric


def optimal_xgboost_blend_weight(truth, baseline, candidate) -> float:
    """Return validation-optimal candidate weight in a two-model probability blend."""
    truth = np.asarray(truth, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    direction = candidate - baseline
    denominator = float(np.dot(direction, direction))
    if denominator == 0.0:
        return 0.0
    weight = -float(np.dot(baseline - truth, direction)) / denominator
    return float(np.clip(weight, 0.0, 1.0))


def make_model(args: argparse.Namespace) -> XGBoostProbabilityModel:
    return XGBoostProbabilityModel(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        max_bin=args.max_bin,
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-child-weight", type=float, default=100.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.1)
    parser.add_argument("--reg-lambda", type=float, default=15.0)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("artifacts/optimized_ensemble_2024/run_summary.json"),
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=Path("artifacts/optimized_ensemble_2024/metrics.csv"),
    )
    parser.add_argument(
        "--baseline-oof",
        type=Path,
        default=Path("artifacts/optimized_ensemble_2024/oof_predictions.csv"),
    )
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/xgboost_2024"))
    parser.add_argument("--model-output", type=Path, default=Path("model/xgboost_candidate.pkl"))
    parser.add_argument("--skip-final-fit", action="store_true")
    args = parser.parse_args()

    feature_columns, baseline_metric = load_baseline(
        args.baseline_summary, args.baseline_metrics
    )
    train = pd.read_csv(
        args.data_dir / "train.csv",
        usecols=[ID_COL, TARGET_COL, *feature_columns],
    )
    train_mask = train["season"] < args.validation_season
    validation_mask = train["season"] == args.validation_season
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("forward validation split is empty")

    X_train = train.loc[train_mask, feature_columns]
    y_train = train.loc[train_mask, TARGET_COL].to_numpy()
    X_validation = train.loc[validation_mask, feature_columns]
    truth = train.loc[validation_mask, TARGET_COL].to_numpy()

    model = make_model(args)
    started = time.perf_counter()
    model.fit(X_train, y_train, eval_set=(X_validation, truth))
    fit_seconds = time.perf_counter() - started
    infer_started = time.perf_counter()
    prediction = model.predict_proba(X_validation)[:, 1]
    inference_seconds = time.perf_counter() - infer_started

    xgb_metric = {
        "model": "xgboost_classifier",
        "validation_rows": len(truth),
        "target_rate": float(truth.mean()),
        "prediction_mean": float(prediction.mean()),
        "brier": brier_score(truth, prediction),
        "competition_score": competition_score(truth, prediction),
        "best_iterations": model.best_iteration_count(),
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
    }

    baseline_oof = pd.read_csv(args.baseline_oof)
    expected_ids = train.loc[validation_mask, ID_COL].to_numpy()
    baseline_by_id = baseline_oof.set_index(ID_COL)
    if not pd.Index(expected_ids).isin(baseline_by_id.index).all():
        raise ValueError("baseline OOF does not contain every validation row_id")
    baseline_prediction = baseline_by_id.loc[expected_ids, "prediction"].to_numpy(float)
    comparison = paired_brier_comparison(truth, baseline_prediction, prediction)
    xgb_metric.update(comparison)

    xgb_weight = optimal_xgboost_blend_weight(truth, baseline_prediction, prediction)
    blend_prediction = np.clip(
        (1.0 - xgb_weight) * baseline_prediction + xgb_weight * prediction,
        0.0,
        1.0,
    )
    blend_metric = {
        "model": "diagnostic_optimal_baseline_xgboost_blend",
        "validation_rows": len(truth),
        "target_rate": float(truth.mean()),
        "prediction_mean": float(blend_prediction.mean()),
        "brier": brier_score(truth, blend_prediction),
        "competition_score": competition_score(truth, blend_prediction),
        "best_iterations": np.nan,
        "fit_seconds": np.nan,
        "inference_seconds": np.nan,
        "xgboost_weight": xgb_weight,
    }

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    metric_frame = pd.DataFrame([baseline_metric, xgb_metric, blend_metric])
    metric_frame.sort_values("brier", kind="stable").to_csv(
        args.artifact_dir / "metrics.csv", index=False
    )
    pd.DataFrame(
        {
            ID_COL: expected_ids,
            TARGET_COL: truth,
            "baseline_prediction": baseline_prediction,
            "xgboost_prediction": prediction,
            "diagnostic_blend_prediction": blend_prediction,
        }
    ).to_csv(args.artifact_dir / "oof_predictions.csv", index=False)

    importance_names, importance_values = model.feature_importance_frame()
    importance_frame = pd.DataFrame(
        {
            "rank": np.arange(1, len(importance_names) + 1),
            "feature": importance_names,
            "importance": importance_values,
            "importance_percent": 100.0 * importance_values,
            "source_component": model.feature_importance_source,
        }
    )
    importance_frame.to_csv(args.artifact_dir / "feature_importance.csv", index=False)
    (args.artifact_dir / "feature_importance.svg").write_text(
        render_feature_importance_svg(importance_frame), encoding="utf-8"
    )

    baseline_brier = float(baseline_metric["brier"])
    xgb_brier = float(xgb_metric["brier"])
    beats_baseline = xgb_brier < baseline_brier
    summary = {
        "status": "xgboost_beats_baseline" if beats_baseline else "rejected_keep_baseline",
        "validation_season": args.validation_season,
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "feature_columns": feature_columns,
        "feature_count_raw": len(feature_columns),
        "objective": "binary:logistic",
        "early_stopping_metric": "rmse (sqrt of Brier for a binary target)",
        "class_weights": None,
        "best_iterations": model.best_iteration_count(),
        "baseline_brier": baseline_brier,
        "xgboost_brier": xgb_brier,
        "brier_delta_xgboost_minus_baseline": xgb_brier - baseline_brier,
        "baseline_score": float(baseline_metric["competition_score"]),
        "xgboost_score": float(xgb_metric["competition_score"]),
        "paired_brier_delta": float(comparison["paired_brier_delta"]),
        "paired_ci95_low": float(comparison["paired_ci95_low"]),
        "paired_ci95_high": float(comparison["paired_ci95_high"]),
        "diagnostic_optimal_xgboost_weight": xgb_weight,
        "diagnostic_blend_brier": float(blend_metric["brier"]),
        "diagnostic_blend_score": float(blend_metric["competition_score"]),
        "xgboost_version": xgboost.__version__,
        "python_version": platform.python_version(),
        "params": {
            "n_estimators_cap": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "min_child_weight": args.min_child_weight,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
            "max_bin": args.max_bin,
            "early_stopping_rounds": args.early_stopping_rounds,
            "random_state": args.random_state,
        },
    }

    if beats_baseline and not args.skip_final_fit:
        final_model = make_model(args)
        final_started = time.perf_counter()
        final_model.refit_full(
            train[feature_columns],
            train[TARGET_COL].to_numpy(),
            n_estimators=model.best_iteration_count(),
        )
        summary["final_fit_seconds"] = time.perf_counter() - final_started
        artifact = {
            "model": final_model,
            "feature_columns": feature_columns,
            "positive_class": 1,
            "selected_model": "xgboost_classifier",
            "validation_seasons": [args.validation_season],
            "full_train_rows": len(train),
            "random_state": args.random_state,
            "xgboost_version": xgboost.__version__,
        }
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, args.model_output, compress=3)
        summary["model_output"] = str(args.model_output)
    else:
        summary["model_output"] = None

    (args.artifact_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({**xgb_metric, **comparison}, ensure_ascii=False), flush=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
