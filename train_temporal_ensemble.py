#!/usr/bin/env python3
"""Build and validate a multi-window HistGB + ExtraTrees temporal ensemble."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import minimize

from model.ensemble import OptimizedBaseballEnsemble
from model.temporal_ensemble import COMPONENT_NAMES, TemporalWindowEnsemble
from src.lg_aimers import ID_COL, TARGET_COL
from src.lg_aimers.metrics import brier_score, competition_score, paired_brier_comparison


ACTIVE_COMPONENT_NAMES = ("full", "recent_3", "recent_2")


def normalized_fold_brier(truth: np.ndarray, prediction: np.ndarray) -> float:
    event_rate = float(np.mean(truth))
    reference = event_rate * (1.0 - event_rate)
    if reference <= 0.0:
        raise ValueError("fold must contain both target classes")
    return brier_score(truth, prediction) / reference


def select_continuous_blend_weights(
    oof: pd.DataFrame,
    seasons: tuple[int, ...],
    season_weights: tuple[float, ...],
    stability_penalty: float,
) -> tuple[tuple[float, ...], pd.DataFrame]:
    if len(seasons) != len(season_weights):
        raise ValueError("seasons and season_weights must have equal length")
    fold_weights = np.asarray(season_weights, dtype=float)
    if (fold_weights < 0).any() or fold_weights.sum() <= 0:
        raise ValueError("season_weights must be non-negative with a positive sum")
    fold_weights /= fold_weights.sum()
    component_matrix = oof.loc[:, ACTIVE_COMPONENT_NAMES].to_numpy(float)
    truth = oof[TARGET_COL].to_numpy(float)
    season_values = oof["validation_season"].to_numpy(int)

    def evaluate(weights) -> tuple[float, float, float, np.ndarray]:
        # SLSQP's finite-difference probes can exceed a bound by machine epsilon.
        prediction = np.clip(component_matrix @ np.asarray(weights), 0.0, 1.0)
        fold_losses = []
        for season in seasons:
            mask = season_values == season
            if not mask.any():
                raise ValueError(f"OOF has no validation rows for season {season}")
            fold_losses.append(normalized_fold_brier(truth[mask], prediction[mask]))
        fold_losses = np.asarray(fold_losses)
        mean_loss = float(np.dot(fold_weights, fold_losses))
        variance = float(np.dot(fold_weights, np.square(fold_losses - mean_loss)))
        objective = mean_loss + float(stability_penalty) * np.sqrt(variance)
        return objective, mean_loss, np.sqrt(variance), fold_losses

    result = minimize(
        lambda weights: evaluate(weights)[0],
        np.full(len(ACTIVE_COMPONENT_NAMES), 1.0 / len(ACTIVE_COMPONENT_NAMES)),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(ACTIVE_COMPONENT_NAMES),
        constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
        options={"ftol": 1e-15, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"continuous blend optimization failed: {result.message}")
    active_weights = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
    active_weights /= active_weights.sum()
    objective, mean_loss, loss_std, fold_losses = evaluate(active_weights)
    all_weights = tuple(active_weights.tolist()) + (0.0,)
    row = {
        "optimizer": "SLSQP",
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective": objective,
        "weighted_mean_relative_brier": mean_loss,
        "relative_brier_std": loss_std,
    }
    row.update({name: weight for name, weight in zip(COMPONENT_NAMES, all_weights)})
    row.update({f"relative_brier_{season}": value for season, value in zip(seasons, fold_losses)})
    return all_weights, pd.DataFrame([row])


def fixed_blend_weights(values, label: str) -> tuple[tuple[float, ...], pd.DataFrame]:
    weights = np.asarray(values, dtype=float)
    if weights.shape != (len(ACTIVE_COMPONENT_NAMES),):
        raise ValueError(
            f"{label} must provide {len(ACTIVE_COMPONENT_NAMES)} active component weights"
        )
    if not np.isfinite(weights).all() or (weights < 0.0).any() or weights.sum() <= 0.0:
        raise ValueError(f"{label} weights must be finite, non-negative, and have a positive sum")
    weights /= weights.sum()
    all_weights = tuple(weights.tolist()) + (0.0,)
    row = {"strategy": "fixed_from_prior_temporal_ensemble"}
    row.update({name: weight for name, weight in zip(COMPONENT_NAMES, all_weights)})
    return all_weights, pd.DataFrame([row])


def cache_file(
    cache_dir: Path,
    validation_season: int,
    component: str,
    time_decay: float,
    hist_max_iter: int,
    n_estimators: int,
    smoothing_lambdas: tuple[float, ...] = (),
) -> Path:
    smoothing_signature = ",".join(f"{value:g}" for value in smoothing_lambdas)
    version = "v3" if smoothing_lambdas else "v2"
    signature = hashlib.sha256(
        f"{version}|{validation_season}|{component}|{time_decay:.6f}|"
        f"{hist_max_iter}|{n_estimators}|{smoothing_signature}".encode()
    ).hexdigest()[:10]
    return cache_dir / f"{validation_season}_{component}_{signature}.npz"


def load_existing_2024_full_prediction(
    baseline_oof: Path,
    validation_ids: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray | None:
    if not baseline_oof.is_file():
        return None
    source = pd.read_csv(baseline_oof).set_index(ID_COL)
    if not pd.Index(validation_ids).isin(source.index).all():
        return None
    aligned = source.loc[validation_ids]
    if TARGET_COL in aligned and not np.array_equal(
        aligned[TARGET_COL].to_numpy(), truth
    ):
        raise ValueError("existing 2024 baseline OOF target does not match current train.csv")
    return aligned["prediction"].to_numpy(float)


def fit_component_prediction(
    train: pd.DataFrame,
    feature_columns: list[str],
    validation_season: int,
    component: str,
    time_decay: float,
    hist_weight: float,
    hist_max_iter: int,
    n_estimators: int,
    random_state: int,
    cache_dir: Path,
    baseline_oof_2024: Path,
    smoothing_lambdas: tuple[float, ...] = (),
) -> tuple[np.ndarray, float, str]:
    train_mask = train["season"] < validation_season
    validation_mask = train["season"] == validation_season
    validation_indices = np.flatnonzero(validation_mask.to_numpy())
    truth = train.loc[validation_mask, TARGET_COL].to_numpy()
    validation_ids = train.loc[validation_mask, ID_COL].to_numpy()

    destination = cache_file(
        cache_dir,
        validation_season,
        component,
        time_decay,
        hist_max_iter,
        n_estimators,
        smoothing_lambdas,
    )
    if destination.is_file():
        cached = np.load(destination)
        if np.array_equal(cached["validation_indices"], validation_indices):
            return cached["prediction"].astype(float), 0.0, "cache"

    if (
        validation_season == 2024
        and component == "full"
        and hist_max_iter == 300
        and not smoothing_lambdas
    ):
        existing = load_existing_2024_full_prediction(
            baseline_oof_2024, validation_ids, truth
        )
        if existing is not None:
            np.savez_compressed(
                destination,
                validation_indices=validation_indices,
                prediction=existing.astype(np.float32),
            )
            return existing, 0.0, "existing_2024_oof"

    latest_training_season = validation_season - 1
    training_seasons = train.loc[train_mask, "season"].to_numpy()
    masks = TemporalWindowEnsemble.component_masks(
        training_seasons, latest_training_season
    )
    component_mask_within_train = masks[component]
    training_indices = np.flatnonzero(train_mask.to_numpy())[component_mask_within_train]
    sample_weight = None
    if component == "time_weighted":
        sample_weight = TemporalWindowEnsemble.temporal_sample_weight(
            train.loc[training_indices, "season"].to_numpy(),
            latest_training_season,
            time_decay,
        )
    model = OptimizedBaseballEnsemble(
        hist_weight=hist_weight,
        hist_max_iter=hist_max_iter,
        n_estimators=n_estimators,
        random_state=random_state,
        smoothing_lambdas=smoothing_lambdas,
    )
    started = time.perf_counter()
    model.fit(
        train.loc[training_indices, feature_columns],
        train.loc[training_indices, TARGET_COL].to_numpy(),
        sample_weight=sample_weight,
    )
    fit_seconds = time.perf_counter() - started
    prediction = model.predict_proba(train.loc[validation_mask, feature_columns])[:, 1]
    np.savez_compressed(
        destination,
        validation_indices=validation_indices,
        prediction=np.asarray(prediction, dtype=np.float32),
    )
    del model
    gc.collect()
    return np.asarray(prediction, dtype=float), fit_seconds, "fitted"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--validation-seasons", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument("--development-seasons", type=int, nargs="+", default=[2022, 2023])
    parser.add_argument("--outer-season", type=int, default=2024)
    parser.add_argument("--development-season-weights", type=float, nargs="+", default=[0.4, 0.6])
    parser.add_argument("--final-season-weights", type=float, nargs="+", default=[0.2, 0.3, 0.5])
    parser.add_argument("--stability-penalty", type=float, default=0.10)
    parser.add_argument("--fixed-development-weights", type=float, nargs=3)
    parser.add_argument("--fixed-final-weights", type=float, nargs=3)
    parser.add_argument("--hist-weight", type=float, default=0.45)
    parser.add_argument("--hist-max-iter", type=int, default=300)
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--smoothing-lambdas", type=float, nargs="*", default=[])
    parser.add_argument("--min-outer-brier-improvement", type=float, default=0.00002)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("artifacts/optimized_ensemble_2024/run_summary.json"),
    )
    parser.add_argument(
        "--baseline-oof-2024",
        type=Path,
        default=Path("artifacts/optimized_ensemble_2024/oof_predictions.csv"),
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/temporal_ensemble")
    )
    parser.add_argument("--model-output", type=Path, default=Path("model/final_model.pkl"))
    parser.add_argument(
        "--backup-output", type=Path, default=Path("model/final_model_single_window.pkl")
    )
    parser.add_argument("--skip-final-fit", action="store_true")
    args = parser.parse_args()

    validation_seasons = tuple(int(value) for value in args.validation_seasons)
    development_seasons = tuple(int(value) for value in args.development_seasons)
    if args.outer_season in development_seasons:
        raise ValueError("outer season must not be used to select development weights")
    if tuple(validation_seasons) != tuple(sorted(validation_seasons)):
        raise ValueError("validation seasons must be increasing")

    baseline_summary = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
    feature_columns = list(baseline_summary["feature_columns"])
    train = pd.read_csv(
        args.data_dir / "train.csv",
        usecols=[ID_COL, TARGET_COL, *feature_columns],
    )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.artifact_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    selected_decay = 0.5  # Unused; kept only so existing non-weighted OOF caches remain reusable.

    oof_parts = []
    component_metric_rows = []
    for validation_season in validation_seasons:
        validation_mask = train["season"] == validation_season
        fold = pd.DataFrame(
            {
                ID_COL: train.loc[validation_mask, ID_COL].to_numpy(),
                TARGET_COL: train.loc[validation_mask, TARGET_COL].to_numpy(),
                "validation_season": validation_season,
            }
        )
        training_seasons = train.loc[train["season"] < validation_season, "season"].to_numpy()
        fold_masks = TemporalWindowEnsemble.component_masks(
            training_seasons, validation_season - 1
        )
        predictions: dict[str, np.ndarray] = {}
        for component in ACTIVE_COMPONENT_NAMES:
            duplicate_of = None
            for prior_name in predictions:
                if np.array_equal(fold_masks[component], fold_masks[prior_name]) and component != "time_weighted":
                    duplicate_of = prior_name
                    break
            if duplicate_of is not None:
                prediction = predictions[duplicate_of].copy()
                fit_seconds = 0.0
                source = f"same_window_as_{duplicate_of}"
            else:
                prediction, fit_seconds, source = fit_component_prediction(
                    train,
                    feature_columns,
                    validation_season,
                    component,
                    selected_decay,
                    args.hist_weight,
                    args.hist_max_iter,
                    args.n_estimators,
                    args.random_state,
                    cache_dir,
                    args.baseline_oof_2024,
                    tuple(args.smoothing_lambdas),
                )
            predictions[component] = prediction
            fold[component] = prediction
            truth = fold[TARGET_COL].to_numpy()
            component_metric_rows.append(
                {
                    "validation_season": validation_season,
                    "component": component,
                    "brier": brier_score(truth, prediction),
                    "competition_score": competition_score(truth, prediction),
                    "prediction_mean": float(prediction.mean()),
                    "fit_seconds": fit_seconds,
                    "source": source,
                }
            )
            print(
                f"season={validation_season} component={component} "
                f"brier={component_metric_rows[-1]['brier']:.8f} "
                f"score={component_metric_rows[-1]['competition_score']:.5f} "
                f"fit={fit_seconds:.1f}s source={source}",
                flush=True,
            )
        oof_parts.append(fold)

    oof = pd.concat(oof_parts, ignore_index=True)
    if args.fixed_development_weights is None:
        development_weights, development_search = select_continuous_blend_weights(
            oof,
            development_seasons,
            tuple(args.development_season_weights),
            args.stability_penalty,
        )
    else:
        development_weights, development_search = fixed_blend_weights(
            args.fixed_development_weights, "fixed-development-weights"
        )
    if args.fixed_final_weights is None:
        final_weights, final_search = select_continuous_blend_weights(
            oof,
            validation_seasons,
            tuple(args.final_season_weights),
            args.stability_penalty,
        )
    else:
        final_weights, final_search = fixed_blend_weights(
            args.fixed_final_weights, "fixed-final-weights"
        )
    development_search.to_csv(
        args.artifact_dir / "development_weight_optimization.csv", index=False
    )
    final_search.to_csv(
        args.artifact_dir / "final_weight_optimization.csv", index=False
    )

    matrix = oof.loc[:, ACTIVE_COMPONENT_NAMES].to_numpy(float)
    oof["development_blend"] = matrix @ np.asarray(development_weights[:3])
    oof["final_blend_diagnostic"] = matrix @ np.asarray(final_weights[:3])
    oof.to_csv(args.artifact_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(component_metric_rows).to_csv(
        args.artifact_dir / "component_metrics.csv", index=False
    )

    outer_mask = oof["validation_season"] == args.outer_season
    outer_truth = oof.loc[outer_mask, TARGET_COL].to_numpy(float)
    outer_baseline = oof.loc[outer_mask, "full"].to_numpy(float)
    outer_candidate = oof.loc[outer_mask, "development_blend"].to_numpy(float)
    comparison = paired_brier_comparison(outer_truth, outer_baseline, outer_candidate)
    baseline_brier = brier_score(outer_truth, outer_baseline)
    candidate_brier = brier_score(outer_truth, outer_candidate)
    improvement = baseline_brier - candidate_brier
    passes_outer = (
        improvement >= args.min_outer_brier_improvement
        and comparison["paired_ci95_high"] < 0.0
    )

    blend_metric_rows = []
    for season in validation_seasons:
        mask = oof["validation_season"] == season
        truth = oof.loc[mask, TARGET_COL].to_numpy(float)
        for name in ("full", "development_blend", "final_blend_diagnostic"):
            prediction = oof.loc[mask, name].to_numpy(float)
            blend_metric_rows.append(
                {
                    "validation_season": season,
                    "model": name,
                    "brier": brier_score(truth, prediction),
                    "competition_score": competition_score(truth, prediction),
                    "prediction_mean": float(prediction.mean()),
                }
            )
    pd.DataFrame(blend_metric_rows).to_csv(
        args.artifact_dir / "blend_metrics.csv", index=False
    )

    summary = {
        "status": "promoted_temporal_ensemble" if passes_outer else "rejected_keep_single_window",
        "validation_seasons": list(validation_seasons),
        "development_seasons": list(development_seasons),
        "outer_season": args.outer_season,
        "weight_strategy": (
            "fixed_from_prior_temporal_ensemble"
            if args.fixed_development_weights is not None
            else "continuous SLSQP with non-negative simplex constraints"
        ),
        "time_weighted_component": "excluded",
        "development_component_weights": dict(zip(COMPONENT_NAMES, development_weights)),
        "final_component_weights": dict(zip(COMPONENT_NAMES, final_weights)),
        "outer_baseline_brier": baseline_brier,
        "outer_candidate_brier": candidate_brier,
        "outer_brier_improvement": improvement,
        "outer_baseline_score": competition_score(outer_truth, outer_baseline),
        "outer_candidate_score": competition_score(outer_truth, outer_candidate),
        "paired_comparison": comparison,
        "minimum_outer_brier_improvement": args.min_outer_brier_improvement,
        "feature_columns": feature_columns,
        "hist_weight_within_each_component": args.hist_weight,
        "hist_max_iter_per_component": args.hist_max_iter,
        "extra_trees_weight_within_each_component": 1.0 - args.hist_weight,
        "n_estimators_per_component": args.n_estimators,
        "smoothing_lambdas": list(args.smoothing_lambdas),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "model_output": None,
    }

    if passes_outer and not args.skip_final_fit:
        final_model = TemporalWindowEnsemble(
            component_weights=final_weights,
            time_decay=selected_decay,
            hist_weight=args.hist_weight,
            hist_max_iter=args.hist_max_iter,
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            smoothing_lambdas=tuple(args.smoothing_lambdas),
        )
        started = time.perf_counter()
        final_model.fit(train[feature_columns], train[TARGET_COL].to_numpy())
        summary["final_fit_seconds"] = time.perf_counter() - started
        if args.model_output.is_file() and not args.backup_output.exists():
            args.backup_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.model_output, args.backup_output)
        artifact = {
            "model": final_model,
            "feature_columns": feature_columns,
            "positive_class": 1,
            "selected_model": "temporal_window_ensemble",
            "validation_seasons": list(validation_seasons),
            "full_train_rows": len(train),
            "random_state": args.random_state,
            "component_weights": dict(zip(COMPONENT_NAMES, final_weights)),
            "time_weighted_component": "excluded",
            "smoothing_lambdas": list(args.smoothing_lambdas),
        }
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, args.model_output, compress=3)
        summary["model_output"] = str(args.model_output)
        summary["backup_output"] = str(args.backup_output)

    (args.artifact_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
