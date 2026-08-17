#!/usr/bin/env python3
"""Test removing season from the official temporal HistGB + ExtraTrees model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from model.ensemble import OptimizedBaseballEnsemble
from model.oof_stacking import apply_logit_shift, robust_stack_objective
from model.temporal_ensemble import TemporalWindowEnsemble
from src.lg_aimers.metrics import brier_score, competition_score, paired_brier_comparison
from train_probability_refinement import rolling_refinement


ID_COL = "row_id"
TARGET_COL = "control_success"
CURRENT_COL = "rolling_calibrated"
ACTIVE_COMPONENTS = ("full", "recent_3", "recent_2")
VALIDATION_SEASONS = (2022, 2023, 2024)
DEVELOPMENT_SEASONS = (2022, 2023)


def without_season_feature_columns(feature_columns: list[str]) -> list[str]:
    if "season" not in feature_columns:
        raise ValueError("official feature list does not contain season")
    result = [column for column in feature_columns if column != "season"]
    if len(result) != len(feature_columns) - 1:
        raise ValueError("official feature list contains duplicate season columns")
    return result


def cache_path(
    cache_dir: Path,
    validation_season: int,
    component: str,
    feature_columns: list[str],
) -> Path:
    signature = hashlib.sha256(
        (
            "base-season-ablation-v1|"
            + str(validation_season)
            + "|"
            + component
            + "|"
            + "|".join(feature_columns)
        ).encode()
    ).hexdigest()[:12]
    return cache_dir / f"{validation_season}_{component}_{signature}.npz"


def fit_component_prediction(
    train: pd.DataFrame,
    feature_columns: list[str],
    validation_season: int,
    component: str,
    cache_dir: Path,
    random_state: int,
) -> tuple[np.ndarray, float, str]:
    training_mask = train["season"].lt(validation_season).to_numpy()
    validation_mask = train["season"].eq(validation_season).to_numpy()
    validation_indices = np.flatnonzero(validation_mask)
    destination = cache_path(
        cache_dir, validation_season, component, feature_columns
    )
    if destination.is_file():
        cached = np.load(destination)
        if np.array_equal(cached["validation_indices"], validation_indices):
            return cached["prediction"].astype(float), 0.0, "cache"

    latest_training_season = validation_season - 1
    training_seasons = train.loc[training_mask, "season"].to_numpy()
    component_mask = TemporalWindowEnsemble.component_masks(
        training_seasons, latest_training_season
    )[component]
    training_indices = np.flatnonzero(training_mask)[component_mask]
    model = OptimizedBaseballEnsemble(
        hist_weight=0.45,
        hist_max_iter=300,
        n_estimators=160,
        random_state=random_state,
        smoothing_lambdas=(),
    )
    started = time.perf_counter()
    model.fit(
        train.loc[training_indices, feature_columns],
        train.loc[training_indices, TARGET_COL].to_numpy(),
    )
    prediction = model.predict_proba(
        train.loc[validation_mask, feature_columns]
    )[:, 1]
    elapsed = time.perf_counter() - started
    np.savez_compressed(
        destination,
        validation_indices=validation_indices,
        prediction=np.asarray(prediction, dtype=np.float32),
    )
    del model
    gc.collect()
    return np.asarray(prediction, dtype=float), elapsed, "fitted"


def season_component_predictions(
    train: pd.DataFrame,
    feature_columns: list[str],
    validation_season: int,
    cache_dir: Path,
    random_state: int,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    training_seasons = train.loc[
        train["season"].lt(validation_season), "season"
    ].to_numpy()
    masks = TemporalWindowEnsemble.component_masks(
        training_seasons, validation_season - 1
    )
    predictions = {}
    fit_rows = []
    for component in ACTIVE_COMPONENTS:
        duplicate_of = next(
            (
                name
                for name in predictions
                if np.array_equal(masks[component], masks[name])
            ),
            None,
        )
        if duplicate_of is not None:
            prediction = predictions[duplicate_of].copy()
            seconds = 0.0
            source = f"same_window_as_{duplicate_of}"
        else:
            prediction, seconds, source = fit_component_prediction(
                train,
                feature_columns,
                validation_season,
                component,
                cache_dir,
                random_state,
            )
        predictions[component] = prediction
        fit_rows.append(
            {
                "validation_season": validation_season,
                "component": component,
                "fit_seconds": seconds,
                "source": source,
            }
        )
        print(
            f"season={validation_season} component={component} "
            f"source={source} seconds={seconds:.1f}",
            flush=True,
        )
    return predictions, fit_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--temporal-summary",
        type=Path,
        default=Path("artifacts/temporal_ensemble/run_summary.json"),
    )
    parser.add_argument(
        "--current-oof",
        type=Path,
        default=Path("artifacts/probability_refinement/final_comparison/oof_predictions.csv"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/base_season_ablation"),
    )
    parser.add_argument("--final-logit-shift", type=float, default=-0.0461)
    parser.add_argument("--stability-penalty", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.artifact_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporal_summary = json.loads(args.temporal_summary.read_text(encoding="utf-8"))
    official_features = list(temporal_summary["feature_columns"])
    candidate_features = without_season_feature_columns(official_features)
    weights_by_name = temporal_summary["development_component_weights"]
    component_weights = np.asarray(
        [float(weights_by_name[name]) for name in ACTIVE_COMPONENTS], dtype=float
    )
    component_weights /= component_weights.sum()
    train = pd.read_csv(
        args.data_dir / "train.csv",
        usecols=[ID_COL, TARGET_COL, *official_features],
    )
    current = pd.read_csv(args.current_oof)
    current = current[current["branch"].eq("temporal_original")][
        [ID_COL, TARGET_COL, "validation_season", CURRENT_COL]
    ].copy()

    oof_parts = []
    fit_rows = []
    for season in VALIDATION_SEASONS:
        predictions, rows = season_component_predictions(
            train,
            candidate_features,
            season,
            cache_dir,
            args.random_state,
        )
        fit_rows.extend(rows)
        mask = train["season"].eq(season)
        fold = train.loc[mask, [ID_COL, TARGET_COL, "game_type"]].copy()
        fold["validation_season"] = season
        fold["candidate_raw"] = sum(
            float(weight) * predictions[component]
            for component, weight in zip(ACTIVE_COMPONENTS, component_weights)
        )
        oof_parts.append(fold)
    candidate_oof = pd.concat(oof_parts, ignore_index=True)
    refined_input = candidate_oof[
        [ID_COL, TARGET_COL, "validation_season", "game_type", "candidate_raw"]
    ].rename(columns={"candidate_raw": "development_blend"})
    refined = rolling_refinement(
        refined_input,
        game_strength=0.10,
        game_shrinkage=100_000.0,
        calibration_strength=0.25,
        season_decay=0.6,
    )
    candidate_oof["candidate_refined"] = refined[CURRENT_COL].to_numpy(float)
    comparison = candidate_oof.merge(
        current,
        on=[ID_COL, TARGET_COL, "validation_season"],
        how="inner",
        validate="one_to_one",
    )
    if len(comparison) != len(candidate_oof):
        raise ValueError("official OOF rows do not match candidate validation rows")
    comparison["official_probability"] = apply_logit_shift(
        comparison[CURRENT_COL], args.final_logit_shift
    )
    comparison["candidate_probability"] = apply_logit_shift(
        comparison["candidate_refined"], args.final_logit_shift
    )

    metric_rows = []
    paired_by_season = {}
    for season in VALIDATION_SEASONS:
        fold = comparison[comparison["validation_season"].eq(season)]
        truth = fold[TARGET_COL].to_numpy(float)
        official = fold["official_probability"].to_numpy(float)
        candidate = fold["candidate_probability"].to_numpy(float)
        official_brier = brier_score(truth, official)
        candidate_brier = brier_score(truth, candidate)
        metric_rows.append(
            {
                "validation_season": season,
                "official_brier": official_brier,
                "candidate_brier": candidate_brier,
                "candidate_minus_official_brier": candidate_brier - official_brier,
                "official_score": competition_score(truth, official),
                "candidate_score": competition_score(truth, candidate),
                "official_prediction_mean": float(official.mean()),
                "candidate_prediction_mean": float(candidate.mean()),
                "target_mean": float(truth.mean()),
            }
        )
        paired_by_season[str(season)] = paired_brier_comparison(
            truth, official, candidate
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.artifact_dir / "season_metrics.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(args.artifact_dir / "fit_summary.csv", index=False)
    cohort_rows = []
    for (season, game_type), cohort in comparison.groupby(
        ["validation_season", "game_type"], dropna=False, sort=True
    ):
        truth = cohort[TARGET_COL].to_numpy(float)
        official = cohort["official_probability"].to_numpy(float)
        candidate = cohort["candidate_probability"].to_numpy(float)
        official_brier = brier_score(truth, official)
        candidate_brier = brier_score(truth, candidate)
        cohort_rows.append(
            {
                "validation_season": int(season),
                "game_type": str(game_type),
                "rows": len(cohort),
                "target_mean": float(truth.mean()),
                "official_prediction_mean": float(official.mean()),
                "candidate_prediction_mean": float(candidate.mean()),
                "official_brier": official_brier,
                "candidate_brier": candidate_brier,
                "candidate_minus_official_brier": candidate_brier - official_brier,
            }
        )
    cohort_metrics = pd.DataFrame(cohort_rows)
    cohort_metrics.to_csv(args.artifact_dir / "game_type_metrics.csv", index=False)

    development = comparison[
        comparison["validation_season"].isin(DEVELOPMENT_SEASONS)
    ]
    development_truth = development[TARGET_COL].to_numpy(float)
    development_seasons = development["validation_season"].to_numpy(int)
    official_objective = robust_stack_objective(
        development_truth,
        development[CURRENT_COL].to_numpy(float).reshape(-1, 1),
        development_seasons,
        np.asarray([1.0]),
        args.final_logit_shift,
        season_weights=(0.4, 0.6),
        stability_penalty=args.stability_penalty,
    )
    candidate_objective = robust_stack_objective(
        development_truth,
        development["candidate_refined"].to_numpy(float).reshape(-1, 1),
        development_seasons,
        np.asarray([1.0]),
        args.final_logit_shift,
        season_weights=(0.4, 0.6),
        stability_penalty=args.stability_penalty,
    )
    dev_metrics = metrics[metrics["validation_season"].isin(DEVELOPMENT_SEASONS)]
    development_non_degraded = bool(
        (dev_metrics["candidate_minus_official_brier"] <= 0.0).all()
    )
    development_selected = bool(
        development_non_degraded and candidate_objective < official_objective
    )
    outer_row = metrics[metrics["validation_season"].eq(2024)].iloc[0]
    outer_improved = bool(outer_row["candidate_minus_official_brier"] < 0.0)
    status = (
        "diagnostic_improvement_requires_fresh_validation"
        if development_selected and outer_improved
        else "rejected_keep_official_852_model"
    )
    summary = {
        "status": status,
        "official_leaderboard_baseline_score": 852.1984993386,
        "experiment_scope": (
            "remove season from every temporal HistGB45/ExtraTrees55 component; keep "
            "all remaining features, component weights, rolling corrections and logit shift fixed"
        ),
        "selection_seasons": list(DEVELOPMENT_SEASONS),
        "outer_diagnostic_season": 2024,
        "outer_is_reused_not_one_shot": True,
        "removed_feature": "season",
        "official_feature_count": len(official_features),
        "candidate_feature_count": len(candidate_features),
        "component_weights": dict(
            zip(ACTIVE_COMPONENTS, component_weights.tolist())
        ),
        "hist_weight": 0.45,
        "hist_max_iter": 300,
        "extra_trees_weight": 0.55,
        "extra_trees_estimators": 160,
        "fixed_final_logit_shift": args.final_logit_shift,
        "official_development_objective": official_objective,
        "candidate_development_objective": candidate_objective,
        "development_non_degraded": development_non_degraded,
        "development_selected": development_selected,
        "season_metrics": metrics.to_dict(orient="records"),
        "game_type_metrics": cohort_metrics.to_dict(orient="records"),
        "paired_comparisons": paired_by_season,
        "outer_improved": outer_improved,
        "adopted": False,
        "adoption_note": (
            "The official 852 model remains unchanged. The 2024 fold is reused, so an "
            "improvement would still need fresh confirmation; a worse result is rejected."
        ),
    }
    (args.artifact_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    comparison[
        [ID_COL, TARGET_COL, "validation_season", "official_probability", "candidate_probability"]
    ].to_csv(args.artifact_dir / "oof_predictions.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
