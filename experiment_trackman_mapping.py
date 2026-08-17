#!/usr/bin/env python3
"""Evaluate a leakage-safe Trackman pitcher-feature branch on the 852-point model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from experiment_new_pitcher_fallback import apply_logit_shift
from model.ensemble import OptimizedBaseballEnsemble
from model.temporal_ensemble import TemporalWindowEnsemble
from model.trackman_features import (
    MAIN_RATE_COLUMNS,
    PHYSICAL_COLUMNS,
    TrackmanMatchThresholds,
    add_temporal_trackman_features,
    trackman_feature_columns,
)
from src.lg_aimers.metrics import brier_score, competition_score, paired_brier_comparison
from train_probability_refinement import rolling_refinement


ID_COL = "row_id"
TARGET_COL = "control_success"
ACTIVE_COMPONENTS = ("full", "recent_3", "recent_2")
CURRENT_COL = "rolling_calibrated"
TRACKMAN_COL = "trackman_rolling_calibrated"


def normalized_fold_brier(truth, prediction) -> float:
    truth = np.asarray(truth, dtype=float)
    event_rate = float(truth.mean())
    reference = event_rate * (1.0 - event_rate)
    if reference <= 0.0:
        raise ValueError("fold must contain both target classes")
    return brier_score(truth, prediction) / reference


def development_objective(
    frame: pd.DataFrame,
    prediction,
    season_weights=(0.4, 0.6),
    stability_penalty: float = 0.10,
) -> float:
    prediction = np.asarray(prediction, dtype=float)
    seasons = sorted(int(value) for value in frame["validation_season"].unique())
    weights = np.asarray(season_weights[-len(seasons):], dtype=float)
    weights /= weights.sum()
    losses = []
    season_values = frame["validation_season"].to_numpy(int)
    truth = frame[TARGET_COL].to_numpy(float)
    for season in seasons:
        mask = season_values == season
        losses.append(normalized_fold_brier(truth[mask], prediction[mask]))
    losses = np.asarray(losses, dtype=float)
    mean = float(np.dot(weights, losses))
    variance = float(np.dot(weights, np.square(losses - mean)))
    return mean + float(stability_penalty) * np.sqrt(variance)


def select_trackman_weight(
    development: pd.DataFrame,
    final_logit_shift: float,
) -> tuple[float, pd.DataFrame]:
    current = development[CURRENT_COL].to_numpy(float)
    trackman = development[TRACKMAN_COL].to_numpy(float)

    def evaluate(weight: float) -> float:
        mixed = (1.0 - weight) * current + weight * trackman
        shifted = apply_logit_shift(mixed, final_logit_shift)
        return development_objective(development, shifted)

    result = minimize_scalar(
        evaluate,
        method="bounded",
        bounds=(0.0, 1.0),
        options={"xatol": 1e-8, "maxiter": 500},
    )
    candidates = [(0.0, evaluate(0.0)), (1.0, evaluate(1.0))]
    if result.success:
        candidates.append((float(result.x), float(result.fun)))
    weight, objective = min(candidates, key=lambda item: item[1])
    diagnostics = pd.DataFrame(
        [
            {
                "optimizer": "bounded_scalar",
                "success": bool(result.success),
                "message": str(result.message),
                "iterations": int(result.nfev),
                "trackman_weight": weight,
                "current_weight": 1.0 - weight,
                "objective": objective,
                "baseline_objective": evaluate(0.0),
                "direct_trackman_objective": evaluate(1.0),
            }
        ]
    )
    return weight, diagnostics


def mapping_stability(mapping: pd.DataFrame, cutoffs=(2022, 2023, 2024)) -> dict:
    accepted = mapping.loc[mapping["high_confidence"]].copy()
    maps = {
        cutoff: dict(
            zip(
                accepted.loc[accepted["cutoff_season"] == cutoff, "pitcher_id"],
                accepted.loc[
                    accepted["cutoff_season"] == cutoff, "pitcher_trackman_id"
                ],
            )
        )
        for cutoff in cutoffs
    }
    stable = 0
    compared = 0
    rows = []
    for earlier, later in zip(cutoffs[:-1], cutoffs[1:]):
        common = set(maps[earlier]) & set(maps[later])
        same = sum(maps[earlier][pitcher] == maps[later][pitcher] for pitcher in common)
        stable += same
        compared += len(common)
        rows.append(
            {
                "earlier_cutoff": earlier,
                "later_cutoff": later,
                "common_mappings": len(common),
                "stable_mappings": same,
                "stability_rate": same / len(common) if common else np.nan,
            }
        )
    return {
        "stable_mappings": stable,
        "compared_mappings": compared,
        "stability_rate": stable / compared if compared else np.nan,
        "adjacent_cutoff_rows": rows,
    }


def cache_file(
    cache_dir: Path,
    validation_season: int,
    component: str,
    feature_columns: list[str],
    n_estimators: int,
) -> Path:
    signature = hashlib.sha256(
        (
            "trackman-v1|"
            + str(validation_season)
            + "|"
            + component
            + "|"
            + str(n_estimators)
            + "|"
            + "|".join(feature_columns)
        ).encode()
    ).hexdigest()[:12]
    return cache_dir / f"{validation_season}_{component}_{signature}.npz"


def fit_component(
    train: pd.DataFrame,
    feature_columns: list[str],
    validation_season: int,
    component: str,
    cache_dir: Path,
    n_estimators: int,
) -> tuple[np.ndarray, float, str]:
    train_mask = train["season"] < validation_season
    validation_mask = train["season"] == validation_season
    validation_indices = np.flatnonzero(validation_mask.to_numpy())
    destination = cache_file(
        cache_dir, validation_season, component, feature_columns, n_estimators
    )
    if destination.is_file():
        cached = np.load(destination)
        if np.array_equal(cached["validation_indices"], validation_indices):
            return cached["prediction"].astype(float), 0.0, "cache"

    latest_training_season = validation_season - 1
    training_indices = np.flatnonzero(train_mask.to_numpy())
    component_masks = TemporalWindowEnsemble.component_masks(
        train.loc[train_mask, "season"].to_numpy(), latest_training_season
    )
    selected_indices = training_indices[component_masks[component]]
    model = OptimizedBaseballEnsemble(
        hist_weight=0.45,
        n_estimators=n_estimators,
        random_state=42,
        smoothing_lambdas=(),
    )
    started = time.perf_counter()
    model.fit(
        train.loc[selected_indices, feature_columns],
        train.loc[selected_indices, TARGET_COL].to_numpy(),
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


def cohort_metrics(frame: pd.DataFrame, prediction) -> dict[str, float]:
    truth = frame[TARGET_COL].to_numpy(float)
    prediction = np.asarray(prediction, dtype=float)
    mapped = frame["tm_is_mapped"].to_numpy(float) > 0.5
    values = {
        "brier": brier_score(truth, prediction),
        "competition_score": competition_score(truth, prediction),
        "prediction_mean": float(prediction.mean()),
        "mapped_row_rate": float(mapped.mean()),
    }
    values["mapped_pitcher_brier"] = (
        brier_score(truth[mapped], prediction[mapped]) if mapped.any() else np.nan
    )
    values["unmapped_pitcher_brier"] = (
        brier_score(truth[~mapped], prediction[~mapped]) if (~mapped).any() else np.nan
    )
    return values


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
        "--artifact-dir", type=Path, default=Path("artifacts/trackman_mapping")
    )
    parser.add_argument("--validation-seasons", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument("--development-seasons", type=int, nargs="+", default=[2022, 2023])
    parser.add_argument("--outer-season", type=int, default=2024)
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--trackman-shrinkage", type=float, default=200.0)
    parser.add_argument("--final-logit-shift", type=float, default=-0.0461)
    parser.add_argument("--season-decay", type=float, default=0.6)
    parser.add_argument("--game-strength", type=float, default=0.10)
    parser.add_argument("--game-shrinkage", type=float, default=100_000.0)
    parser.add_argument("--calibration-strength", type=float, default=0.25)
    args = parser.parse_args()

    summary = json.loads(args.temporal_summary.read_text(encoding="utf-8"))
    base_features = list(summary["feature_columns"])
    weights = np.asarray(
        [summary["development_component_weights"][name] for name in ACTIVE_COMPONENTS],
        dtype=float,
    )
    weights /= weights.sum()
    mapping_main_columns = [
        "pitcher_id",
        "asof_pitcher_pitchmix_n",
        *MAIN_RATE_COLUMNS,
    ]
    read_columns = list(
        dict.fromkeys([ID_COL, TARGET_COL, *base_features, *mapping_main_columns])
    )
    train = pd.read_csv(args.data_dir / "train.csv", usecols=read_columns)
    track_columns = [
        "season",
        "pitcher_trackman_id",
        "pitcher_hand",
        "pitch_type_group",
        *PHYSICAL_COLUMNS,
    ]
    trackman = pd.read_csv(args.data_dir / "trackman_history.csv", usecols=track_columns)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.artifact_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    thresholds = TrackmanMatchThresholds()
    enriched, mapping = add_temporal_trackman_features(
        train,
        trackman,
        thresholds=thresholds,
        shrinkage=args.trackman_shrinkage,
    )
    mapping.to_csv(args.artifact_dir / "mapping_assignments.csv", index=False)
    tm_features = trackman_feature_columns()
    feature_columns = [*base_features, *tm_features]

    mapping_rows = []
    for season in sorted(int(value) for value in train["season"].unique()):
        assignments = mapping[mapping["cutoff_season"] == season]
        accepted = assignments.loc[
            assignments["high_confidence"].astype(bool)
        ]
        season_mask = enriched["season"] == season
        mapping_rows.append(
            {
                "cutoff_season": season,
                "candidate_assignments": len(assignments),
                "accepted_mappings": len(accepted),
                "accepted_main_history_rows": float(accepted["main_history_n"].sum()),
                "mapped_validation_rows": int(
                    (enriched.loc[season_mask, "tm_is_mapped"] > 0.5).sum()
                ),
                "validation_rows": int(season_mask.sum()),
                "mapped_validation_row_rate": float(
                    enriched.loc[season_mask, "tm_is_mapped"].mean()
                ),
            }
        )
    mapping_summary = pd.DataFrame(mapping_rows)
    mapping_summary.to_csv(args.artifact_dir / "mapping_summary.csv", index=False)
    stability = mapping_stability(mapping)

    oof_parts = []
    fit_rows = []
    for validation_season in args.validation_seasons:
        validation_mask = enriched["season"] == validation_season
        fold = pd.DataFrame(
            {
                ID_COL: enriched.loc[validation_mask, ID_COL].to_numpy(),
                TARGET_COL: enriched.loc[validation_mask, TARGET_COL].to_numpy(),
                "validation_season": validation_season,
                "game_type": enriched.loc[validation_mask, "game_type"].to_numpy(),
                "tm_is_mapped": enriched.loc[validation_mask, "tm_is_mapped"].to_numpy(),
            }
        )
        training_seasons = enriched.loc[
            enriched["season"] < validation_season, "season"
        ].to_numpy()
        component_masks = TemporalWindowEnsemble.component_masks(
            training_seasons, validation_season - 1
        )
        predictions: dict[str, np.ndarray] = {}
        for component in ACTIVE_COMPONENTS:
            duplicate_of = next(
                (
                    prior
                    for prior in predictions
                    if np.array_equal(component_masks[component], component_masks[prior])
                ),
                None,
            )
            if duplicate_of is not None:
                prediction = predictions[duplicate_of].copy()
                fit_seconds = 0.0
                source = f"same_window_as_{duplicate_of}"
            else:
                prediction, fit_seconds, source = fit_component(
                    enriched,
                    feature_columns,
                    validation_season,
                    component,
                    cache_dir,
                    args.n_estimators,
                )
            predictions[component] = prediction
            fit_rows.append(
                {
                    "validation_season": validation_season,
                    "component": component,
                    "brier": brier_score(fold[TARGET_COL], prediction),
                    "competition_score": competition_score(fold[TARGET_COL], prediction),
                    "fit_seconds": fit_seconds,
                    "source": source,
                }
            )
            print(
                f"season={validation_season} component={component} "
                f"brier={fit_rows[-1]['brier']:.8f} fit={fit_seconds:.1f}s "
                f"source={source}",
                flush=True,
            )
        fold["development_blend"] = np.column_stack(
            [predictions[name] for name in ACTIVE_COMPONENTS]
        ) @ weights
        oof_parts.append(fold)

    raw_trackman_oof = pd.concat(oof_parts, ignore_index=True)
    refined = rolling_refinement(
        raw_trackman_oof,
        game_strength=args.game_strength,
        game_shrinkage=args.game_shrinkage,
        calibration_strength=args.calibration_strength,
        season_decay=args.season_decay,
    ).rename(columns={"rolling_calibrated": TRACKMAN_COL})

    current = pd.read_csv(args.current_oof)
    current = current[current["branch"].eq("temporal_original")][
        [ID_COL, TARGET_COL, "validation_season", CURRENT_COL]
    ]
    comparison = current.merge(
        refined[[ID_COL, TRACKMAN_COL, "tm_is_mapped"]],
        on=ID_COL,
        how="inner",
        validate="one_to_one",
    )
    if len(comparison) != len(current):
        raise ValueError("Trackman and current OOF row sets do not match")

    development = comparison[
        comparison["validation_season"].isin(args.development_seasons)
    ].reset_index(drop=True)
    selected_weight, selection = select_trackman_weight(
        development, args.final_logit_shift
    )
    selection.to_csv(args.artifact_dir / "blend_selection.csv", index=False)

    outer = comparison[comparison["validation_season"] == args.outer_season].copy()
    baseline = apply_logit_shift(outer[CURRENT_COL], args.final_logit_shift)
    direct_trackman = apply_logit_shift(outer[TRACKMAN_COL], args.final_logit_shift)
    mixed = (
        (1.0 - selected_weight) * outer[CURRENT_COL].to_numpy(float)
        + selected_weight * outer[TRACKMAN_COL].to_numpy(float)
    )
    candidate = apply_logit_shift(mixed, args.final_logit_shift)
    baseline_metrics = cohort_metrics(outer, baseline)
    direct_metrics = cohort_metrics(outer, direct_trackman)
    candidate_metrics = cohort_metrics(outer, candidate)
    outer_metrics = pd.DataFrame(
        [
            {"model": "official_852_structure_baseline", **baseline_metrics},
            {"model": "direct_trackman_augmented_branch", **direct_metrics},
            {"model": "selected_trackman_blend", **candidate_metrics},
        ]
    )
    outer_metrics.to_csv(args.artifact_dir / "outer_metrics.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(
        args.artifact_dir / "component_metrics.csv", index=False
    )
    outer.assign(
        baseline_probability=baseline,
        direct_trackman_probability=direct_trackman,
        candidate_probability=candidate,
    ).to_csv(args.artifact_dir / "outer_predictions.csv", index=False)

    truth = outer[TARGET_COL].to_numpy(float)
    paired = paired_brier_comparison(truth, baseline, candidate)
    mapped_not_worse = (
        np.isnan(candidate_metrics["mapped_pitcher_brier"])
        or candidate_metrics["mapped_pitcher_brier"]
        <= baseline_metrics["mapped_pitcher_brier"]
    )
    accepted = bool(
        selected_weight > 1e-4
        and candidate_metrics["brier"] < baseline_metrics["brier"]
        and paired["paired_ci95_high"] < 0.0
        and mapped_not_worse
        and stability["stability_rate"] >= 0.97
    )
    run_summary = {
        "status": (
            "accepted_trackman_candidate_requires_rule_confirmation"
            if accepted
            else "rejected_keep_official_852_model"
        ),
        "official_leaderboard_baseline_score": 852.1984993386,
        "fixed_final_logit_shift": args.final_logit_shift,
        "architecture": (
            "official temporal refined model + leakage-safe Trackman augmented temporal "
            "branch -> development-selected probability blend -> fixed logit shift"
        ),
        "mapping_rule": {
            **thresholds.__dict__,
            "one_to_one_assignment": "Hungarian within pitcher hand",
            "validation_rule": "mapping and features use only seasons before each row season",
        },
        "mapping_stability": stability,
        "mapping_summary": mapping_summary.to_dict(orient="records"),
        "trackman_feature_columns": tm_features,
        "trackman_shrinkage": args.trackman_shrinkage,
        "selected_trackman_weight": selected_weight,
        "selected_current_weight": 1.0 - selected_weight,
        "outer_baseline": baseline_metrics,
        "outer_direct_trackman": direct_metrics,
        "outer_candidate": candidate_metrics,
        "paired_comparison_candidate_vs_baseline": paired,
        "adoption_rule": (
            "positive development-selected Trackman weight; shifted 2024 overall Brier "
            "improves with paired CI below zero; mapped cohort not worse; historical "
            "mapping stability at least 97%"
        ),
        "rule_note": (
            "Do not create or submit a final ZIP until anonymous-ID probabilistic mapping "
            "is confirmed as allowed by the competition operator."
        ),
    }
    (args.artifact_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
