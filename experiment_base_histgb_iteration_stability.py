#!/usr/bin/env python3
"""Tune HistGB iterations inside the official temporal HistGB + ExtraTrees model."""

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
from model.temporal_ensemble import COMPONENT_NAMES, TemporalWindowEnsemble
from src.lg_aimers.metrics import brier_score, competition_score, paired_brier_comparison
from train_probability_refinement import rolling_refinement


ID_COL = "row_id"
TARGET_COL = "control_success"
CURRENT_COL = "rolling_calibrated"
ACTIVE_COMPONENTS = ("full", "recent_3", "recent_2")
DEVELOPMENT_SEASONS = (2022, 2023)
OUTER_SEASON = 2024


def validated_iteration_candidates(values) -> np.ndarray:
    candidates = np.asarray(values, dtype=int)
    if candidates.ndim != 1 or len(candidates) == 0:
        raise ValueError("at least one HistGB iteration candidate is required")
    if (candidates <= 0).any() or len(np.unique(candidates)) != len(candidates):
        raise ValueError("HistGB iteration candidates must be unique and positive")
    candidates.sort()
    if 300 not in candidates:
        raise ValueError("the official 300-iteration baseline must be included")
    return candidates


def curve_cache_path(
    cache_dir: Path,
    validation_season: int,
    component: str,
    feature_columns: list[str],
    candidates: np.ndarray,
) -> Path:
    signature = hashlib.sha256(
        (
            "base-histgb-iteration-stability-v1|"
            + str(validation_season)
            + "|"
            + component
            + "|"
            + ",".join(str(int(value)) for value in candidates)
            + "|"
            + "|".join(feature_columns)
        ).encode()
    ).hexdigest()[:12]
    return cache_dir / f"{validation_season}_{component}_{signature}.npz"


def fit_component_curve(
    train: pd.DataFrame,
    feature_columns: list[str],
    validation_season: int,
    component: str,
    candidates: np.ndarray,
    cache_dir: Path,
    random_state: int,
) -> tuple[np.ndarray, float, str]:
    training_mask = train["season"].lt(validation_season).to_numpy()
    validation_mask = train["season"].eq(validation_season).to_numpy()
    validation_indices = np.flatnonzero(validation_mask)
    destination = curve_cache_path(
        cache_dir,
        validation_season,
        component,
        feature_columns,
        candidates,
    )
    if destination.is_file():
        cached = np.load(destination)
        if (
            np.array_equal(cached["validation_indices"], validation_indices)
            and np.array_equal(cached["iterations"], candidates)
        ):
            return cached["predictions"].astype(float), 0.0, "cache"

    latest_training_season = validation_season - 1
    training_seasons = train.loc[training_mask, "season"].to_numpy()
    component_mask = TemporalWindowEnsemble.component_masks(
        training_seasons, latest_training_season
    )[component]
    training_indices = np.flatnonzero(training_mask)[component_mask]
    model = OptimizedBaseballEnsemble(
        hist_weight=0.45,
        hist_max_iter=int(candidates.max()),
        n_estimators=160,
        random_state=random_state,
        smoothing_lambdas=(),
    )
    started = time.perf_counter()
    model.fit(
        train.loc[training_indices, feature_columns],
        train.loc[training_indices, TARGET_COL].to_numpy(),
    )
    validation_features = train.loc[validation_mask, feature_columns]
    transformed = model.preprocessor_.transform(validation_features)
    extra_probability = model._positive_probability(model.extra_model_, transformed)
    candidate_positions = {int(value): index for index, value in enumerate(candidates)}
    predictions = np.empty((len(candidates), len(validation_indices)), dtype=np.float32)
    captured = set()
    for iteration, probability in enumerate(
        model.hist_model_.staged_predict_proba(transformed), start=1
    ):
        if iteration not in candidate_positions:
            continue
        hist_probability = probability[:, 1]
        position = candidate_positions[iteration]
        predictions[position] = 0.45 * hist_probability + 0.55 * extra_probability
        captured.add(iteration)
    if captured != set(candidate_positions):
        raise ValueError(
            f"missing staged HistGB predictions: {sorted(set(candidate_positions) - captured)}"
        )
    elapsed = time.perf_counter() - started
    np.savez_compressed(
        destination,
        validation_indices=validation_indices,
        iterations=candidates,
        predictions=predictions,
    )
    del model, transformed, extra_probability
    gc.collect()
    return predictions.astype(float), elapsed, "fitted"


def component_curves_for_season(
    train: pd.DataFrame,
    feature_columns: list[str],
    validation_season: int,
    candidates: np.ndarray,
    cache_dir: Path,
    random_state: int,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    training_seasons = train.loc[
        train["season"].lt(validation_season), "season"
    ].to_numpy()
    masks = TemporalWindowEnsemble.component_masks(
        training_seasons, validation_season - 1
    )
    curves = {}
    fit_rows = []
    for component in ACTIVE_COMPONENTS:
        duplicate_of = next(
            (
                name
                for name in curves
                if np.array_equal(masks[component], masks[name])
            ),
            None,
        )
        if duplicate_of is not None:
            curves[component] = curves[duplicate_of].copy()
            seconds = 0.0
            source = f"same_window_as_{duplicate_of}"
        else:
            curve, seconds, source = fit_component_curve(
                train,
                feature_columns,
                validation_season,
                component,
                candidates,
                cache_dir,
                random_state,
            )
            curves[component] = curve
        fit_rows.append(
            {
                "validation_season": validation_season,
                "component": component,
                "iterations": ",".join(str(int(value)) for value in candidates),
                "fit_seconds": seconds,
                "source": source,
            }
        )
        print(
            f"season={validation_season} component={component} "
            f"source={source} seconds={seconds:.1f}",
            flush=True,
        )
    return curves, fit_rows


def raw_temporal_prediction(
    curves: dict[str, np.ndarray],
    candidate_index: int,
    component_weights: np.ndarray,
) -> np.ndarray:
    return sum(
        float(weight) * curves[component][candidate_index]
        for component, weight in zip(ACTIVE_COMPONENTS, component_weights)
    )


def refine_oof(base: pd.DataFrame, raw_probability: np.ndarray) -> np.ndarray:
    source = base[
        [ID_COL, TARGET_COL, "validation_season", "game_type"]
    ].copy()
    source["development_blend"] = np.asarray(raw_probability, dtype=float)
    refined = rolling_refinement(
        source,
        game_strength=0.10,
        game_shrinkage=100_000.0,
        calibration_strength=0.25,
        season_decay=0.6,
    )
    return refined["rolling_calibrated"].to_numpy(float)


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
        default=Path("artifacts/base_histgb_iteration_stability"),
    )
    parser.add_argument(
        "--iteration-candidates",
        type=int,
        nargs="+",
        default=[50, 100, 150, 200, 250, 300, 350],
    )
    parser.add_argument("--final-logit-shift", type=float, default=-0.0461)
    parser.add_argument("--stability-penalty", type=float, default=0.25)
    parser.add_argument("--equivalence-tolerance", type=float, default=5e-6)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    candidates = validated_iteration_candidates(args.iteration_candidates)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.artifact_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporal_summary = json.loads(args.temporal_summary.read_text(encoding="utf-8"))
    feature_columns = list(temporal_summary["feature_columns"])
    weights_by_name = temporal_summary["development_component_weights"]
    component_weights = np.asarray(
        [float(weights_by_name[name]) for name in ACTIVE_COMPONENTS], dtype=float
    )
    component_weights /= component_weights.sum()

    train = pd.read_csv(
        args.data_dir / "train.csv",
        usecols=[ID_COL, TARGET_COL, *feature_columns],
    )
    current = pd.read_csv(args.current_oof)
    current = current[current["branch"].eq("temporal_original")][
        [ID_COL, TARGET_COL, "validation_season", CURRENT_COL]
    ].copy()

    season_curves = {}
    fit_rows = []
    development_parts = []
    for season in DEVELOPMENT_SEASONS:
        curves, rows = component_curves_for_season(
            train,
            feature_columns,
            season,
            candidates,
            cache_dir,
            args.random_state,
        )
        season_curves[season] = curves
        fit_rows.extend(rows)
        mask = train["season"].eq(season)
        fold = train.loc[mask, [ID_COL, TARGET_COL, "game_type"]].copy()
        fold["validation_season"] = season
        development_parts.append(fold)
    development = pd.concat(development_parts, ignore_index=True)
    development = development.merge(
        current,
        on=[ID_COL, TARGET_COL, "validation_season"],
        how="inner",
        validate="one_to_one",
    )
    if len(development) != sum(len(value) for value in development_parts):
        raise ValueError("official OOF rows do not match the development validation rows")

    curve_rows = []
    refined_by_iteration = {}
    raw_by_iteration = {}
    baseline_shifted = apply_logit_shift(
        development[CURRENT_COL], args.final_logit_shift
    )
    for candidate_index, iterations in enumerate(candidates):
        raw = np.concatenate(
            [
                raw_temporal_prediction(
                    season_curves[season], candidate_index, component_weights
                )
                for season in DEVELOPMENT_SEASONS
            ]
        )
        refined = refine_oof(development, raw)
        shifted = apply_logit_shift(refined, args.final_logit_shift)
        raw_by_iteration[int(iterations)] = raw
        refined_by_iteration[int(iterations)] = refined
        fold_deltas = []
        for season in DEVELOPMENT_SEASONS:
            fold = development["validation_season"].eq(season).to_numpy()
            candidate_brier = brier_score(development.loc[fold, TARGET_COL], shifted[fold])
            baseline_brier = brier_score(
                development.loc[fold, TARGET_COL], baseline_shifted[fold]
            )
            delta = candidate_brier - baseline_brier
            fold_deltas.append(delta)
            curve_rows.append(
                {
                    "validation_season": season,
                    "hist_max_iter": int(iterations),
                    "brier": candidate_brier,
                    "official_brier": baseline_brier,
                    "brier_delta": delta,
                    "competition_score": competition_score(
                        development.loc[fold, TARGET_COL], shifted[fold]
                    ),
                }
            )
        objective = robust_stack_objective(
            development[TARGET_COL].to_numpy(float),
            refined.reshape(-1, 1),
            development["validation_season"].to_numpy(int),
            np.asarray([1.0]),
            args.final_logit_shift,
            season_weights=(0.4, 0.6),
            stability_penalty=args.stability_penalty,
        )
        for row in curve_rows[-len(DEVELOPMENT_SEASONS):]:
            row["robust_objective"] = objective
            row["all_development_folds_non_degraded"] = max(fold_deltas) <= 1e-12

    curve_metrics = pd.DataFrame(curve_rows)
    curve_metrics.to_csv(args.artifact_dir / "development_curve_metrics.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(args.artifact_dir / "fit_summary.csv", index=False)

    official_refined = development[CURRENT_COL].to_numpy(float)
    candidate_300 = refined_by_iteration[300]
    equivalence = {
        "max_absolute_probability_difference": float(
            np.max(np.abs(candidate_300 - official_refined))
        ),
        "mean_absolute_probability_difference": float(
            np.mean(np.abs(candidate_300 - official_refined))
        ),
        "brier_difference": float(
            brier_score(development[TARGET_COL], apply_logit_shift(candidate_300, args.final_logit_shift))
            - brier_score(development[TARGET_COL], baseline_shifted)
        ),
    }
    if equivalence["max_absolute_probability_difference"] > args.equivalence_tolerance:
        raise ValueError(
            "reconstructed 300-iteration baseline does not match current OOF: "
            + json.dumps(equivalence)
        )

    candidate_summary = (
        curve_metrics.groupby("hist_max_iter", as_index=False)
        .agg(
            robust_objective=("robust_objective", "first"),
            maximum_fold_brier_delta=("brier_delta", "max"),
            all_development_folds_non_degraded=(
                "all_development_folds_non_degraded", "first"
            ),
        )
        .sort_values(["all_development_folds_non_degraded", "robust_objective"], ascending=[False, True])
    )
    feasible = candidate_summary[
        candidate_summary["all_development_folds_non_degraded"]
    ]
    if feasible.empty:
        selected_iterations = 300
    else:
        selected_iterations = int(feasible.iloc[0]["hist_max_iter"])
    candidate_summary["selected"] = candidate_summary["hist_max_iter"].eq(
        selected_iterations
    )
    candidate_summary.to_csv(args.artifact_dir / "candidate_selection.csv", index=False)

    outer_candidates = np.asarray([selected_iterations], dtype=int)
    outer_curves, outer_fit_rows = component_curves_for_season(
        train,
        feature_columns,
        OUTER_SEASON,
        outer_candidates,
        cache_dir,
        args.random_state,
    )
    fit_rows.extend(outer_fit_rows)
    pd.DataFrame(fit_rows).to_csv(args.artifact_dir / "fit_summary.csv", index=False)
    outer_mask = train["season"].eq(OUTER_SEASON)
    outer_frame = train.loc[
        outer_mask, [ID_COL, TARGET_COL, "game_type"]
    ].copy()
    outer_frame["validation_season"] = OUTER_SEASON
    combined = pd.concat([development.drop(columns=[CURRENT_COL]), outer_frame], ignore_index=True)
    outer_raw = raw_temporal_prediction(outer_curves, 0, component_weights)
    combined_raw = np.concatenate(
        [raw_by_iteration[selected_iterations], outer_raw]
    )
    combined_refined = refine_oof(combined, combined_raw)
    selected_outer_refined = combined_refined[len(development):]
    outer_current = current[current["validation_season"].eq(OUTER_SEASON)].copy()
    outer = outer_frame.merge(
        outer_current,
        on=[ID_COL, TARGET_COL, "validation_season"],
        how="inner",
        validate="one_to_one",
    )
    if len(outer) != len(outer_frame):
        raise ValueError("official 2024 OOF rows do not match validation rows")
    truth = outer[TARGET_COL].to_numpy(float)
    baseline_probability = apply_logit_shift(
        outer[CURRENT_COL], args.final_logit_shift
    )
    candidate_probability = apply_logit_shift(
        selected_outer_refined, args.final_logit_shift
    )
    baseline_brier = brier_score(truth, baseline_probability)
    candidate_brier = brier_score(truth, candidate_probability)
    paired = paired_brier_comparison(
        truth, baseline_probability, candidate_probability
    )
    diagnostic_improved = candidate_brier < baseline_brier

    outer_metrics = pd.DataFrame(
        [
            {
                "model": "official_852_baseline",
                "hist_max_iter": 300,
                "brier": baseline_brier,
                "competition_score": competition_score(truth, baseline_probability),
                "prediction_mean": float(baseline_probability.mean()),
            },
            {
                "model": "selected_base_histgb_candidate",
                "hist_max_iter": selected_iterations,
                "brier": candidate_brier,
                "competition_score": competition_score(truth, candidate_probability),
                "prediction_mean": float(candidate_probability.mean()),
            },
        ]
    )
    outer_metrics.to_csv(args.artifact_dir / "outer_diagnostic_metrics.csv", index=False)
    outer[[ID_COL, TARGET_COL, "validation_season"]].assign(
        official_852_probability=baseline_probability,
        selected_candidate_probability=candidate_probability,
    ).to_csv(args.artifact_dir / "outer_predictions.csv", index=False)

    summary = {
        "status": (
            "diagnostic_improvement_requires_fresh_validation"
            if diagnostic_improved and selected_iterations != 300
            else "keep_official_852_model"
        ),
        "official_leaderboard_baseline_score": 852.1984993386,
        "experiment_scope": (
            "change HistGB max_iter inside every temporal HistGB45/ExtraTrees55 component; "
            "keep ExtraTrees=160, temporal weights, rolling corrections and logit shift fixed"
        ),
        "selection_seasons": list(DEVELOPMENT_SEASONS),
        "outer_diagnostic_season": OUTER_SEASON,
        "outer_is_reused_not_one_shot": True,
        "iteration_candidates": candidates.tolist(),
        "official_hist_max_iter": 300,
        "selected_hist_max_iter": selected_iterations,
        "component_weights": dict(
            zip(ACTIVE_COMPONENTS, component_weights.tolist())
        ),
        "hist_weight": 0.45,
        "extra_trees_weight": 0.55,
        "extra_trees_estimators": 160,
        "fixed_final_logit_shift": args.final_logit_shift,
        "development_gate": "no Brier degradation in either 2022 or 2023",
        "baseline_reconstruction_equivalence": equivalence,
        "candidate_selection": candidate_summary.to_dict(orient="records"),
        "outer_metrics": outer_metrics.to_dict(orient="records"),
        "outer_brier_delta": candidate_brier - baseline_brier,
        "outer_paired_comparison": paired,
        "diagnostic_improved": diagnostic_improved,
        "adopted": False,
        "adoption_note": (
            "The official 852 model stays unchanged. The reused 2024 diagnostic cannot "
            "authorize promotion; a worse candidate is rejected immediately."
        ),
    }
    (args.artifact_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
