#!/usr/bin/env python3
"""Evaluate a context-only fallback for new and low-sample pitchers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_hierarchical_calibration import add_pitcher_cohorts
from model.ensemble import OptimizedBaseballEnsemble
from src.lg_aimers.metrics import brier_score, competition_score, paired_brier_comparison
from train_probability_refinement import rolling_refinement


ID_COL = "row_id"
TARGET_COL = "control_success"
CURRENT_COL = "rolling_calibrated"
FALLBACK_RAW_COL = "fallback_raw"
FALLBACK_REFINED_COL = "fallback_refined"
DEFAULT_LAMBDAS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
DEFAULT_THRESHOLDS = (25.0, 50.0, 100.0, 200.0)


def apply_logit_shift(probability, shift: float) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-9, 1.0 - 1e-9)
    logit = np.log(probability / (1.0 - probability))
    return 1.0 / (1.0 + np.exp(-(logit + float(shift))))


def gate_weight(
    count,
    newcomer,
    mode: str,
    lambda_value: float,
    low_sample_threshold: float = 100.0,
) -> np.ndarray:
    count = np.asarray(count, dtype=float)
    newcomer = np.asarray(newcomer, dtype=bool)
    safe_count = np.nan_to_num(count, nan=0.0, posinf=0.0, neginf=0.0)
    safe_count = np.clip(safe_count, 0.0, None)
    if lambda_value < 0.0:
        raise ValueError("lambda_value must be non-negative")
    if lambda_value == 0.0 or mode == "identity":
        return np.ones(len(safe_count), dtype=float)
    reliability = safe_count / (safe_count + float(lambda_value))
    if mode == "all":
        return reliability
    if mode == "new_only":
        return np.where(newcomer, reliability, 1.0)
    if mode == "new_or_low":
        eligible = newcomer | (safe_count < float(low_sample_threshold))
        return np.where(eligible, reliability, 1.0)
    raise ValueError(f"unknown gate mode: {mode}")


def fallback_feature_columns(feature_columns: list[str]) -> list[str]:
    return [
        column
        for column in feature_columns
        if not (column.startswith("asof_pitcher_") and column != "asof_pitcher_n")
    ]


def cache_path(cache_dir: Path, validation_season: int, features: list[str]) -> Path:
    signature = hashlib.sha256(
        ("fallback-v1|" + "|".join(features) + f"|{validation_season}|160").encode()
    ).hexdigest()[:12]
    return cache_dir / f"fallback_{validation_season}_{signature}.npz"


def fit_fallback_oof(
    train: pd.DataFrame,
    features: list[str],
    validation_season: int,
    cache_dir: Path,
) -> tuple[np.ndarray, float, str]:
    train_mask = train["season"] < validation_season
    validation_mask = train["season"] == validation_season
    validation_indices = np.flatnonzero(validation_mask.to_numpy())
    destination = cache_path(cache_dir, validation_season, features)
    if destination.is_file():
        cached = np.load(destination)
        if np.array_equal(cached["validation_indices"], validation_indices):
            return cached["prediction"].astype(float), 0.0, "cache"

    model = OptimizedBaseballEnsemble(
        hist_weight=0.45,
        n_estimators=160,
        random_state=42,
        smoothing_lambdas=(),
    )
    started = time.perf_counter()
    model.fit(
        train.loc[train_mask, features],
        train.loc[train_mask, TARGET_COL].to_numpy(),
    )
    fit_seconds = time.perf_counter() - started
    prediction = model.predict_proba(train.loc[validation_mask, features])[:, 1]
    np.savez_compressed(
        destination,
        validation_indices=validation_indices,
        prediction=np.asarray(prediction, dtype=np.float32),
    )
    del model
    gc.collect()
    return np.asarray(prediction, dtype=float), fit_seconds, "fitted"


def prediction_metrics(frame: pd.DataFrame, prediction) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=float)
    truth = frame[TARGET_COL].to_numpy(float)
    newcomer = frame["is_new_pitcher"].to_numpy(bool)
    existing = ~newcomer
    return {
        "brier": brier_score(truth, prediction),
        "competition_score": competition_score(truth, prediction),
        "prediction_mean": float(prediction.mean()),
        "new_pitcher_brier": brier_score(truth[newcomer], prediction[newcomer]),
        "existing_pitcher_brier": brier_score(truth[existing], prediction[existing]),
    }


def candidate_prediction(
    frame: pd.DataFrame,
    mode: str,
    lambda_value: float,
    threshold: float,
    final_logit_shift: float,
) -> np.ndarray:
    weight = gate_weight(
        frame["asof_pitcher_n"],
        frame["is_new_pitcher"],
        mode,
        lambda_value,
        threshold,
    )
    mixed = (
        weight * frame[CURRENT_COL].to_numpy(float)
        + (1.0 - weight) * frame[FALLBACK_REFINED_COL].to_numpy(float)
    )
    return apply_logit_shift(mixed, final_logit_shift)


def selection_objective(frame: pd.DataFrame, prediction) -> float:
    rows = []
    for season, fold in frame.groupby("validation_season", sort=True):
        probability = np.asarray(prediction)[frame.index.get_indexer(fold.index)]
        truth = fold[TARGET_COL].to_numpy(float)
        reference = truth.mean() * (1.0 - truth.mean())
        rows.append((int(season), brier_score(truth, probability) / reference))
    values = np.asarray([value for _, value in rows], dtype=float)
    weights = np.asarray([0.4, 0.6][-len(values):], dtype=float)
    weights /= weights.sum()
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return mean + 0.10 * np.sqrt(variance)


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
        default=Path("artifacts/new_pitcher_fallback"),
    )
    parser.add_argument("--validation-seasons", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument("--development-seasons", type=int, nargs="+", default=[2022, 2023])
    parser.add_argument("--outer-season", type=int, default=2024)
    parser.add_argument("--final-logit-shift", type=float, default=-0.0461)
    parser.add_argument("--season-decay", type=float, default=0.6)
    parser.add_argument("--game-strength", type=float, default=0.10)
    parser.add_argument("--game-shrinkage", type=float, default=100_000.0)
    parser.add_argument("--calibration-strength", type=float, default=0.25)
    args = parser.parse_args()

    summary = json.loads(args.temporal_summary.read_text(encoding="utf-8"))
    main_features = list(summary["feature_columns"])
    fallback_features = fallback_feature_columns(main_features)
    read_columns = list(dict.fromkeys([ID_COL, TARGET_COL, "pitcher_id", *main_features]))
    train = pd.read_csv(args.data_dir / "train.csv", usecols=read_columns)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.artifact_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    fallback_parts = []
    fit_rows = []
    for validation_season in args.validation_seasons:
        prediction, fit_seconds, source = fit_fallback_oof(
            train, fallback_features, validation_season, cache_dir
        )
        mask = train["season"] == validation_season
        truth = train.loc[mask, TARGET_COL].to_numpy(float)
        fold = pd.DataFrame(
            {
                ID_COL: train.loc[mask, ID_COL].to_numpy(),
                TARGET_COL: truth,
                "validation_season": validation_season,
                FALLBACK_RAW_COL: prediction,
                "game_type": train.loc[mask, "game_type"].to_numpy(),
                "branch": "fallback",
            }
        )
        fallback_parts.append(fold)
        fit_rows.append(
            {
                "validation_season": validation_season,
                "brier": brier_score(truth, prediction),
                "competition_score": competition_score(truth, prediction),
                "fit_seconds": fit_seconds,
                "source": source,
            }
        )
        print(
            f"season={validation_season} fallback_brier={fit_rows[-1]['brier']:.8f} "
            f"score={fit_rows[-1]['competition_score']:.5f} "
            f"fit={fit_seconds:.1f}s source={source}",
            flush=True,
        )

    fallback = pd.concat(fallback_parts, ignore_index=True)
    rolling_input = fallback.rename(columns={FALLBACK_RAW_COL: "development_blend"})
    refined_fallback = rolling_refinement(
        rolling_input,
        game_strength=args.game_strength,
        game_shrinkage=args.game_shrinkage,
        calibration_strength=args.calibration_strength,
        season_decay=args.season_decay,
    )
    fallback[FALLBACK_REFINED_COL] = refined_fallback["rolling_calibrated"].to_numpy(float)

    current = pd.read_csv(args.current_oof)
    current = current[current["branch"].eq("temporal_original")][
        [ID_COL, TARGET_COL, "validation_season", CURRENT_COL]
    ]
    frame = current.merge(
        fallback[[ID_COL, FALLBACK_RAW_COL, FALLBACK_REFINED_COL]],
        on=ID_COL,
        how="inner",
        validate="one_to_one",
    )
    cohort_source = train[[ID_COL, "season", "pitcher_id", "asof_pitcher_n", "game_type"]]
    frame = add_pitcher_cohorts(frame, cohort_source)
    if len(frame) != len(current):
        raise ValueError("fallback and current OOF row sets do not match")

    development = frame[frame["validation_season"].isin(args.development_seasons)].copy()
    development = development.reset_index(drop=True)
    baseline_development = apply_logit_shift(
        development[CURRENT_COL], args.final_logit_shift
    )
    baseline_objective = selection_objective(development, baseline_development)
    baseline_new_brier = prediction_metrics(development, baseline_development)[
        "new_pitcher_brier"
    ]

    candidates = [
        {
            "mode": "identity",
            "lambda": 0.0,
            "threshold": 0.0,
            "prediction": baseline_development,
        }
    ]
    for mode in ("all", "new_only"):
        for lambda_value in DEFAULT_LAMBDAS:
            candidates.append(
                {
                    "mode": mode,
                    "lambda": lambda_value,
                    "threshold": 0.0,
                    "prediction": candidate_prediction(
                        development,
                        mode,
                        lambda_value,
                        0.0,
                        args.final_logit_shift,
                    ),
                }
            )
    for threshold in DEFAULT_THRESHOLDS:
        for lambda_value in DEFAULT_LAMBDAS:
            candidates.append(
                {
                    "mode": "new_or_low",
                    "lambda": lambda_value,
                    "threshold": threshold,
                    "prediction": candidate_prediction(
                        development,
                        "new_or_low",
                        lambda_value,
                        threshold,
                        args.final_logit_shift,
                    ),
                }
            )

    selection_rows = []
    for candidate in candidates:
        metrics = prediction_metrics(development, candidate["prediction"])
        objective = selection_objective(development, candidate["prediction"])
        selection_rows.append(
            {
                "mode": candidate["mode"],
                "lambda": candidate["lambda"],
                "threshold": candidate["threshold"],
                "objective": objective,
                "objective_delta": objective - baseline_objective,
                **metrics,
                "new_pitcher_brier_delta": metrics["new_pitcher_brier"]
                - baseline_new_brier,
            }
        )
    selection_metrics = pd.DataFrame(selection_rows).sort_values(
        ["objective", "new_pitcher_brier", "mode", "lambda"], kind="stable"
    )
    selected = selection_metrics.iloc[0]

    outer = frame[frame["validation_season"] == args.outer_season].copy().reset_index(drop=True)
    outer_baseline = apply_logit_shift(outer[CURRENT_COL], args.final_logit_shift)
    outer_candidate = candidate_prediction(
        outer,
        str(selected["mode"]),
        float(selected["lambda"]),
        float(selected["threshold"]),
        args.final_logit_shift,
    )
    outer_rows = []
    for name, prediction in (
        ("official_852_structure_baseline", outer_baseline),
        ("new_pitcher_fallback_candidate", outer_candidate),
    ):
        outer_rows.append({"model": name, **prediction_metrics(outer, prediction)})
    outer_metrics = pd.DataFrame(outer_rows)
    truth = outer[TARGET_COL].to_numpy(float)
    comparison = paired_brier_comparison(truth, outer_baseline, outer_candidate)
    baseline_row = outer_metrics.iloc[0]
    candidate_row = outer_metrics.iloc[1]
    passes = bool(
        candidate_row["brier"] < baseline_row["brier"]
        and candidate_row["new_pitcher_brier"] <= baseline_row["new_pitcher_brier"]
        and comparison["paired_ci95_high"] < 0.0
    )

    run_summary = {
        "status": "promote" if passes else "rejected_keep_official_852_model",
        "official_leaderboard_baseline_score": 852.1984993386,
        "fixed_final_logit_shift": args.final_logit_shift,
        "fallback_feature_columns": fallback_features,
        "excluded_pitcher_history_columns": sorted(set(main_features) - set(fallback_features)),
        "selected_gate": {
            "mode": str(selected["mode"]),
            "lambda": float(selected["lambda"]),
            "threshold": float(selected["threshold"]),
        },
        "development_baseline_objective": baseline_objective,
        "development_candidate_objective": float(selected["objective"]),
        "outer_baseline_brier": float(baseline_row["brier"]),
        "outer_candidate_brier": float(candidate_row["brier"]),
        "outer_baseline_local_score": float(baseline_row["competition_score"]),
        "outer_candidate_local_score": float(candidate_row["competition_score"]),
        "outer_baseline_new_pitcher_brier": float(baseline_row["new_pitcher_brier"]),
        "outer_candidate_new_pitcher_brier": float(candidate_row["new_pitcher_brier"]),
        "paired_comparison_candidate_vs_baseline": comparison,
        "adoption_rule": (
            "improve shifted 2024 overall Brier with paired CI below zero and do not "
            "worsen shifted 2024 new-pitcher Brier"
        ),
    }

    pd.DataFrame(fit_rows).to_csv(args.artifact_dir / "fallback_fold_metrics.csv", index=False)
    selection_metrics.to_csv(args.artifact_dir / "gate_selection_metrics.csv", index=False)
    outer_metrics.to_csv(args.artifact_dir / "outer_metrics.csv", index=False)
    outer[[ID_COL, TARGET_COL, "is_new_pitcher", "asof_pitcher_n"]].assign(
        baseline_probability=outer_baseline,
        candidate_probability=outer_candidate,
    ).to_csv(args.artifact_dir / "outer_predictions.csv", index=False)
    (args.artifact_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(selection_metrics.head(10).to_string(index=False), flush=True)
    print(outer_metrics.to_string(index=False), flush=True)
    print(json.dumps(run_summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
