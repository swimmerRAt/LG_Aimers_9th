#!/usr/bin/env python3
"""Blend a leakage-safe monthly rate time series with the official 852 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from model.monthly_time_series import MonthlyRateTimeSeries
from model.oof_stacking import apply_logit_shift
from src.lg_aimers.metrics import brier_score, competition_score, paired_brier_comparison


ID_COL = "row_id"
TARGET_COL = "control_success"
CURRENT_COL = "rolling_calibrated"
VALIDATION_SEASONS = (2022, 2023, 2024)
DEVELOPMENT_SEASONS = (2022, 2023)


def logit(probability) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), 1e-7, 1.0 - 1e-7)
    return np.log(values / (1.0 - values))


def sigmoid(values) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(values, dtype=float), -35.0, 35.0)))


def add_time_series_offset(base_probability, offset, strength: float) -> np.ndarray:
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("time-series strength must be inside [0, 1]")
    return sigmoid(logit(base_probability) + float(strength) * np.asarray(offset, dtype=float))


def normalized_brier(truth, prediction) -> float:
    truth = np.asarray(truth, dtype=float)
    reference = float(truth.mean() * (1.0 - truth.mean()))
    if reference <= 0.0:
        raise ValueError("validation fold must contain both target classes")
    return brier_score(truth, prediction) / reference


def robust_objective(
    truth,
    prediction,
    seasons,
    *,
    season_weights=(0.4, 0.6),
    stability_penalty=0.25,
) -> float:
    losses = np.asarray(
        [
            normalized_brier(
                np.asarray(truth)[np.asarray(seasons) == season],
                np.asarray(prediction)[np.asarray(seasons) == season],
            )
            for season in DEVELOPMENT_SEASONS
        ]
    )
    weights = np.asarray(season_weights, dtype=float)
    weights /= weights.sum()
    mean = float(weights @ losses)
    variance = float(weights @ np.square(losses - mean))
    return mean + float(stability_penalty) * np.sqrt(variance)


def configuration_grid() -> list[dict]:
    return [
        {
            "harmonic_order": harmonic_order,
            "ridge": ridge,
            "recency_decay_per_year": decay,
        }
        for harmonic_order in (0, 1, 2)
        for ridge in (0.1, 1.0, 10.0)
        for decay in (1.0, 0.8)
    ]


def rolling_time_series_offset(
    train: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    forecast_rows = []
    for validation_season in VALIDATION_SEASONS:
        training = train[train["season"].lt(validation_season)]
        validation = train[train["season"].eq(validation_season)]
        model = MonthlyRateTimeSeries(**config).fit(
            training[["season", "game_month"]], training[TARGET_COL].to_numpy()
        )
        unique_times = validation[["season", "game_month"]].drop_duplicates().sort_values(
            ["season", "game_month"]
        )
        unique_times = unique_times.copy()
        unique_times["time_series_probability"] = model.predict_proba(unique_times)
        unique_times["time_series_logit_offset"] = model.predict_logit_offset(unique_times)
        unique_times["training_global_rate"] = model.global_rate_
        forecast_rows.append(unique_times)
        fold = validation[[ID_COL, "season", "game_month"]].merge(
            unique_times,
            on=["season", "game_month"],
            how="left",
            validate="many_to_one",
        )
        parts.append(fold)
    return pd.concat(parts, ignore_index=True), pd.concat(forecast_rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--current-oof",
        type=Path,
        default=Path("artifacts/probability_refinement/final_comparison/oof_predictions.csv"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/monthly_time_series"),
    )
    parser.add_argument("--final-logit-shift", type=float, default=-0.0461)
    parser.add_argument("--stability-penalty", type=float, default=0.25)
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(
        args.data_dir / "train.csv",
        usecols=[ID_COL, TARGET_COL, "season", "game_month"],
    )
    current = pd.read_csv(args.current_oof)
    current = current[current["branch"].eq("temporal_original")][
        [ID_COL, TARGET_COL, "validation_season", CURRENT_COL]
    ].copy()
    current["official_probability"] = apply_logit_shift(
        current[CURRENT_COL], args.final_logit_shift
    )
    validation = train[train["season"].isin(VALIDATION_SEASONS)][
        [ID_COL, "season", "game_month"]
    ].rename(columns={"season": "validation_season"})
    base = current.merge(
        validation,
        on=[ID_COL, "validation_season"],
        how="inner",
        validate="one_to_one",
    )
    if len(base) != len(current):
        raise ValueError("current OOF does not align with validation rows")

    development_mask = base["validation_season"].isin(DEVELOPMENT_SEASONS).to_numpy()
    development_truth = base.loc[development_mask, TARGET_COL].to_numpy(float)
    development_base = base.loc[development_mask, "official_probability"].to_numpy(float)
    development_season = base.loc[development_mask, "validation_season"].to_numpy(int)
    official_objective = robust_objective(
        development_truth,
        development_base,
        development_season,
        stability_penalty=args.stability_penalty,
    )

    search_rows = []
    candidate_payloads = {}
    forecast_parts = []
    for config_id, config in enumerate(configuration_grid(), start=1):
        offsets, forecasts = rolling_time_series_offset(train, config)
        offsets = offsets.rename(columns={"season": "validation_season"})
        aligned = base[[ID_COL, "validation_season"]].merge(
            offsets[[ID_COL, "validation_season", "time_series_logit_offset"]],
            on=[ID_COL, "validation_season"],
            how="left",
            validate="one_to_one",
        )
        all_offsets = aligned["time_series_logit_offset"].to_numpy(float)
        development_offsets = all_offsets[development_mask]

        def objective(strength: float) -> float:
            prediction = add_time_series_offset(
                development_base, development_offsets, strength
            )
            return robust_objective(
                development_truth,
                prediction,
                development_season,
                stability_penalty=args.stability_penalty,
            )

        result = minimize_scalar(
            objective,
            bounds=(0.001, 0.5),
            method="bounded",
            options={"xatol": 1e-7, "maxiter": 200},
        )
        strength = float(result.x)
        prediction = add_time_series_offset(
            development_base, development_offsets, strength
        )
        deltas = {}
        non_degraded = True
        for season in DEVELOPMENT_SEASONS:
            fold = development_season == season
            delta = brier_score(development_truth[fold], prediction[fold]) - brier_score(
                development_truth[fold], development_base[fold]
            )
            deltas[season] = delta
            non_degraded &= delta <= 0.0
        row = {
            "config_id": config_id,
            **config,
            "strength": strength,
            "objective": float(result.fun),
            "objective_improvement": official_objective - float(result.fun),
            "delta_2022": deltas[2022],
            "delta_2023": deltas[2023],
            "development_non_degraded": bool(non_degraded),
            "optimizer_success": bool(result.success),
        }
        search_rows.append(row)
        candidate_payloads[config_id] = (all_offsets, forecasts, config)
        forecasts = forecasts.copy()
        forecasts.insert(0, "config_id", config_id)
        forecast_parts.append(forecasts)
        print(
            f"config={config_id:02d} harmonic={config['harmonic_order']} "
            f"ridge={config['ridge']:g} decay={config['recency_decay_per_year']:.1f} "
            f"strength={strength:.6f} objective={result.fun:.9f} "
            f"d22={deltas[2022]:+.8f} d23={deltas[2023]:+.8f}",
            flush=True,
        )

    search = pd.DataFrame(search_rows)
    eligible = search[
        search["development_non_degraded"]
        & search["objective"].lt(official_objective)
    ]
    selection_passed = not eligible.empty
    selected_row = (
        eligible.sort_values(["objective", "strength"]).iloc[0]
        if selection_passed
        else search.sort_values(["objective", "strength"]).iloc[0]
    )
    selected_id = int(selected_row["config_id"])
    selected_offsets, selected_forecasts, selected_config = candidate_payloads[selected_id]
    selected_strength = float(selected_row["strength"])
    comparison = base.copy()
    comparison["time_series_logit_offset"] = selected_offsets
    comparison["candidate_probability"] = add_time_series_offset(
        comparison["official_probability"], selected_offsets, selected_strength
    )

    metric_rows = []
    paired = {}
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
                "target_mean": float(truth.mean()),
                "official_prediction_mean": float(official.mean()),
                "candidate_prediction_mean": float(candidate.mean()),
                "time_series_offset_mean": float(
                    fold["time_series_logit_offset"].mean()
                ),
            }
        )
        paired[str(season)] = paired_brier_comparison(truth, official, candidate)
    metrics = pd.DataFrame(metric_rows)
    outer_improved = bool(
        metrics.loc[
            metrics["validation_season"].eq(2024),
            "candidate_minus_official_brier",
        ].iloc[0]
        < 0.0
    )
    adopted = bool(selection_passed and outer_improved)
    status = (
        "diagnostic_improvement_requires_fresh_validation"
        if adopted
        else "rejected_keep_official_852_model"
    )

    search.to_csv(args.artifact_dir / "configuration_search.csv", index=False)
    metrics.to_csv(args.artifact_dir / "season_metrics.csv", index=False)
    selected_forecasts.to_csv(args.artifact_dir / "selected_monthly_forecasts.csv", index=False)
    pd.concat(forecast_parts, ignore_index=True).to_csv(
        args.artifact_dir / "all_monthly_forecasts.csv", index=False
    )
    comparison.to_csv(args.artifact_dir / "oof_predictions.csv", index=False)
    summary = {
        "status": status,
        "official_leaderboard_baseline_score": 852.1984993386,
        "model_type": "monthly harmonic ridge dynamic regression on aggregate target logits",
        "experiment_structure": (
            "fit monthly time series using only seasons earlier than each validation season; "
            "add a tuned fraction of its forecast logit deviation to the official model logit"
        ),
        "row_independence": (
            "forecast depends only on fitted training history and the row's season/game_month; "
            "no other validation or test row is used at inference"
        ),
        "selection_seasons": list(DEVELOPMENT_SEASONS),
        "outer_diagnostic_season": 2024,
        "outer_is_reused_not_one_shot": True,
        "fixed_final_logit_shift": args.final_logit_shift,
        "official_development_objective": official_objective,
        "selected_configuration": selected_config,
        "selected_strength": selected_strength,
        "selected_development_objective": float(selected_row["objective"]),
        "selection_passed": selection_passed,
        "season_metrics": metrics.to_dict(orient="records"),
        "paired_comparisons": paired,
        "outer_improved": outer_improved,
        "adopted": False,
        "adoption_note": (
            "The official model and ZIP remain unchanged. Even a diagnostic improvement would "
            "require fresh confirmation before fitting a deployable wrapper."
        ),
    }
    (args.artifact_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
