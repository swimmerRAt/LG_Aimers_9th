#!/usr/bin/env python3
"""Evaluate the official-algorithm PatchTST as a monthly probability branch."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar

from experiment_monthly_time_series import (
    add_time_series_offset,
    robust_objective,
)
from model.oof_stacking import apply_logit_shift
from model.patchtst_model import PatchTST, parameter_count
from src.lg_aimers.metrics import brier_score, competition_score, paired_brier_comparison


ID_COL = "row_id"
TARGET_COL = "control_success"
CURRENT_COL = "rolling_calibrated"
VALIDATION_SEASONS = (2022, 2023, 2024)
DEVELOPMENT_SEASONS = (2022, 2023)
PLAY_MONTHS = tuple(range(3, 11))
CHANNEL_NAMES = ("overall", "game_type_F", "game_type_R")


def _logit(probability) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def build_monthly_panel(
    frame: pd.DataFrame,
    *,
    minimum_season: int,
    maximum_season: int,
    prior_strength: float = 500.0,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, float]]:
    """Build a regular March–October panel with overall/F/R rate channels."""
    training = frame[frame["season"].between(minimum_season, maximum_season)].copy()
    if training.empty:
        raise ValueError("monthly panel has no training rows")
    grid = pd.MultiIndex.from_product(
        [range(minimum_season, maximum_season + 1), PLAY_MONTHS],
        names=["season", "game_month"],
    ).to_frame(index=False)
    priors = {"overall": float(training[TARGET_COL].mean())}
    for game_type in ("F", "R"):
        subset = training[training["game_type"].eq(game_type)]
        if subset.empty:
            raise ValueError(f"training data has no game_type={game_type}")
        priors[f"game_type_{game_type}"] = float(subset[TARGET_COL].mean())

    series = []
    for channel in CHANNEL_NAMES:
        subset = training
        if channel != "overall":
            subset = training[training["game_type"].eq(channel[-1])]
        aggregate = subset.groupby(["season", "game_month"])[TARGET_COL].agg(
            successes="sum", rows="size"
        ).reset_index()
        aggregate = grid.merge(
            aggregate, on=["season", "game_month"], how="left", validate="one_to_one"
        )
        prior = priors[channel]
        successes = aggregate["successes"].fillna(0.0).to_numpy(float)
        rows = aggregate["rows"].fillna(0.0).to_numpy(float)
        rate = (successes + float(prior_strength) * prior) / (
            rows + float(prior_strength)
        )
        series.append(rate)
    panel = np.column_stack(series)
    return panel, grid, priors


def sliding_windows(
    panel: np.ndarray, context_length: int, prediction_length: int
) -> tuple[np.ndarray, np.ndarray]:
    sample_count = len(panel) - context_length - prediction_length + 1
    if sample_count <= 0:
        raise ValueError("monthly panel is too short for PatchTST windows")
    inputs = []
    targets = []
    for start in range(sample_count):
        inputs.append(panel[start : start + context_length].T)
        targets.append(
            panel[
                start + context_length : start + context_length + prediction_length
            ].T
        )
    return np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def _fit_epochs(
    model: PatchTST,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean(torch.square(model(inputs) - targets))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def fit_patchtst_forecast(
    panel: np.ndarray,
    *,
    seed: int,
    context_length: int = 12,
    prediction_length: int = 8,
    max_epochs: int = 300,
    patience: int = 35,
    learning_rate: float = 1e-3,
) -> tuple[np.ndarray, dict]:
    """Select epoch count on the latest past window, refit all windows, forecast."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    inputs, targets = sliding_windows(panel, context_length, prediction_length)
    validation_count = 1 if len(inputs) < 10 else 2
    training_inputs = torch.from_numpy(inputs[:-validation_count])
    training_targets = torch.from_numpy(targets[:-validation_count])
    validation_inputs = torch.from_numpy(inputs[-validation_count:])
    validation_targets = torch.from_numpy(targets[-validation_count:])
    config = dict(
        channels=panel.shape[1],
        context_length=context_length,
        prediction_length=prediction_length,
        patch_length=4,
        stride=2,
        n_layers=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        dropout=0.1,
        attention_dropout=0.0,
        head_dropout=0.0,
        padding_end=True,
        revin=True,
    )
    selection_model = PatchTST(**config)
    optimizer = torch.optim.AdamW(
        selection_model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    best_state = None
    best_epoch = 0
    best_validation = float("inf")
    stale = 0
    for epoch in range(max_epochs):
        selection_model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean(
            torch.square(selection_model(training_inputs) - training_targets)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selection_model.parameters(), 1.0)
        optimizer.step()
        selection_model.eval()
        with torch.no_grad():
            validation_loss = float(
                torch.mean(
                    torch.square(selection_model(validation_inputs) - validation_targets)
                ).item()
            )
        if validation_loss < best_validation - 1e-9:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(selection_model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("PatchTST epoch selection did not produce a model")

    torch.manual_seed(seed)
    final_model = PatchTST(**config)
    all_inputs = torch.from_numpy(inputs)
    all_targets = torch.from_numpy(targets)
    _fit_epochs(
        final_model,
        all_inputs,
        all_targets,
        epochs=best_epoch + 1,
        learning_rate=learning_rate,
    )
    final_context = torch.from_numpy(
        panel[-context_length:].T[None].astype(np.float32)
    )
    final_model.eval()
    with torch.no_grad():
        forecast = final_model(final_context).numpy()[0].T
    forecast = np.clip(forecast, 0.01, 0.99)
    return forecast, {
        "seed": seed,
        "window_count": len(inputs),
        "selection_training_windows": len(training_inputs),
        "selection_validation_windows": validation_count,
        "best_epoch": best_epoch + 1,
        "best_validation_mse": best_validation,
        "parameter_count": parameter_count(final_model),
    }


def rolling_patchtst_forecasts(
    train: pd.DataFrame,
    *,
    seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecast_rows = []
    fit_rows = []
    minimum_season = int(train["season"].min())
    for validation_season in VALIDATION_SEASONS:
        panel, _, priors = build_monthly_panel(
            train,
            minimum_season=minimum_season,
            maximum_season=validation_season - 1,
        )
        seed_forecasts = []
        started = time.perf_counter()
        for seed in seeds:
            forecast, fit = fit_patchtst_forecast(panel, seed=seed)
            seed_forecasts.append(forecast)
            fit["validation_season"] = validation_season
            fit_rows.append(fit)
        ensemble = np.mean(seed_forecasts, axis=0)
        elapsed = time.perf_counter() - started
        for month_position, month in enumerate(PLAY_MONTHS):
            row = {
                "validation_season": validation_season,
                "game_month": month,
                "fit_seconds": elapsed,
            }
            for channel_position, channel in enumerate(CHANNEL_NAMES):
                probability = float(ensemble[month_position, channel_position])
                row[f"forecast_{channel}"] = probability
                row[f"offset_{channel}"] = float(
                    _logit([probability])[0] - _logit([priors[channel]])[0]
                )
                row[f"prior_{channel}"] = priors[channel]
            forecast_rows.append(row)
        print(
            f"season={validation_season} patchtst_seeds={len(seeds)} "
            f"windows={fit_rows[-1]['window_count']} seconds={elapsed:.1f}",
            flush=True,
        )
    return pd.DataFrame(forecast_rows), pd.DataFrame(fit_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--current-oof",
        type=Path,
        default=Path("artifacts/probability_refinement/final_comparison/oof_predictions.csv"),
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/patchtst")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--final-logit-shift", type=float, default=-0.0461)
    parser.add_argument("--stability-penalty", type=float, default=0.25)
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    train = pd.read_csv(
        args.data_dir / "train.csv",
        usecols=[ID_COL, TARGET_COL, "season", "game_month", "game_type"],
    )
    current = pd.read_csv(args.current_oof)
    current = current[current["branch"].eq("temporal_original")][
        [ID_COL, TARGET_COL, "validation_season", CURRENT_COL]
    ].copy()
    current["official_probability"] = apply_logit_shift(
        current[CURRENT_COL], args.final_logit_shift
    )
    row_context = train[train["season"].isin(VALIDATION_SEASONS)][
        [ID_COL, "season", "game_month", "game_type"]
    ].rename(columns={"season": "validation_season"})
    base = current.merge(
        row_context,
        on=[ID_COL, "validation_season"],
        how="inner",
        validate="one_to_one",
    )
    if len(base) != len(current):
        raise ValueError("current OOF does not align with validation rows")

    forecasts, fit_summary = rolling_patchtst_forecasts(
        train, seeds=tuple(args.seeds)
    )
    comparison = base.merge(
        forecasts,
        on=["validation_season", "game_month"],
        how="left",
        validate="many_to_one",
    )
    comparison["group_offset"] = np.where(
        comparison["game_type"].eq("F"),
        comparison["offset_game_type_F"],
        comparison["offset_game_type_R"],
    )

    development = comparison[
        comparison["validation_season"].isin(DEVELOPMENT_SEASONS)
    ]
    truth = development[TARGET_COL].to_numpy(float)
    official = development["official_probability"].to_numpy(float)
    seasons = development["validation_season"].to_numpy(int)
    official_objective = robust_objective(
        truth,
        official,
        seasons,
        stability_penalty=args.stability_penalty,
    )
    search_rows = []
    for group_mix in (0.0, 0.5, 1.0):
        offset = (
            (1.0 - group_mix) * development["offset_overall"].to_numpy(float)
            + group_mix * development["group_offset"].to_numpy(float)
        )

        def objective(strength: float) -> float:
            prediction = add_time_series_offset(official, offset, strength)
            return robust_objective(
                truth,
                prediction,
                seasons,
                stability_penalty=args.stability_penalty,
            )

        result = minimize_scalar(
            objective,
            bounds=(0.001, 0.5),
            method="bounded",
            options={"xatol": 1e-7, "maxiter": 200},
        )
        strength = float(result.x)
        prediction = add_time_series_offset(official, offset, strength)
        deltas = {}
        non_degraded = True
        for season in DEVELOPMENT_SEASONS:
            mask = seasons == season
            delta = brier_score(truth[mask], prediction[mask]) - brier_score(
                truth[mask], official[mask]
            )
            deltas[season] = delta
            non_degraded &= delta <= 0.0
        search_rows.append(
            {
                "group_mix": group_mix,
                "strength": strength,
                "objective": float(result.fun),
                "objective_improvement": official_objective - float(result.fun),
                "delta_2022": deltas[2022],
                "delta_2023": deltas[2023],
                "development_non_degraded": bool(non_degraded),
            }
        )
    search = pd.DataFrame(search_rows)
    eligible = search[
        search["development_non_degraded"]
        & search["objective"].lt(official_objective)
    ]
    selection_passed = not eligible.empty
    selected = (
        eligible.sort_values("objective").iloc[0]
        if selection_passed
        else search.sort_values("objective").iloc[0]
    )
    group_mix = float(selected["group_mix"])
    strength = float(selected["strength"])
    comparison["selected_offset"] = (
        (1.0 - group_mix) * comparison["offset_overall"]
        + group_mix * comparison["group_offset"]
    )
    comparison["candidate_probability"] = add_time_series_offset(
        comparison["official_probability"], comparison["selected_offset"], strength
    )

    metric_rows = []
    paired = {}
    for season in VALIDATION_SEASONS:
        fold = comparison[comparison["validation_season"].eq(season)]
        fold_truth = fold[TARGET_COL].to_numpy(float)
        fold_official = fold["official_probability"].to_numpy(float)
        fold_candidate = fold["candidate_probability"].to_numpy(float)
        official_brier = brier_score(fold_truth, fold_official)
        candidate_brier = brier_score(fold_truth, fold_candidate)
        metric_rows.append(
            {
                "validation_season": season,
                "official_brier": official_brier,
                "candidate_brier": candidate_brier,
                "candidate_minus_official_brier": candidate_brier - official_brier,
                "official_score": competition_score(fold_truth, fold_official),
                "candidate_score": competition_score(fold_truth, fold_candidate),
                "target_mean": float(fold_truth.mean()),
                "official_prediction_mean": float(fold_official.mean()),
                "candidate_prediction_mean": float(fold_candidate.mean()),
                "selected_offset_mean": float(fold["selected_offset"].mean()),
            }
        )
        paired[str(season)] = paired_brier_comparison(
            fold_truth, fold_official, fold_candidate
        )
    metrics = pd.DataFrame(metric_rows)
    outer_improved = bool(
        metrics.loc[
            metrics["validation_season"].eq(2024),
            "candidate_minus_official_brier",
        ].iloc[0]
        < 0.0
    )
    diagnostic_passed = bool(selection_passed and outer_improved)
    status = (
        "diagnostic_improvement_requires_fresh_validation"
        if diagnostic_passed
        else "rejected_keep_official_852_model"
    )

    forecasts.to_csv(args.artifact_dir / "monthly_forecasts.csv", index=False)
    fit_summary.to_csv(args.artifact_dir / "fit_summary.csv", index=False)
    search.to_csv(args.artifact_dir / "blend_search.csv", index=False)
    metrics.to_csv(args.artifact_dir / "season_metrics.csv", index=False)
    comparison.to_csv(args.artifact_dir / "oof_predictions.csv", index=False)
    summary = {
        "status": status,
        "official_leaderboard_baseline_score": 852.1984993386,
        "upstream_repository": "https://github.com/yuqinie98/PatchTST",
        "upstream_commit": "204c21efe0b39603ad6e2ca640ef5896646ab1a9",
        "upstream_license": "Apache-2.0",
        "algorithm": (
            "RevIN + overlapping patches + shared channel-independent projection and "
            "residual-attention Transformer encoder + flatten forecast head"
        ),
        "adaptation": (
            "March-October monthly rates for overall/game_type F/game_type R channels; "
            "small model dimensions because only 24-40 monthly slots precede validation"
        ),
        "model_parameters": int(fit_summary["parameter_count"].iloc[0]),
        "seeds": list(args.seeds),
        "selection_seasons": list(DEVELOPMENT_SEASONS),
        "outer_diagnostic_season": 2024,
        "outer_is_reused_not_one_shot": True,
        "official_development_objective": official_objective,
        "selected_group_mix": group_mix,
        "selected_strength": strength,
        "selected_development_objective": float(selected["objective"]),
        "selection_passed": selection_passed,
        "season_metrics": metrics.to_dict(orient="records"),
        "paired_comparisons": paired,
        "outer_improved": outer_improved,
        "adopted": False,
        "adoption_note": (
            "The official model and ZIP remain unchanged. PatchTST is retained only as an "
            "experimental branch unless development and outer diagnostics both improve."
        ),
        "limitations": [
            "only 24-40 regularized monthly slots precede each validation season",
            "months without games are filled by the past channel prior because upstream PatchTST has no missing-value mask",
            "the data has no exact game date or game identifier",
        ],
    }
    (args.artifact_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
