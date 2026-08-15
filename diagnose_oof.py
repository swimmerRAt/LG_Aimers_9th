#!/usr/bin/env python3
"""Decompose forward-validation Brier loss by calibration and baseball segments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.lg_aimers.metrics import brier_score, competition_score


ROOT = Path(__file__).resolve().parent
OOF_PATH = ROOT / "artifacts" / "forward_cv_2022_2024" / "oof_predictions.csv"
TRAIN_PATH = ROOT / "data" / "train.csv"
OUT_DIR = ROOT / "artifacts" / "score_diagnostics"


def add_segments(frame: pd.DataFrame, full_train: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["count_state"] = (
        frame["balls_before"].astype(str) + "-" + frame["strikes_before"].astype(str)
    )
    frame["pitcher_n_bucket"] = pd.cut(
        frame["asof_pitcher_n"], [-1, 0, 10, 50, 200, 1000, np.inf],
        labels=["0", "1-10", "11-50", "51-200", "201-1000", "1001+"],
    ).astype(str)
    frame["batter_n_bucket"] = pd.cut(
        frame["asof_batter_n"], [-1, 0, 10, 50, 200, 1000, np.inf],
        labels=["0", "1-10", "11-50", "51-200", "201-1000", "1001+"],
    ).astype(str)
    frame["li_bucket"] = pd.cut(
        frame["li"], [-np.inf, 0.5, 1.0, 2.0, np.inf],
        labels=["<=0.5", "0.5-1", "1-2", "2+"],
    ).astype(str)
    frame["score_bucket"] = pd.cut(
        frame["score_diff_pitcher_team"].abs(), [-1, 0, 1, 3, np.inf],
        labels=["tie", "one_run", "two_three", "four_plus"],
    ).astype(str)
    history_cols = [
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ]
    frame["pitcher_history_missing"] = frame[history_cols].isna().any(axis=1).map(
        {True: "missing", False: "available"}
    )

    frame["pitcher_seen_before"] = "unknown"
    frame["batter_seen_before"] = "unknown"
    for season in sorted(frame["validation_season"].unique()):
        past = full_train.loc[full_train["season"] < season]
        mask = frame["validation_season"] == season
        frame.loc[mask, "pitcher_seen_before"] = np.where(
            frame.loc[mask, "pitcher_id"].isin(set(past["pitcher_id"])), "seen", "new"
        )
        frame.loc[mask, "batter_seen_before"] = np.where(
            frame.loc[mask, "batter_id"].isin(set(past["batter_id"])), "seen", "new"
        )
    return frame


def main() -> None:
    usecols = [
        "row_id", "season", "game_month", "balls_before", "strikes_before",
        "score_diff_pitcher_team", "li", "pitcher_id", "batter_id",
        "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
        "asof_pitcher_n", "asof_batter_n",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ]
    full_train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", usecols=usecols)
    oof = pd.read_csv(OOF_PATH)
    oof = oof[oof["model"].isin(["constant", "random_forest", "histgb"])]
    merged = oof.merge(full_train, on="row_id", how="left", validate="many_to_one")
    if merged["season"].isna().any():
        raise ValueError("OOF rows failed to join train features")
    merged = add_segments(merged, full_train)

    summary_rows = []
    calibration_parts = []
    for (model, season), group in merged.groupby(["model", "validation_season"]):
        truth = group["control_success"].to_numpy()
        pred = group["prediction"].to_numpy()
        summary_rows.append({
            "model": model,
            "validation_season": season,
            "rows": len(group),
            "actual_rate": truth.mean(),
            "predicted_rate": pred.mean(),
            "mean_bias": pred.mean() - truth.mean(),
            "brier": brier_score(truth, pred),
            "competition_score": competition_score(truth, pred),
        })
        group = group.copy()
        group["probability_bin"] = pd.qcut(
            group["prediction"].rank(method="first"), 10, labels=False
        )
        calibration = group.groupby("probability_bin", observed=True).agg(
            rows=("control_success", "size"),
            predicted_rate=("prediction", "mean"),
            actual_rate=("control_success", "mean"),
        ).reset_index()
        calibration["model"] = model
        calibration["validation_season"] = season
        calibration["calibration_gap"] = calibration["predicted_rate"] - calibration["actual_rate"]
        calibration_parts.append(calibration)

    dimensions = [
        "game_month", "count_state", "pitcher_n_bucket", "batter_n_bucket",
        "li_bucket", "score_bucket", "pitcher_history_missing",
        "pitcher_seen_before", "batter_seen_before", "pitcher_hand", "batter_hand",
        "pitcher_team_id", "batter_team_id",
    ]
    segment_rows = []
    for (model, season), model_frame in merged.groupby(["model", "validation_season"]):
        fold_rate = model_frame["control_success"].mean()
        fold_reference = fold_rate * (1.0 - fold_rate)
        for dimension in dimensions:
            for value, group in model_frame.groupby(dimension, observed=True, dropna=False):
                truth = group["control_success"].to_numpy()
                pred = group["prediction"].to_numpy()
                segment_brier = brier_score(truth, pred)
                segment_rows.append({
                    "model": model,
                    "validation_season": season,
                    "dimension": dimension,
                    "segment": str(value),
                    "rows": len(group),
                    "weight": len(group) / len(model_frame),
                    "actual_rate": truth.mean(),
                    "predicted_rate": pred.mean(),
                    "mean_bias": pred.mean() - truth.mean(),
                    "brier": segment_brier,
                    "weighted_excess_vs_fold_reference": (
                        len(group) / len(model_frame) * (segment_brier - fold_reference)
                    ),
                })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    calibration = pd.concat(calibration_parts, ignore_index=True)
    segments = pd.DataFrame(segment_rows)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    calibration.to_csv(OUT_DIR / "calibration.csv", index=False)
    segments.to_csv(OUT_DIR / "segments.csv", index=False)

    recent = segments[
        (segments["validation_season"] == 2024)
        & (segments["model"].isin(["random_forest", "histgb"]))
    ]
    print("\n=== Model/season summary ===")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n=== Largest 2024 weighted excess-loss segments ===")
    print(
        recent.sort_values("weighted_excess_vs_fold_reference", ascending=False)
        .head(25)[[
            "model", "dimension", "segment", "rows", "actual_rate", "predicted_rate",
            "mean_bias", "brier", "weighted_excess_vs_fold_reference",
        ]]
        .to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print(f"\nsaved diagnostics to {OUT_DIR}")


if __name__ == "__main__":
    main()
