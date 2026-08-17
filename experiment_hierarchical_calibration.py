#!/usr/bin/env python3
"""Leakage-safe hierarchical probability calibration experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.lg_aimers.metrics import brier_score, competition_score, paired_brier_comparison


ID_COL = "row_id"
TARGET_COL = "control_success"
RAW_COL = "development_blend"
SPECS = (
    "global_logit",
    "beta",
    "hierarchical_logit",
    "hierarchical_beta",
    "hierarchical_interaction",
)
DEFAULT_C_VALUES = (
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    1.0,
)


def add_pitcher_cohorts(oof: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    columns = [ID_COL, "season", "pitcher_id", "asof_pitcher_n", "game_type"]
    result = oof.merge(train.loc[:, columns], on=ID_COL, how="left", validate="one_to_one")
    if result[columns[1:]].isna().all(axis=1).any():
        raise ValueError("OOF contains row_id values missing from train.csv")
    result["is_new_pitcher"] = 0
    for season in sorted(int(value) for value in result["validation_season"].unique()):
        prior_pitchers = set(
            train.loc[train["season"] < season, "pitcher_id"].dropna().unique()
        )
        mask = result["validation_season"] == season
        result.loc[mask, "is_new_pitcher"] = (
            ~result.loc[mask, "pitcher_id"].isin(prior_pitchers)
        ).astype(int)
    count = pd.to_numeric(result["asof_pitcher_n"], errors="coerce").fillna(0.0).clip(lower=0.0)
    result["is_low_sample"] = (count < 100.0).astype(int)
    result["log_pitcher_n"] = np.log1p(count)
    return result


def calibration_features(frame: pd.DataFrame, spec: str) -> np.ndarray:
    probability = np.clip(frame[RAW_COL].to_numpy(float), 1e-6, 1.0 - 1e-6)
    logit = np.log(probability / (1.0 - probability))
    log_probability = np.log(probability)
    log_failure = np.log1p(-probability)
    game_f = frame["game_type"].eq("F").to_numpy(float)
    newcomer = frame["is_new_pitcher"].to_numpy(float)
    low_sample = frame["is_low_sample"].to_numpy(float)
    log_count = frame["log_pitcher_n"].to_numpy(float)

    if spec == "global_logit":
        return np.column_stack([logit])
    if spec == "beta":
        return np.column_stack([log_probability, log_failure])
    if spec == "hierarchical_logit":
        return np.column_stack([logit, game_f, newcomer, low_sample, log_count])
    if spec == "hierarchical_beta":
        return np.column_stack(
            [log_probability, log_failure, game_f, newcomer, low_sample, log_count]
        )
    if spec == "hierarchical_interaction":
        return np.column_stack(
            [
                logit,
                game_f,
                newcomer,
                low_sample,
                log_count,
                logit * game_f,
                logit * newcomer,
                game_f * newcomer,
            ]
        )
    raise ValueError(f"unknown calibration spec: {spec}")


def make_calibrator(c_value: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(c_value), max_iter=300, solver="lbfgs"),
    )


def cohort_brier(frame: pd.DataFrame, prediction: np.ndarray, mask) -> float:
    mask = np.asarray(mask, dtype=bool)
    return brier_score(
        frame.loc[mask, TARGET_COL].to_numpy(float), np.asarray(prediction)[mask]
    )


def evaluate_prediction(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    truth = frame[TARGET_COL].to_numpy(float)
    newcomer = frame["is_new_pitcher"].to_numpy(bool)
    game_f = frame["game_type"].eq("F").to_numpy(bool)
    return {
        "brier": brier_score(truth, prediction),
        "competition_score": competition_score(truth, prediction),
        "prediction_mean": float(np.mean(prediction)),
        "new_pitcher_brier": cohort_brier(frame, prediction, newcomer),
        "existing_pitcher_brier": cohort_brier(frame, prediction, ~newcomer),
        "game_type_f_brier": cohort_brier(frame, prediction, game_f),
        "game_type_r_brier": cohort_brier(frame, prediction, ~game_f),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--temporal-oof",
        type=Path,
        default=Path("artifacts/temporal_ensemble/oof_predictions.csv"),
    )
    parser.add_argument(
        "--refined-oof",
        type=Path,
        default=Path("artifacts/probability_refinement/final_comparison/oof_predictions.csv"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/hierarchical_calibration"),
    )
    parser.add_argument("--selection-season", type=int, default=2023)
    parser.add_argument("--outer-season", type=int, default=2024)
    parser.add_argument("--season-decay", type=float, default=0.6)
    args = parser.parse_args()

    train = pd.read_csv(
        args.data_dir / "train.csv",
        usecols=[ID_COL, "season", "pitcher_id", "asof_pitcher_n", "game_type"],
    )
    temporal = pd.read_csv(
        args.temporal_oof,
        usecols=[ID_COL, TARGET_COL, "validation_season", RAW_COL],
    )
    frame = add_pitcher_cohorts(temporal, train)
    selection_train = frame[frame["validation_season"] < args.selection_season].copy()
    selection_validation = frame[frame["validation_season"] == args.selection_season].copy()
    if selection_train.empty or selection_validation.empty:
        raise ValueError("selection split is empty")

    selection_rows = []
    selection_predictions: dict[tuple[str, float], np.ndarray] = {}
    for spec in SPECS:
        for c_value in DEFAULT_C_VALUES:
            model = make_calibrator(c_value)
            model.fit(
                calibration_features(selection_train, spec),
                selection_train[TARGET_COL].to_numpy(),
            )
            prediction = model.predict_proba(
                calibration_features(selection_validation, spec)
            )[:, 1]
            metrics = evaluate_prediction(selection_validation, prediction)
            selection_rows.append({"spec": spec, "C": c_value, **metrics})
            selection_predictions[(spec, c_value)] = prediction

    selection_metrics = pd.DataFrame(selection_rows).sort_values(
        ["brier", "new_pitcher_brier", "spec", "C"], kind="stable"
    )
    best = selection_metrics.iloc[0]
    selected_spec = str(best["spec"])
    selected_c = float(best["C"])

    outer_train = frame[frame["validation_season"] < args.outer_season].copy()
    outer_validation = frame[frame["validation_season"] == args.outer_season].copy()
    latest_training_season = int(outer_train["validation_season"].max())
    sample_weight = np.power(
        float(args.season_decay),
        latest_training_season - outer_train["validation_season"].to_numpy(float),
    )
    selected_model = make_calibrator(selected_c)
    selected_model.fit(
        calibration_features(outer_train, selected_spec),
        outer_train[TARGET_COL].to_numpy(),
        logisticregression__sample_weight=sample_weight,
    )
    outer_prediction = selected_model.predict_proba(
        calibration_features(outer_validation, selected_spec)
    )[:, 1]

    refined = pd.read_csv(
        args.refined_oof,
        usecols=[ID_COL, "branch", "validation_season", "rolling_calibrated"],
    )
    refined = refined[
        refined["branch"].eq("temporal_original")
        & refined["validation_season"].eq(args.outer_season)
    ][[ID_COL, "rolling_calibrated"]]
    outer_validation = outer_validation.merge(
        refined, on=ID_COL, how="left", validate="one_to_one"
    )
    if outer_validation["rolling_calibrated"].isna().any():
        raise ValueError("current refined OOF is missing outer rows")

    predictions = {
        "raw_temporal": outer_validation[RAW_COL].to_numpy(float),
        "current_refined": outer_validation["rolling_calibrated"].to_numpy(float),
        "selected_hierarchical": outer_prediction,
    }
    outer_rows = []
    for name, prediction in predictions.items():
        outer_rows.append(
            {"model": name, **evaluate_prediction(outer_validation, prediction)}
        )
    outer_metrics = pd.DataFrame(outer_rows)
    truth = outer_validation[TARGET_COL].to_numpy(float)
    comparison = paired_brier_comparison(
        truth,
        predictions["current_refined"],
        predictions["selected_hierarchical"],
    )
    current_row = outer_metrics[outer_metrics["model"] == "current_refined"].iloc[0]
    candidate_row = outer_metrics[
        outer_metrics["model"] == "selected_hierarchical"
    ].iloc[0]
    passes = bool(
        candidate_row["brier"] <= current_row["brier"]
        and candidate_row["new_pitcher_brier"] <= current_row["new_pitcher_brier"]
        and comparison["paired_ci95_high"] < 0.0
    )

    summary = {
        "status": "promote" if passes else "rejected_keep_current_refined",
        "selection_train_seasons": sorted(
            int(value) for value in selection_train["validation_season"].unique()
        ),
        "selection_validation_season": args.selection_season,
        "outer_train_seasons": sorted(
            int(value) for value in outer_train["validation_season"].unique()
        ),
        "outer_validation_season": args.outer_season,
        "selected_spec": selected_spec,
        "selected_C": selected_c,
        "season_decay": args.season_decay,
        "selection_brier": float(best["brier"]),
        "outer_current_brier": float(current_row["brier"]),
        "outer_candidate_brier": float(candidate_row["brier"]),
        "outer_current_score": float(current_row["competition_score"]),
        "outer_candidate_score": float(candidate_row["competition_score"]),
        "outer_current_new_pitcher_brier": float(current_row["new_pitcher_brier"]),
        "outer_candidate_new_pitcher_brier": float(candidate_row["new_pitcher_brier"]),
        "paired_comparison_candidate_vs_current": comparison,
        "decision_reason": (
            "candidate must improve paired overall Brier and must not worsen the "
            "2024 new-pitcher cohort"
        ),
    }

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    selection_metrics.to_csv(args.artifact_dir / "selection_metrics.csv", index=False)
    outer_metrics.to_csv(args.artifact_dir / "outer_metrics.csv", index=False)
    outer_validation.loc[:, [ID_COL, TARGET_COL, RAW_COL, "rolling_calibrated"]].assign(
        selected_hierarchical=outer_prediction
    ).to_csv(args.artifact_dir / "outer_predictions.csv", index=False)
    (args.artifact_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(selection_metrics.head(10).to_string(index=False))
    print(outer_metrics.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
