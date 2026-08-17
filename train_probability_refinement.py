"""Compare rolling probability refinements on temporal-window ensemble OOF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model.probability_refinement import GameTypeLogitAdjuster, LogitInterceptCalibrator
from src.lg_aimers.metrics import brier_score, competition_score, paired_brier_comparison


ID_COL = "row_id"
TARGET_COL = "control_success"
PREDICTION_COL = "development_blend"


def season_weights(seasons, decay: float) -> np.ndarray:
    seasons = np.asarray(seasons, dtype=int)
    latest = int(seasons.max())
    return np.power(float(decay), latest - seasons)


def rolling_refinement(
    frame: pd.DataFrame,
    game_strength: float,
    game_shrinkage: float,
    calibration_strength: float,
    season_decay: float,
) -> pd.DataFrame:
    """Fit every season's correction only on OOF rows from earlier seasons."""
    result = frame.copy()
    result["game_type_corrected"] = result[PREDICTION_COL].astype(float)
    result["rolling_calibrated"] = result[PREDICTION_COL].astype(float)
    seasons = sorted(int(value) for value in result["validation_season"].unique())

    for validation_season in seasons[1:]:
        history = result["validation_season"] < validation_season
        current = result["validation_season"] == validation_season
        weights = season_weights(result.loc[history, "validation_season"], season_decay)

        game_adjuster = GameTypeLogitAdjuster(
            strength=game_strength,
            shrinkage=game_shrinkage,
        ).fit(
            result.loc[history, PREDICTION_COL],
            result.loc[history, TARGET_COL],
            result.loc[history, "game_type"],
            sample_weight=weights,
        )
        result.loc[current, "game_type_corrected"] = game_adjuster.transform(
            result.loc[current, PREDICTION_COL],
            result.loc[current, "game_type"],
        )

        calibrator = LogitInterceptCalibrator(strength=calibration_strength).fit(
            result.loc[history, "game_type_corrected"],
            result.loc[history, TARGET_COL],
            sample_weight=weights,
        )
        result.loc[current, "rolling_calibrated"] = calibrator.transform(
            result.loc[current, "game_type_corrected"]
        )
    return result


def load_branch(path: Path, branch: str, game_types: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=[ID_COL, TARGET_COL, "validation_season", PREDICTION_COL],
    )
    frame = frame.merge(game_types, on=ID_COL, how="left", validate="one_to_one")
    if frame["game_type"].isna().any():
        raise ValueError(f"{branch} OOF contains row_id values missing from train.csv")
    frame["branch"] = branch
    return frame


def metric_rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for branch, branch_frame in frame.groupby("branch", sort=False):
        for season, fold in branch_frame.groupby("validation_season", sort=True):
            truth = fold[TARGET_COL].to_numpy(float)
            for stage in (PREDICTION_COL, "game_type_corrected", "rolling_calibrated"):
                prediction = fold[stage].to_numpy(float)
                rows.append(
                    {
                        "branch": branch,
                        "validation_season": int(season),
                        "stage": stage,
                        "brier": brier_score(truth, prediction),
                        "competition_score": competition_score(truth, prediction),
                        "prediction_mean": float(prediction.mean()),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--temporal-oof", type=Path,
        default=Path("artifacts/temporal_ensemble/oof_predictions.csv"),
    )
    parser.add_argument(
        "--smoothed-oof", type=Path,
        default=Path("artifacts/probability_refinement/oof_predictions.csv"),
    )
    parser.add_argument(
        "--artifact-dir", type=Path,
        default=Path("artifacts/probability_refinement/final_comparison"),
    )
    parser.add_argument("--outer-season", type=int, default=2024)
    parser.add_argument("--season-decay", type=float, default=0.6)
    parser.add_argument("--game-strength", type=float, default=0.10)
    parser.add_argument("--game-shrinkage", type=float, default=100_000.0)
    parser.add_argument("--calibration-strength", type=float, default=0.25)
    args = parser.parse_args()

    game_types = pd.read_csv(
        args.data_dir / "train.csv", usecols=[ID_COL, "game_type"]
    )
    branches = []
    for path, name in (
        (args.temporal_oof, "temporal_original"),
        (args.smoothed_oof, "temporal_with_smoothing"),
    ):
        branch = load_branch(path, name, game_types)
        branches.append(
            rolling_refinement(
                branch,
                game_strength=args.game_strength,
                game_shrinkage=args.game_shrinkage,
                calibration_strength=args.calibration_strength,
                season_decay=args.season_decay,
            )
        )
    combined = pd.concat(branches, ignore_index=True)
    metrics = pd.DataFrame(metric_rows(combined))

    outer = combined[combined["validation_season"] == args.outer_season]
    raw = outer[outer["branch"] == "temporal_original"]
    smoothed = outer[outer["branch"] == "temporal_with_smoothing"]
    if not np.array_equal(raw[ID_COL].to_numpy(), smoothed[ID_COL].to_numpy()):
        raise ValueError("temporal branches have different outer-season row order")
    truth = raw[TARGET_COL].to_numpy(float)
    baseline = raw[PREDICTION_COL].to_numpy(float)
    full_chain = smoothed["rolling_calibrated"].to_numpy(float)
    best_refined = raw["rolling_calibrated"].to_numpy(float)

    summary = {
        "outer_season": args.outer_season,
        "architecture": (
            "temporal-window ensemble -> optional rate smoothing -> "
            "rolling game_type logit correction -> rolling global logit calibration"
        ),
        "rolling_rule": "each validation season uses only earlier-season OOF rows",
        "season_decay": args.season_decay,
        "game_strength": args.game_strength,
        "game_shrinkage": args.game_shrinkage,
        "calibration_strength": args.calibration_strength,
        "temporal_baseline_brier": brier_score(truth, baseline),
        "temporal_baseline_score": competition_score(truth, baseline),
        "full_chain_with_smoothing_brier": brier_score(truth, full_chain),
        "full_chain_with_smoothing_score": competition_score(truth, full_chain),
        "full_chain_vs_baseline": paired_brier_comparison(truth, baseline, full_chain),
        "refined_without_smoothing_brier": brier_score(truth, best_refined),
        "refined_without_smoothing_score": competition_score(truth, best_refined),
        "refined_without_smoothing_vs_baseline": paired_brier_comparison(
            truth, baseline, best_refined
        ),
        "decision": (
            "reject smoothing; retain temporal ensemble and treat post-processing "
            "result as diagnostic until a fresh outer season or leaderboard confirms it"
        ),
    }

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.artifact_dir / "metrics.csv", index=False)
    combined.to_csv(args.artifact_dir / "oof_predictions.csv", index=False)
    (args.artifact_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metrics[metrics["validation_season"] == args.outer_season].to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
