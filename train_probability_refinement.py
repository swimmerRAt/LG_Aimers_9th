"""Compare rolling probability refinements on temporal-window ensemble OOF."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from model.probability_refinement import (
    GameTypeLogitAdjuster,
    LogitInterceptCalibrator,
    RefinedProbabilityClassifier,
)
from model.temporal_ensemble import COMPONENT_NAMES, TemporalWindowEnsemble
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


def fit_final_refined_model(
    train: pd.DataFrame,
    feature_columns: list[str],
    temporal_oof: pd.DataFrame,
    component_weights: tuple[float, ...],
    season_decay: float,
    game_strength: float,
    game_shrinkage: float,
    calibration_strength: float,
) -> RefinedProbabilityClassifier:
    base_model = TemporalWindowEnsemble(
        component_weights=component_weights,
        hist_weight=0.45,
        n_estimators=160,
        random_state=42,
        smoothing_lambdas=(),
    ).fit(train.loc[:, feature_columns], train[TARGET_COL].to_numpy())

    weights = season_weights(temporal_oof["validation_season"], season_decay)
    raw = temporal_oof[PREDICTION_COL].to_numpy(float)
    target = temporal_oof[TARGET_COL].to_numpy(float)
    groups = temporal_oof["game_type"].to_numpy(object)
    game_adjuster = GameTypeLogitAdjuster(
        strength=game_strength,
        shrinkage=game_shrinkage,
    ).fit(raw, target, groups, sample_weight=weights)
    game_corrected = game_adjuster.transform(raw, groups)
    calibrator = LogitInterceptCalibrator(strength=calibration_strength).fit(
        game_corrected, target, sample_weight=weights
    )
    return RefinedProbabilityClassifier(base_model, game_adjuster, calibrator)


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
    parser.add_argument("--fit-final", action="store_true")
    parser.add_argument(
        "--temporal-summary", type=Path,
        default=Path("artifacts/temporal_ensemble/run_summary.json"),
    )
    parser.add_argument("--model-output", type=Path, default=Path("model/final_model.pkl"))
    parser.add_argument(
        "--backup-output", type=Path,
        default=Path("model/final_model_before_temporal_refinement.pkl"),
    )
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
            "reject smoothing; promote temporal ensemble plus rolling game_type and "
            "global logit corrections by user decision"
        ),
        "model_output": None,
    }

    if args.fit_final:
        temporal_summary = json.loads(args.temporal_summary.read_text(encoding="utf-8"))
        feature_columns = list(temporal_summary["feature_columns"])
        development_weights = temporal_summary["development_component_weights"]
        component_weights = tuple(
            float(development_weights[name]) for name in COMPONENT_NAMES
        )
        train = pd.read_csv(
            args.data_dir / "train.csv",
            usecols=[ID_COL, TARGET_COL, *feature_columns],
        )
        original_oof = combined[combined["branch"] == "temporal_original"].copy()
        started = time.perf_counter()
        final_model = fit_final_refined_model(
            train,
            feature_columns,
            original_oof,
            component_weights,
            args.season_decay,
            args.game_strength,
            args.game_shrinkage,
            args.calibration_strength,
        )
        summary["final_fit_seconds"] = time.perf_counter() - started
        artifact = {
            "model": final_model,
            "feature_columns": feature_columns,
            "positive_class": 1,
            "selected_model": "temporal_window_ensemble_with_probability_refinement",
            "validation_seasons": [2022, 2023, 2024],
            "full_train_rows": len(train),
            "full_train_target_rate": float(train[TARGET_COL].mean()),
            "component_weights": dict(zip(COMPONENT_NAMES, component_weights)),
            "smoothing_lambdas": [],
            "season_decay": args.season_decay,
            "game_strength": args.game_strength,
            "game_shrinkage": args.game_shrinkage,
            "calibration_strength": args.calibration_strength,
            "versions": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__,
            },
        }
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        candidate_path = args.model_output.with_name(
            f"{args.model_output.stem}_candidate{args.model_output.suffix}"
        )
        joblib.dump(artifact, candidate_path, compress=3)
        loaded = joblib.load(candidate_path)
        sample = train.loc[:4, feature_columns]
        sample_probability = loaded["model"].predict_proba(sample)
        if sample_probability.shape != (len(sample), 2) or not np.isfinite(sample_probability).all():
            raise ValueError("saved candidate model failed prediction smoke test")
        if args.model_output.is_file() and not args.backup_output.is_file():
            shutil.copy2(args.model_output, args.backup_output)
        candidate_path.replace(args.model_output)
        summary["model_output"] = str(args.model_output)
        summary["backup_output"] = str(args.backup_output)
        summary["model_size_mb"] = args.model_output.stat().st_size / 1024 / 1024

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
