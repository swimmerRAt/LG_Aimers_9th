#!/usr/bin/env python3
"""Tune constrained official/CatBoost/residual OOF stack weights with Optuna."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from model.oof_stacking import (
    apply_logit_shift,
    constrained_stack_weights,
    robust_stack_objective,
)
from src.lg_aimers.metrics import brier_score, competition_score, paired_brier_comparison
from train_probability_refinement import rolling_refinement


ID_COL = "row_id"
TARGET_COL = "control_success"
CURRENT_COL = "rolling_calibrated"
VALIDATION_SEASONS = (2022, 2023, 2024)
DEVELOPMENT_SEASONS = (2022, 2023)
VARIANTS = ("without_mapping", "with_mapping")
BRANCHES = ("catboost", "spline_logistic")


def unique_cache(cache_dir: Path, branch: str, variant: str, season: int) -> Path:
    matches = sorted(cache_dir.glob(f"{branch}_{variant}_{season}_*.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one cache for {branch}/{variant}/{season}, found {len(matches)}"
        )
    return matches[0]


def reconstruct_refined_branch(
    train: pd.DataFrame,
    cache_dir: Path,
    branch: str,
    variant: str,
) -> pd.DataFrame:
    parts = []
    for season in VALIDATION_SEASONS:
        validation_mask = train["season"].eq(season).to_numpy()
        validation_indices = np.flatnonzero(validation_mask)
        cached = np.load(unique_cache(cache_dir, branch, variant, season))
        if not np.array_equal(cached["validation_indices"], validation_indices):
            raise ValueError(f"cache row mismatch for {branch}/{variant}/{season}")
        prediction = cached["prediction"].astype(float)
        if len(prediction) != int(validation_mask.sum()):
            raise ValueError(f"cache length mismatch for {branch}/{variant}/{season}")
        fold = train.loc[
            validation_mask, [ID_COL, TARGET_COL, "game_type"]
        ].copy()
        fold["validation_season"] = season
        fold["development_blend"] = prediction
        parts.append(fold)
    raw = pd.concat(parts, ignore_index=True)
    refined = rolling_refinement(
        raw,
        game_strength=0.10,
        game_shrinkage=100_000.0,
        calibration_strength=0.25,
        season_decay=0.6,
    )
    output_column = "catboost_refined" if branch == "catboost" else "residual_refined"
    return refined[[ID_COL, "rolling_calibrated"]].rename(
        columns={"rolling_calibrated": output_column}
    )


def load_variant_frame(
    train: pd.DataFrame,
    current: pd.DataFrame,
    cache_dir: Path,
    variant: str,
) -> pd.DataFrame:
    frame = current.copy()
    for branch in BRANCHES:
        frame = frame.merge(
            reconstruct_refined_branch(train, cache_dir, branch, variant),
            on=ID_COL,
            how="inner",
            validate="one_to_one",
        )
    if len(frame) != len(current):
        raise ValueError(f"{variant} branch rows do not match the official OOF rows")
    return frame


def tune_variant(
    frame: pd.DataFrame,
    trials: int,
    seed: int,
    minimum_official_weight: float,
    final_logit_shift: float,
    stability_penalty: float,
) -> tuple[optuna.Study, np.ndarray, float]:
    development = frame[frame["validation_season"].isin(DEVELOPMENT_SEASONS)]
    truth = development[TARGET_COL].to_numpy(float)
    matrix = development[
        [CURRENT_COL, "catboost_refined", "residual_refined"]
    ].to_numpy(float)
    seasons = development["validation_season"].to_numpy(int)

    def objective(trial: optuna.Trial) -> float:
        official = trial.suggest_float(
            "official_weight", minimum_official_weight, 1.0
        )
        cat_fraction = trial.suggest_float(
            "catboost_fraction_of_remainder", 0.0, 1.0
        )
        weights = constrained_stack_weights(
            official, cat_fraction, minimum_official_weight
        )
        trial.set_user_attr("catboost_weight", float(weights[1]))
        trial.set_user_attr("residual_weight", float(weights[2]))
        return robust_stack_objective(
            truth,
            matrix,
            seasons,
            weights,
            final_logit_shift,
            season_weights=(0.4, 0.6),
            stability_penalty=stability_penalty,
        )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
    )
    # Always compare the incumbent and interpretable boundary mixtures exactly.
    for official, cat_fraction in (
        (1.0, 0.5),
        (0.9, 1.0),
        (0.75, 1.0),
        (minimum_official_weight, 1.0),
        (0.9, 0.0),
        (0.75, 0.0),
        (minimum_official_weight, 0.0),
        (minimum_official_weight, 0.5),
    ):
        study.enqueue_trial(
            {
                "official_weight": official,
                "catboost_fraction_of_remainder": cat_fraction,
            }
        )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    params = study.best_params
    weights = constrained_stack_weights(
        params["official_weight"],
        params["catboost_fraction_of_remainder"],
        minimum_official_weight,
    )
    baseline_weights = np.asarray([1.0, 0.0, 0.0])
    baseline_objective = robust_stack_objective(
        truth,
        matrix,
        seasons,
        baseline_weights,
        final_logit_shift,
        season_weights=(0.4, 0.6),
        stability_penalty=stability_penalty,
    )
    return study, weights, float(baseline_objective)


def metric_record(name: str, truth, probability, mapped) -> dict:
    truth = np.asarray(truth, dtype=float)
    probability = np.asarray(probability, dtype=float)
    mapped = np.asarray(mapped, dtype=bool)
    return {
        "model": name,
        "brier": brier_score(truth, probability),
        "competition_score": competition_score(truth, probability),
        "prediction_mean": float(probability.mean()),
        "mapped_brier": brier_score(truth[mapped], probability[mapped]),
        "unmapped_brier": brier_score(truth[~mapped], probability[~mapped]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--current-oof",
        type=Path,
        default=Path("artifacts/probability_refinement/final_comparison/oof_predictions.csv"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/oof_stacking_mapping_ablation/cache"),
    )
    parser.add_argument(
        "--cohort-file",
        type=Path,
        default=Path("artifacts/oof_stacking_mapping_ablation/outer_predictions.csv"),
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/optuna_stack_weights")
    )
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-official-weight", type=float, default=0.5)
    parser.add_argument("--final-logit-shift", type=float, default=-0.0461)
    parser.add_argument("--stability-penalty", type=float, default=0.25)
    args = parser.parse_args()

    if args.trials < 8:
        raise ValueError("trials must be at least 8 to include all seeded candidates")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(
        args.data_dir / "train.csv",
        usecols=[ID_COL, TARGET_COL, "season", "game_type"],
    )
    current = pd.read_csv(args.current_oof)
    current = current[current["branch"].eq("temporal_original")][
        [ID_COL, TARGET_COL, "validation_season", CURRENT_COL]
    ].copy()
    current = current[current["validation_season"].isin(VALIDATION_SEASONS)]
    cohort = pd.read_csv(args.cohort_file, usecols=[ID_COL, "tm_is_mapped"])

    selections = []
    variant_results = {}
    for offset, variant in enumerate(VARIANTS):
        frame = load_variant_frame(train, current, args.cache_dir, variant)
        study, weights, baseline_objective = tune_variant(
            frame,
            args.trials,
            args.seed + offset,
            args.minimum_official_weight,
            args.final_logit_shift,
            args.stability_penalty,
        )
        trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
        trials.to_csv(args.artifact_dir / f"{variant}_trials.csv", index=False)
        selections.append(
            {
                "variant": variant,
                "trials": len(study.trials),
                "development_objective": float(study.best_value),
                "official_only_objective": baseline_objective,
                "development_objective_delta": float(study.best_value - baseline_objective),
                "official_weight": float(weights[0]),
                "catboost_weight": float(weights[1]),
                "residual_weight": float(weights[2]),
            }
        )
        variant_results[variant] = {"frame": frame, "weights": weights}
        print(
            f"{variant}: objective={study.best_value:.10f}, "
            f"weights={weights.tolist()}",
            flush=True,
        )

    selection = pd.DataFrame(selections).sort_values("development_objective")
    selection.to_csv(args.artifact_dir / "selected_weights.csv", index=False)
    chosen_variant = str(selection.iloc[0]["variant"])

    development_rows = []
    for variant, result in variant_results.items():
        development = result["frame"][
            result["frame"]["validation_season"].isin(DEVELOPMENT_SEASONS)
        ]
        for season, fold in development.groupby("validation_season", sort=True):
            truth_fold = fold[TARGET_COL].to_numpy(float)
            matrix_fold = fold[
                [CURRENT_COL, "catboost_refined", "residual_refined"]
            ].to_numpy(float)
            for model_name, probability in (
                (
                    "official_852_baseline",
                    apply_logit_shift(fold[CURRENT_COL], args.final_logit_shift),
                ),
                (
                    f"optuna_{variant}",
                    apply_logit_shift(
                        matrix_fold @ result["weights"], args.final_logit_shift
                    ),
                ),
            ):
                development_rows.append(
                    {
                        "validation_season": int(season),
                        "model": model_name,
                        "brier": brier_score(truth_fold, probability),
                        "competition_score": competition_score(
                            truth_fold, probability
                        ),
                    }
                )
    development_metrics = pd.DataFrame(development_rows).drop_duplicates()
    development_metrics.to_csv(
        args.artifact_dir / "development_fold_metrics.csv", index=False
    )

    outer_reference = variant_results["without_mapping"]["frame"]
    outer_reference = outer_reference[outer_reference["validation_season"].eq(2024)].copy()
    outer_reference = outer_reference.merge(
        cohort, on=ID_COL, how="left", validate="one_to_one"
    )
    if outer_reference["tm_is_mapped"].isna().any():
        raise ValueError("missing Trackman cohort flags for 2024 OOF rows")
    truth = outer_reference[TARGET_COL].to_numpy(float)
    mapped = outer_reference["tm_is_mapped"].to_numpy(float) > 0.5
    baseline_probability = apply_logit_shift(
        outer_reference[CURRENT_COL], args.final_logit_shift
    )
    outer_rows = [
        metric_record("official_852_baseline", truth, baseline_probability, mapped)
    ]
    predictions = {"official_852_baseline": baseline_probability}
    pairwise = {}
    for variant, result in variant_results.items():
        outer = result["frame"][result["frame"]["validation_season"].eq(2024)]
        if not np.array_equal(outer[ID_COL].to_numpy(), outer_reference[ID_COL].to_numpy()):
            raise ValueError(f"2024 row order mismatch for {variant}")
        matrix = outer[
            [CURRENT_COL, "catboost_refined", "residual_refined"]
        ].to_numpy(float)
        probability = apply_logit_shift(matrix @ result["weights"], args.final_logit_shift)
        name = f"optuna_{variant}"
        predictions[name] = probability
        outer_rows.append(metric_record(name, truth, probability, mapped))
        pairwise[f"{name}_vs_baseline"] = paired_brier_comparison(
            truth, baseline_probability, probability
        )
    pairwise["with_mapping_vs_without_mapping"] = paired_brier_comparison(
        truth,
        predictions["optuna_without_mapping"],
        predictions["optuna_with_mapping"],
    )
    outer_metrics = pd.DataFrame(outer_rows)
    outer_metrics.to_csv(args.artifact_dir / "outer_diagnostic_metrics.csv", index=False)
    outer_reference.assign(**predictions).to_csv(
        args.artifact_dir / "outer_predictions.csv", index=False
    )

    chosen_name = f"optuna_{chosen_variant}"
    chosen_metrics = outer_metrics[outer_metrics["model"].eq(chosen_name)].iloc[0]
    baseline_metrics = outer_metrics[
        outer_metrics["model"].eq("official_852_baseline")
    ].iloc[0]
    diagnostic_improved = bool(chosen_metrics["brier"] < baseline_metrics["brier"])
    summary = {
        "status": (
            "diagnostic_improvement_requires_fresh_validation"
            if diagnostic_improved
            else "rejected_keep_official_852_model"
        ),
        "official_leaderboard_baseline_score": 852.1984993386,
        "design": (
            "Optuna TPE continuously tunes official/CatBoost/spline-logistic weights; "
            "official weight is constrained to at least 50%"
        ),
        "trials_per_variant": args.trials,
        "selection_seasons": list(DEVELOPMENT_SEASONS),
        "outer_diagnostic_season": 2024,
        "outer_is_reused_not_one_shot": True,
        "minimum_official_weight": args.minimum_official_weight,
        "fixed_final_logit_shift": args.final_logit_shift,
        "stability_penalty": args.stability_penalty,
        "selected_variant_from_development": chosen_variant,
        "selections": selection.to_dict(orient="records"),
        "development_fold_metrics": development_metrics.to_dict(orient="records"),
        "outer_diagnostic_metrics": outer_metrics.to_dict(orient="records"),
        "paired_comparisons": pairwise,
        "diagnostic_improved": diagnostic_improved,
        "adopted": False,
        "adoption_note": (
            "2024 has already informed prior experiments, so it is a repeated diagnostic. "
            "Keep the official model unless a fresh leaderboard/future holdout confirms the candidate."
        ),
    }
    (args.artifact_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
