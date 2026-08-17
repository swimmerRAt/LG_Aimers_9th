#!/usr/bin/env python3
"""Stabilize CatBoost tree count, then blend it with the official 852 model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lg_aimers_matplotlib")
)

import matplotlib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from experiment_oof_stacking_mapping_ablation import (
    CAT_COLUMNS,
    catboost_frame,
    catboost_params,
)
from model.oof_stacking import apply_logit_shift, robust_stack_objective
from src.lg_aimers.metrics import brier_score, competition_score, paired_brier_comparison
from train_probability_refinement import rolling_refinement


ID_COL = "row_id"
TARGET_COL = "control_success"
CURRENT_COL = "rolling_calibrated"
VALIDATION_SEASONS = (2022, 2023, 2024)
DEVELOPMENT_SEASONS = (2022, 2023)


def iteration_grid(max_iterations: int, eval_period: int) -> np.ndarray:
    if max_iterations < eval_period or eval_period <= 0:
        raise ValueError("max_iterations must be at least one positive eval_period")
    values = list(range(eval_period, max_iterations + 1, eval_period))
    if values[-1] != max_iterations:
        values.append(max_iterations)
    return np.asarray(values, dtype=int)


def cache_path(
    cache_dir: Path,
    validation_season: int,
    features: list[str],
    max_iterations: int,
    eval_period: int,
) -> Path:
    signature = hashlib.sha256(
        (
            "catboost-iteration-stability-v1|"
            + str(validation_season)
            + f"|{max_iterations}|{eval_period}|"
            + "|".join(features)
        ).encode()
    ).hexdigest()[:12]
    return cache_dir / f"learning_curve_{validation_season}_{signature}.npz"


def fit_learning_curve(
    data: pd.DataFrame,
    features: list[str],
    validation_season: int,
    max_iterations: int,
    eval_period: int,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    validation_mask = data["season"].eq(validation_season).to_numpy()
    validation_indices = np.flatnonzero(validation_mask)
    candidates = iteration_grid(max_iterations, eval_period)
    destination = cache_path(
        cache_dir,
        validation_season,
        features,
        max_iterations,
        eval_period,
    )
    if destination.is_file():
        cached = np.load(destination)
        if (
            np.array_equal(cached["validation_indices"], validation_indices)
            and np.array_equal(cached["iterations"], candidates)
        ):
            return cached["predictions"].astype(float), candidates, 0.0, "cache"

    training_mask = data["season"].lt(validation_season).to_numpy()
    categorical = [column for column in CAT_COLUMNS if column in features]
    model = CatBoostClassifier(**catboost_params(max_iterations))
    started = time.perf_counter()
    model.fit(
        catboost_frame(data.loc[training_mask], features),
        data.loc[training_mask, TARGET_COL].to_numpy(),
        cat_features=categorical,
    )
    validation = catboost_frame(data.loc[validation_mask], features)
    staged = list(model.staged_predict_proba(validation, eval_period=eval_period))
    predictions = np.asarray([value[:, 1] for value in staged], dtype=np.float32)
    elapsed = time.perf_counter() - started
    if predictions.shape != (len(candidates), len(validation_indices)):
        raise ValueError(
            f"unexpected staged prediction shape {predictions.shape}; "
            f"expected {(len(candidates), len(validation_indices))}"
        )
    np.savez_compressed(
        destination,
        validation_indices=validation_indices,
        iterations=candidates,
        predictions=predictions,
    )
    del model, validation, staged
    gc.collect()
    return predictions.astype(float), candidates, elapsed, "fitted"


def refine_candidate(
    base_oof: pd.DataFrame,
    raw_prediction: np.ndarray,
) -> np.ndarray:
    source = base_oof[
        [ID_COL, TARGET_COL, "validation_season", "game_type"]
    ].copy()
    source["development_blend"] = np.asarray(raw_prediction, dtype=float)
    refined = rolling_refinement(
        source,
        game_strength=0.10,
        game_shrinkage=100_000.0,
        calibration_strength=0.25,
        season_decay=0.6,
    )
    return refined["rolling_calibrated"].to_numpy(float)


def plot_learning_curves(
    learning_curve: pd.DataFrame,
    selected_iterations: int,
    destination: Path,
) -> None:
    """Render season-level refined Brier curves with honest independent y-scales."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    blue = "#2563EB"
    gold = "#C58A00"
    charcoal = "#374151"
    for axis, season in zip(axes, VALIDATION_SEASONS):
        fold = learning_curve[learning_curve["validation_season"].eq(season)]
        best = fold.sort_values("refined_brier").iloc[0]
        axis.plot(
            fold["iterations"],
            fold["refined_brier"],
            color=blue,
            linewidth=2.0,
        )
        axis.scatter(
            [best["iterations"]],
            [best["refined_brier"]],
            color=gold,
            edgecolor=charcoal,
            linewidth=0.8,
            zorder=3,
            label=f"season minimum: {int(best['iterations'])}",
        )
        axis.axvline(
            selected_iterations,
            color=charcoal,
            linestyle="--",
            linewidth=1.2,
            label=f"selected: {selected_iterations}",
        )
        axis.set_ylabel(f"{season} Brier")
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="best", frameon=False, fontsize=9)
    axes[-1].set_xlabel("CatBoost iterations")
    fig.suptitle("CatBoost iteration learning curves", fontsize=16, x=0.08, ha="left")
    fig.text(
        0.08,
        0.945,
        "Rolling-refined Brier by validation season; lower is better",
        fontsize=10,
        color="#4B5563",
        ha="left",
    )
    fig.tight_layout(rect=(0.04, 0.04, 0.99, 0.92))
    fig.savefig(destination, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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
        default=Path("artifacts/catboost_iteration_stability"),
    )
    parser.add_argument("--max-iterations", type=int, default=400)
    parser.add_argument("--eval-period", type=int, default=10)
    parser.add_argument("--optuna-trials", type=int, default=1000)
    parser.add_argument("--minimum-official-weight", type=float, default=0.5)
    parser.add_argument("--final-logit-shift", type=float, default=-0.0461)
    parser.add_argument("--stability-penalty", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 <= args.minimum_official_weight <= 1.0:
        raise ValueError("minimum official weight must be inside [0, 1]")
    if args.optuna_trials < 8:
        raise ValueError("optuna-trials must be at least 8")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.artifact_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    temporal_summary = json.loads(args.temporal_summary.read_text(encoding="utf-8"))
    features = list(
        dict.fromkeys([*temporal_summary["feature_columns"], "pitcher_id", "batter_id"])
    )
    train = pd.read_csv(
        args.data_dir / "train.csv",
        usecols=[ID_COL, TARGET_COL, *features],
    )
    current = pd.read_csv(args.current_oof)
    current = current[
        current["branch"].eq("temporal_original")
        & current["validation_season"].isin(VALIDATION_SEASONS)
    ][[ID_COL, TARGET_COL, "validation_season", CURRENT_COL]].copy()

    curve_by_season = {}
    fit_rows = []
    candidates = iteration_grid(args.max_iterations, args.eval_period)
    base_parts = []
    for season in VALIDATION_SEASONS:
        validation_mask = train["season"].eq(season).to_numpy()
        fold = train.loc[validation_mask, [ID_COL, TARGET_COL, "game_type"]].copy()
        fold["validation_season"] = season
        base_parts.append(fold)
        predictions, fold_candidates, seconds, source = fit_learning_curve(
            train,
            features,
            season,
            args.max_iterations,
            args.eval_period,
            cache_dir,
        )
        if not np.array_equal(candidates, fold_candidates):
            raise ValueError("candidate iteration grids differ across folds")
        curve_by_season[season] = predictions
        fit_rows.append(
            {
                "validation_season": season,
                "source": source,
                "fit_and_prediction_seconds": seconds,
                "training_rows": int(train["season"].lt(season).sum()),
                "validation_rows": int(validation_mask.sum()),
            }
        )
        print(
            f"season={season} source={source} seconds={seconds:.1f} "
            f"stages={len(candidates)}",
            flush=True,
        )

    base_oof = pd.concat(base_parts, ignore_index=True)
    base_oof = base_oof.merge(
        current,
        on=[ID_COL, TARGET_COL, "validation_season"],
        how="inner",
        validate="one_to_one",
    )
    expected_rows = sum(len(part) for part in base_parts)
    if len(base_oof) != expected_rows:
        raise ValueError("official OOF rows do not align with CatBoost validation rows")

    raw_matrix = np.concatenate(
        [curve_by_season[season] for season in VALIDATION_SEASONS], axis=1
    )
    refined_matrix = np.empty_like(raw_matrix, dtype=np.float32)
    curve_rows = []
    for index, iterations in enumerate(candidates):
        refined_matrix[index] = refine_candidate(base_oof, raw_matrix[index])
        for season in VALIDATION_SEASONS:
            mask = base_oof["validation_season"].eq(season).to_numpy()
            curve_rows.append(
                {
                    "validation_season": season,
                    "iterations": int(iterations),
                    "raw_brier": brier_score(
                        base_oof.loc[mask, TARGET_COL], raw_matrix[index, mask]
                    ),
                    "refined_brier": brier_score(
                        base_oof.loc[mask, TARGET_COL], refined_matrix[index, mask]
                    ),
                }
            )
    learning_curve = pd.DataFrame(curve_rows)
    learning_curve.to_csv(args.artifact_dir / "learning_curve_metrics.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(args.artifact_dir / "fit_summary.csv", index=False)

    development_mask = base_oof["validation_season"].isin(DEVELOPMENT_SEASONS).to_numpy()
    development_truth = base_oof.loc[development_mask, TARGET_COL].to_numpy(float)
    development_official = base_oof.loc[development_mask, CURRENT_COL].to_numpy(float)
    development_seasons = base_oof.loc[
        development_mask, "validation_season"
    ].to_numpy(int)
    baseline_matrix = np.column_stack([development_official, development_official])
    baseline_development_objective = robust_stack_objective(
        development_truth,
        baseline_matrix,
        development_seasons,
        np.asarray([1.0, 0.0]),
        args.final_logit_shift,
        season_weights=(0.4, 0.6),
        stability_penalty=args.stability_penalty,
    )
    shifted_development_official = apply_logit_shift(
        development_official, args.final_logit_shift
    )

    def objective(trial: optuna.Trial) -> float:
        iterations = trial.suggest_categorical(
            "iterations", [int(value) for value in candidates]
        )
        official_weight = trial.suggest_float(
            "official_weight", args.minimum_official_weight, 1.0
        )
        candidate_index = int(np.flatnonzero(candidates == iterations)[0])
        catboost_prediction = refined_matrix[candidate_index, development_mask]
        matrix = np.column_stack([development_official, catboost_prediction])
        weights = np.asarray([official_weight, 1.0 - official_weight])
        candidate_probability = apply_logit_shift(
            matrix @ weights, args.final_logit_shift
        )
        fold_deltas = []
        for season in DEVELOPMENT_SEASONS:
            fold = development_seasons == season
            fold_deltas.append(
                brier_score(development_truth[fold], candidate_probability[fold])
                - brier_score(
                    development_truth[fold], shifted_development_official[fold]
                )
            )
        maximum_fold_delta = float(max(fold_deltas))
        all_folds_non_degraded = maximum_fold_delta <= 1e-12
        trial.set_user_attr("maximum_development_fold_brier_delta", maximum_fold_delta)
        trial.set_user_attr(
            "all_development_folds_non_degraded", all_folds_non_degraded
        )
        if not all_folds_non_degraded:
            # Keep all infeasible trials strictly behind the enqueued official-only trial.
            return baseline_development_objective + 1.0 + maximum_fold_delta
        return robust_stack_objective(
            development_truth,
            matrix,
            development_seasons,
            weights,
            args.final_logit_shift,
            season_weights=(0.4, 0.6),
            stability_penalty=args.stability_penalty,
        )

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    seeded_iterations = [10, 50, 100, 150, 200, 300, 400]
    for iterations in seeded_iterations:
        if iterations in candidates:
            study.enqueue_trial({"iterations": iterations, "official_weight": 0.5})
    study.enqueue_trial({"iterations": int(candidates[0]), "official_weight": 1.0})
    study.optimize(objective, n_trials=args.optuna_trials, show_progress_bar=False)
    trials = study.trials_dataframe(
        attrs=("number", "value", "params", "user_attrs", "state")
    )
    trials.to_csv(args.artifact_dir / "optuna_trials.csv", index=False)

    selected_iterations = int(study.best_params["iterations"])
    official_weight = float(study.best_params["official_weight"])
    catboost_weight = 1.0 - official_weight
    selected_index = int(np.flatnonzero(candidates == selected_iterations)[0])
    feasible_nonzero_trials = sum(
        bool(trial.user_attrs.get("all_development_folds_non_degraded", False))
        and float(trial.params.get("official_weight", 1.0)) < 1.0 - 1e-12
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    )

    fold_rows = []
    for season in VALIDATION_SEASONS:
        mask = base_oof["validation_season"].eq(season).to_numpy()
        truth = base_oof.loc[mask, TARGET_COL].to_numpy(float)
        official = apply_logit_shift(
            base_oof.loc[mask, CURRENT_COL], args.final_logit_shift
        )
        catboost = apply_logit_shift(
            refined_matrix[selected_index, mask], args.final_logit_shift
        )
        blend = apply_logit_shift(
            official_weight * base_oof.loc[mask, CURRENT_COL].to_numpy(float)
            + catboost_weight * refined_matrix[selected_index, mask],
            args.final_logit_shift,
        )
        for name, probability in (
            ("official_852_baseline", official),
            ("fixed_iteration_catboost", catboost),
            ("selected_blend", blend),
        ):
            fold_rows.append(
                {
                    "validation_season": season,
                    "model": name,
                    "brier": brier_score(truth, probability),
                    "competition_score": competition_score(truth, probability),
                    "prediction_mean": float(np.mean(probability)),
                }
            )
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(args.artifact_dir / "fold_metrics.csv", index=False)
    plot_learning_curves(
        learning_curve,
        selected_iterations,
        args.artifact_dir / "learning_curve.png",
    )

    outer = base_oof["validation_season"].eq(2024).to_numpy()
    outer_truth = base_oof.loc[outer, TARGET_COL].to_numpy(float)
    outer_official = apply_logit_shift(
        base_oof.loc[outer, CURRENT_COL], args.final_logit_shift
    )
    outer_blend = apply_logit_shift(
        official_weight * base_oof.loc[outer, CURRENT_COL].to_numpy(float)
        + catboost_weight * refined_matrix[selected_index, outer],
        args.final_logit_shift,
    )
    paired = paired_brier_comparison(outer_truth, outer_official, outer_blend)
    outer_baseline_brier = brier_score(outer_truth, outer_official)
    outer_candidate_brier = brier_score(outer_truth, outer_blend)
    diagnostic_improved = bool(outer_candidate_brier < outer_baseline_brier)

    best_raw_by_season = {}
    best_refined_by_season = {}
    for season in VALIDATION_SEASONS:
        season_curve = learning_curve[learning_curve["validation_season"].eq(season)]
        best_raw = season_curve.sort_values("raw_brier").iloc[0]
        best_refined = season_curve.sort_values("refined_brier").iloc[0]
        best_raw_by_season[str(season)] = {
            "iterations": int(best_raw["iterations"]),
            "brier": float(best_raw["raw_brier"]),
        }
        best_refined_by_season[str(season)] = {
            "iterations": int(best_refined["iterations"]),
            "brier": float(best_refined["refined_brier"]),
        }

    summary = {
        "status": (
            "diagnostic_improvement_requires_fresh_validation"
            if diagnostic_improved
            else "rejected_keep_official_852_model"
        ),
        "official_leaderboard_baseline_score": 852.1984993386,
        "question": (
            "Does selecting one robust CatBoost iteration count across 2022-2023 "
            "fix the prior 75->208->1 early-stopping collapse?"
        ),
        "features": "official feature set plus categorical pitcher_id and batter_id",
        "selection_seasons": list(DEVELOPMENT_SEASONS),
        "outer_diagnostic_season": 2024,
        "outer_is_reused_not_one_shot": True,
        "iteration_candidates": candidates.tolist(),
        "optuna_trials": args.optuna_trials,
        "minimum_official_weight": args.minimum_official_weight,
        "fixed_final_logit_shift": args.final_logit_shift,
        "stability_penalty": args.stability_penalty,
        "selected_iterations": selected_iterations,
        "official_weight": official_weight,
        "catboost_weight": catboost_weight,
        "development_objective": float(study.best_value),
        "official_only_development_objective": baseline_development_objective,
        "development_gate": (
            "candidate Brier must not exceed the shifted official baseline in either "
            "2022 or 2023"
        ),
        "feasible_nonzero_catboost_trials": feasible_nonzero_trials,
        "best_raw_iteration_by_season": best_raw_by_season,
        "best_refined_iteration_by_season": best_refined_by_season,
        "fold_metrics": fold_metrics.to_dict(orient="records"),
        "outer_baseline_brier": outer_baseline_brier,
        "outer_candidate_brier": outer_candidate_brier,
        "outer_brier_delta": outer_candidate_brier - outer_baseline_brier,
        "outer_paired_comparison": paired,
        "diagnostic_improved": diagnostic_improved,
        "adopted": False,
        "adoption_note": (
            "The official 852 model remains the submission baseline. The 2024 fold is "
            "reused and cannot independently authorize adoption."
        ),
    }
    (args.artifact_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    base_oof.loc[outer, [ID_COL, TARGET_COL, "validation_season"]].assign(
        official_852_probability=outer_official,
        selected_blend_probability=outer_blend,
    ).to_csv(args.artifact_dir / "outer_predictions.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
