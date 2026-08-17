#!/usr/bin/env python3
"""Compare 852 baseline, OOF stacking, and the same stack with Trackman mapping."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from model.oof_stacking import (
    apply_logit_shift,
    error_correlation_long,
    select_simplex_stack_weights,
)
from model.trackman_features import (
    MAIN_RATE_COLUMNS,
    PHYSICAL_COLUMNS,
    TrackmanMatchThresholds,
    add_temporal_trackman_features,
    trackman_feature_columns,
)
from src.lg_aimers.metrics import brier_score, competition_score, paired_brier_comparison
from train_probability_refinement import rolling_refinement


ID_COL = "row_id"
TARGET_COL = "control_success"
CURRENT_COL = "rolling_calibrated"
VALIDATION_SEASONS = (2022, 2023, 2024)
DEVELOPMENT_SEASONS = (2022, 2023)
CAT_COLUMNS = (
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_hand",
    "batter_hand",
    "top_bottom",
    "game_type",
    "base_state",
)
RESIDUAL_CAT_COLUMNS = (
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_hand",
    "batter_hand",
    "top_bottom",
    "game_type",
    "base_state",
)
SPLINE_COLUMNS = (
    "season",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "home_win_expectancy",
    "li",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
)


def catboost_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    result = frame.loc[:, features].copy()
    for column in CAT_COLUMNS:
        if column in result:
            result[column] = result[column].fillna("__MISSING__").astype(str)
    return result


def catboost_params(iterations: int, random_state: int = 42) -> dict:
    return {
        "loss_function": "Logloss",
        "eval_metric": "BrierScore",
        "iterations": int(iterations),
        "learning_rate": 0.05,
        "depth": 7,
        "l2_leaf_reg": 10.0,
        "random_strength": 1.0,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "rsm": 0.8,
        "random_seed": random_state,
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": False,
    }


def residual_pipeline(features: list[str]) -> Pipeline:
    categorical = [column for column in RESIDUAL_CAT_COLUMNS if column in features]
    spline = [column for column in SPLINE_COLUMNS if column in features]
    excluded = set(categorical) | set(spline) | {"pitcher_id", "batter_id"}
    linear = [column for column in features if column not in excluded]
    transformer = ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=True,
                                dtype=np.float32,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "spline",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        (
                            "spline",
                            SplineTransformer(
                                n_knots=4,
                                degree=2,
                                knots="quantile",
                                include_bias=False,
                                sparse_output=True,
                            ),
                        ),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
                spline,
            ),
            (
                "linear",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler(with_mean=False)),
                    ]
                ),
                linear,
            ),
        ],
        sparse_threshold=1.0,
    )
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=5e-5,
        max_iter=80,
        tol=1e-4,
        average=True,
        random_state=42,
    )
    return Pipeline([("features", transformer), ("classifier", classifier)])


def cache_file(
    cache_dir: Path,
    branch: str,
    variant: str,
    validation_season: int,
    features: list[str],
) -> Path:
    signature = hashlib.sha256(
        (
            "stack-ablation-v1|"
            + branch
            + "|"
            + variant
            + "|"
            + str(validation_season)
            + "|"
            + "|".join(features)
        ).encode()
    ).hexdigest()[:12]
    return cache_dir / f"{branch}_{variant}_{validation_season}_{signature}.npz"


def load_cache(path: Path, validation_indices: np.ndarray):
    if not path.is_file():
        return None
    cached = np.load(path)
    if not np.array_equal(cached["validation_indices"], validation_indices):
        return None
    return (
        cached["prediction"].astype(float),
        int(cached["selected_iterations"][0]),
    )


def fit_catboost_oof(
    data: pd.DataFrame,
    features: list[str],
    validation_season: int,
    variant: str,
    cache_dir: Path,
    iterations_cap: int,
) -> tuple[np.ndarray, int, float, str]:
    validation_mask = data["season"] == validation_season
    validation_indices = np.flatnonzero(validation_mask.to_numpy())
    destination = cache_file(
        cache_dir, "catboost", variant, validation_season, features
    )
    cached = load_cache(destination, validation_indices)
    if cached is not None:
        return cached[0], cached[1], 0.0, "cache"

    internal_validation_season = validation_season - 1
    internal_train = data["season"] < internal_validation_season
    internal_validation = data["season"] == internal_validation_season
    categorical = [column for column in CAT_COLUMNS if column in features]
    selector = CatBoostClassifier(**catboost_params(iterations_cap))
    selector.fit(
        catboost_frame(data.loc[internal_train], features),
        data.loc[internal_train, TARGET_COL].to_numpy(),
        cat_features=categorical,
        eval_set=(
            catboost_frame(data.loc[internal_validation], features),
            data.loc[internal_validation, TARGET_COL].to_numpy(),
        ),
        use_best_model=True,
        early_stopping_rounds=60,
    )
    best_iteration = selector.get_best_iteration()
    selected_iterations = int(best_iteration + 1 if best_iteration >= 0 else iterations_cap)
    del selector
    gc.collect()

    training_mask = data["season"] < validation_season
    model = CatBoostClassifier(**catboost_params(selected_iterations))
    started = time.perf_counter()
    model.fit(
        catboost_frame(data.loc[training_mask], features),
        data.loc[training_mask, TARGET_COL].to_numpy(),
        cat_features=categorical,
    )
    prediction = model.predict_proba(
        catboost_frame(data.loc[validation_mask], features)
    )[:, 1]
    fit_seconds = time.perf_counter() - started
    np.savez_compressed(
        destination,
        validation_indices=validation_indices,
        prediction=np.asarray(prediction, dtype=np.float32),
        selected_iterations=np.asarray([selected_iterations], dtype=int),
    )
    del model
    gc.collect()
    return np.asarray(prediction, dtype=float), selected_iterations, fit_seconds, "fitted"


def fit_residual_oof(
    data: pd.DataFrame,
    features: list[str],
    validation_season: int,
    variant: str,
    cache_dir: Path,
) -> tuple[np.ndarray, float, str]:
    training_mask = data["season"] < validation_season
    validation_mask = data["season"] == validation_season
    validation_indices = np.flatnonzero(validation_mask.to_numpy())
    destination = cache_file(
        cache_dir, "spline_logistic", variant, validation_season, features
    )
    cached = load_cache(destination, validation_indices)
    if cached is not None:
        return cached[0], 0.0, "cache"
    model = residual_pipeline(features)
    started = time.perf_counter()
    model.fit(
        data.loc[training_mask, features],
        data.loc[training_mask, TARGET_COL].to_numpy(),
    )
    prediction = model.predict_proba(data.loc[validation_mask, features])[:, 1]
    fit_seconds = time.perf_counter() - started
    np.savez_compressed(
        destination,
        validation_indices=validation_indices,
        prediction=np.asarray(prediction, dtype=np.float32),
        selected_iterations=np.asarray([0], dtype=int),
    )
    del model
    gc.collect()
    return np.asarray(prediction, dtype=float), fit_seconds, "fitted"


def refine_branch(
    oof: pd.DataFrame,
    prediction_column: str,
    output_column: str,
) -> pd.DataFrame:
    source = oof[
        [ID_COL, TARGET_COL, "validation_season", "game_type", prediction_column]
    ].rename(columns={prediction_column: "development_blend"})
    refined = rolling_refinement(
        source,
        game_strength=0.10,
        game_shrinkage=100_000.0,
        calibration_strength=0.25,
        season_decay=0.6,
    )
    return refined[[ID_COL, "rolling_calibrated"]].rename(
        columns={"rolling_calibrated": output_column}
    )


def metrics(frame: pd.DataFrame, prediction) -> dict[str, float]:
    truth = frame[TARGET_COL].to_numpy(float)
    probability = np.asarray(prediction, dtype=float)
    mapped = frame["tm_is_mapped"].to_numpy(float) > 0.5
    return {
        "brier": brier_score(truth, probability),
        "competition_score": competition_score(truth, probability),
        "prediction_mean": float(probability.mean()),
        "mapped_row_rate": float(mapped.mean()),
        "mapped_brier": (
            brier_score(truth[mapped], probability[mapped]) if mapped.any() else np.nan
        ),
        "unmapped_brier": (
            brier_score(truth[~mapped], probability[~mapped]) if (~mapped).any() else np.nan
        ),
    }


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
        default=Path("artifacts/oof_stacking_mapping_ablation"),
    )
    parser.add_argument("--catboost-iterations-cap", type=int, default=800)
    parser.add_argument("--final-logit-shift", type=float, default=-0.0461)
    args = parser.parse_args()

    summary = json.loads(args.temporal_summary.read_text(encoding="utf-8"))
    base_features = list(summary["feature_columns"])
    identity_columns = ["pitcher_id", "batter_id"]
    mapping_columns = [
        "asof_pitcher_pitchmix_n",
        *MAIN_RATE_COLUMNS,
    ]
    read_columns = list(
        dict.fromkeys(
            [ID_COL, TARGET_COL, *base_features, *identity_columns, *mapping_columns]
        )
    )
    train = pd.read_csv(args.data_dir / "train.csv", usecols=read_columns)
    trackman = pd.read_csv(
        args.data_dir / "trackman_history.csv",
        usecols=[
            "season",
            "pitcher_trackman_id",
            "pitcher_hand",
            "pitch_type_group",
            *PHYSICAL_COLUMNS,
        ],
    )
    enriched, mapping = add_temporal_trackman_features(
        train,
        trackman,
        thresholds=TrackmanMatchThresholds(),
        shrinkage=200.0,
    )
    del trackman
    gc.collect()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.artifact_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(args.artifact_dir / "mapping_assignments.csv", index=False)
    tm_features = trackman_feature_columns()
    variants = {
        "without_mapping": {
            "data": train,
            "cat_features": list(dict.fromkeys([*base_features, *identity_columns])),
            "residual_features": base_features,
        },
        "with_mapping": {
            "data": enriched,
            "cat_features": list(
                dict.fromkeys([*base_features, *identity_columns, *tm_features])
            ),
            "residual_features": [*base_features, *tm_features],
        },
    }

    oof_parts = []
    fit_rows = []
    for variant, config in variants.items():
        data = config["data"]
        for validation_season in VALIDATION_SEASONS:
            validation_mask = data["season"] == validation_season
            fold = pd.DataFrame(
                {
                    ID_COL: data.loc[validation_mask, ID_COL].to_numpy(),
                    TARGET_COL: data.loc[validation_mask, TARGET_COL].to_numpy(),
                    "validation_season": validation_season,
                    "game_type": data.loc[validation_mask, "game_type"].to_numpy(),
                    "tm_is_mapped": enriched.loc[
                        validation_mask, "tm_is_mapped"
                    ].to_numpy(),
                    "variant": variant,
                }
            )
            cat_prediction, iterations, seconds, source = fit_catboost_oof(
                data,
                config["cat_features"],
                validation_season,
                variant,
                cache_dir,
                args.catboost_iterations_cap,
            )
            residual_prediction, residual_seconds, residual_source = fit_residual_oof(
                data,
                config["residual_features"],
                validation_season,
                variant,
                cache_dir,
            )
            fold["catboost_raw"] = cat_prediction
            fold["residual_raw"] = residual_prediction
            oof_parts.append(fold)
            for branch, prediction, fit_seconds, branch_source in (
                ("catboost", cat_prediction, seconds, source),
                ("spline_logistic", residual_prediction, residual_seconds, residual_source),
            ):
                fit_rows.append(
                    {
                        "variant": variant,
                        "validation_season": validation_season,
                        "branch": branch,
                        "brier": brier_score(fold[TARGET_COL], prediction),
                        "competition_score": competition_score(
                            fold[TARGET_COL], prediction
                        ),
                        "selected_iterations": iterations if branch == "catboost" else 0,
                        "fit_seconds": fit_seconds,
                        "source": branch_source,
                    }
                )
            print(
                f"variant={variant} season={validation_season} "
                f"cat_brier={fit_rows[-2]['brier']:.8f} residual_brier={fit_rows[-1]['brier']:.8f} "
                f"cat_iterations={iterations}",
                flush=True,
            )

    branch_oof = pd.concat(oof_parts, ignore_index=True)
    refined_parts = []
    for variant, variant_oof in branch_oof.groupby("variant", sort=False):
        refined = variant_oof.copy()
        for raw_column, refined_column in (
            ("catboost_raw", "catboost_refined"),
            ("residual_raw", "residual_refined"),
        ):
            values = refine_branch(refined, raw_column, refined_column)
            refined = refined.merge(values, on=ID_COL, how="left", validate="one_to_one")
        refined_parts.append(refined)
    branch_oof = pd.concat(refined_parts, ignore_index=True)

    current = pd.read_csv(args.current_oof)
    current = current[current["branch"].eq("temporal_original")][
        [ID_COL, TARGET_COL, "validation_season", CURRENT_COL]
    ]
    comparisons = {}
    selection_rows = []
    for variant, variant_oof in branch_oof.groupby("variant", sort=False):
        frame = current.merge(
            variant_oof[
                [ID_COL, "tm_is_mapped", "catboost_refined", "residual_refined"]
            ],
            on=ID_COL,
            how="inner",
            validate="one_to_one",
        )
        if len(frame) != len(current):
            raise ValueError(f"{variant} OOF row set does not match current OOF")
        development = frame[
            frame["validation_season"].isin(DEVELOPMENT_SEASONS)
        ]
        matrix = development[
            [CURRENT_COL, "catboost_refined", "residual_refined"]
        ].to_numpy(float)
        weights, diagnostics = select_simplex_stack_weights(
            development[TARGET_COL].to_numpy(float),
            matrix,
            development["validation_season"].to_numpy(int),
            args.final_logit_shift,
        )
        selection_rows.append(
            {
                "variant": variant,
                **diagnostics,
                "current_weight": weights[0],
                "catboost_weight": weights[1],
                "residual_weight": weights[2],
            }
        )
        comparisons[variant] = {"frame": frame, "weights": weights}

    selection = pd.DataFrame(selection_rows)
    selection.to_csv(args.artifact_dir / "stack_selection.csv", index=False)
    chosen_variant = str(selection.sort_values("objective").iloc[0]["variant"])

    outer_rows = []
    prediction_map = {}
    outer_frames = {}
    for variant, values in comparisons.items():
        outer = values["frame"][
            values["frame"]["validation_season"] == 2024
        ].copy()
        outer_frames[variant] = outer
        matrix = outer[[CURRENT_COL, "catboost_refined", "residual_refined"]].to_numpy(float)
        stack_probability = apply_logit_shift(matrix @ values["weights"], args.final_logit_shift)
        cat_probability = apply_logit_shift(
            outer["catboost_refined"], args.final_logit_shift
        )
        residual_probability = apply_logit_shift(
            outer["residual_refined"], args.final_logit_shift
        )
        prediction_map[f"{variant}_catboost"] = cat_probability
        prediction_map[f"{variant}_residual"] = residual_probability
        prediction_map[f"{variant}_stack"] = stack_probability
        outer_rows.extend(
            [
                {
                    "model": f"{variant}_catboost",
                    "variant": variant,
                    "role": "direct_branch",
                    **metrics(outer, cat_probability),
                },
                {
                    "model": f"{variant}_residual",
                    "variant": variant,
                    "role": "direct_branch",
                    **metrics(outer, residual_probability),
                },
                {
                    "model": f"{variant}_selected_stack",
                    "variant": variant,
                    "role": "selected_stack",
                    **metrics(outer, stack_probability),
                },
            ]
        )

    reference_outer = outer_frames["without_mapping"]
    baseline_probability = apply_logit_shift(
        reference_outer[CURRENT_COL], args.final_logit_shift
    )
    prediction_map = {"official_852_baseline": baseline_probability, **prediction_map}
    outer_rows.insert(
        0,
        {
            "model": "official_852_baseline",
            "variant": "baseline",
            "role": "baseline",
            **metrics(reference_outer, baseline_probability),
        },
    )
    outer_metrics = pd.DataFrame(outer_rows)
    outer_metrics.to_csv(args.artifact_dir / "outer_metrics.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(args.artifact_dir / "branch_metrics.csv", index=False)

    truth = reference_outer[TARGET_COL].to_numpy(float)
    correlations = error_correlation_long(truth, prediction_map)
    correlations.to_csv(args.artifact_dir / "error_correlations.csv", index=False)
    no_map_prediction = prediction_map["without_mapping_stack"]
    map_prediction = prediction_map["with_mapping_stack"]
    pairwise = {
        "without_mapping_vs_baseline": paired_brier_comparison(
            truth, baseline_probability, no_map_prediction
        ),
        "with_mapping_vs_baseline": paired_brier_comparison(
            truth, baseline_probability, map_prediction
        ),
        "with_mapping_vs_without_mapping": paired_brier_comparison(
            truth, no_map_prediction, map_prediction
        ),
    }

    chosen_prediction = prediction_map[f"{chosen_variant}_stack"]
    chosen_comparison = pairwise[f"{chosen_variant}_vs_baseline"]
    chosen_metrics = metrics(reference_outer, chosen_prediction)
    baseline_metrics = metrics(reference_outer, baseline_probability)
    adopted = bool(
        chosen_metrics["brier"] < baseline_metrics["brier"]
        and chosen_comparison["paired_ci95_high"] < 0.0
    )
    run_summary = {
        "status": (
            f"accepted_{chosen_variant}_stack_requires_final_fit"
            if adopted
            else "rejected_keep_official_852_model"
        ),
        "official_leaderboard_baseline_score": 852.1984993386,
        "fixed_final_logit_shift": args.final_logit_shift,
        "comparison_design": (
            "same CatBoost + spline-logistic OOF stack with and without leakage-safe "
            "Trackman mapping features, compared with the official 852 structure"
        ),
        "selection_seasons": list(DEVELOPMENT_SEASONS),
        "outer_season": 2024,
        "chosen_variant_from_development": chosen_variant,
        "stack_selection": selection.to_dict(orient="records"),
        "outer_metrics": outer_metrics.to_dict(orient="records"),
        "paired_comparisons": pairwise,
        "adopted": adopted,
        "adoption_rule": (
            "choose mapping variant only by 2022-2023 robust objective, then require its "
            "2024 shifted Brier to improve with paired 95% CI entirely below zero"
        ),
        "mapping_is_preserved": True,
        "mapping_is_not_in_official_model_unless_stack_is_adopted": True,
    }
    (args.artifact_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    reference_outer.assign(
        official_852_probability=baseline_probability,
        without_mapping_stack_probability=no_map_prediction,
        with_mapping_stack_probability=map_prediction,
    ).to_csv(args.artifact_dir / "outer_predictions.csv", index=False)
    print(json.dumps(run_summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

