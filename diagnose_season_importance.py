#!/usr/bin/env python3
"""Diagnose why the fitted ensemble assigns high importance to season."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def tree_depths(children_left: np.ndarray, children_right: np.ndarray) -> np.ndarray:
    depths = np.zeros(len(children_left), dtype=int)
    stack = [0]
    while stack:
        node = stack.pop()
        for child in (children_left[node], children_right[node]):
            if child >= 0:
                depths[child] = depths[node] + 1
                stack.append(int(child))
    return depths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--model", type=Path, default=Path("model/final_model.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/season_diagnostics"))
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    model = artifact["model"]
    features = list(artifact["feature_columns"])
    columns = ["control_success", *features]
    train = pd.read_csv(args.train, usecols=columns)

    season_summary = train.groupby("season", as_index=False).agg(
        rows=("control_success", "size"),
        target_rate=("control_success", "mean"),
        game_type_regular_share=("game_type", lambda value: float((value == "R").mean())),
        pitcher_history_rate_mean=("asof_pitcher_success_rate", "mean"),
        pitcher_history_rate_missing=("asof_pitcher_success_rate", lambda value: float(value.isna().mean())),
    )
    season_summary["target_rate_change_pp"] = season_summary["target_rate"].diff() * 100.0

    game_type = train.groupby(["season", "game_type"], as_index=False).agg(
        rows=("control_success", "size"),
        target_rate=("control_success", "mean"),
    )
    game_type["season_share"] = game_type["rows"] / game_type.groupby("season")["rows"].transform("sum")

    numeric_season = train["season"].to_numpy(dtype=float)
    target = train["control_success"].to_numpy(dtype=float)
    season_target_correlation = float(np.corrcoef(numeric_season, target)[0, 1])

    names = model.preprocessor_.get_feature_names_out()
    season_positions = np.flatnonzero(names == "num__season")
    if len(season_positions) != 1:
        raise ValueError(f"season not found exactly once in transformed features: {names.tolist()}")
    season_position = int(season_positions[0])
    importance = float(model.extra_model_.feature_importances_[season_position])
    split_count = 0
    internal_count = 0
    trees_using_season = 0
    split_depths: list[int] = []
    for estimator in model.extra_model_.estimators_:
        tree = estimator.tree_
        internal = tree.feature >= 0
        season_nodes = np.flatnonzero(tree.feature == season_position)
        internal_count += int(internal.sum())
        split_count += len(season_nodes)
        trees_using_season += int(len(season_nodes) > 0)
        depths = tree_depths(tree.children_left, tree.children_right)
        split_depths.extend(depths[season_nodes].tolist())

    # Counterfactual sensitivity: change only season on the same 2024 rows.
    # This measures model dependence, not a causal effect or a realistic data scenario.
    validation = train.loc[train["season"] == 2024, features].copy()
    counterfactual_rows = []
    for forced_season in range(2019, 2026):
        changed = validation.copy()
        changed["season"] = forced_season
        prediction = model.predict_proba(changed)[:, 1]
        counterfactual_rows.append({
            "forced_season": forced_season,
            "rows": len(changed),
            "prediction_mean": float(prediction.mean()),
            "prediction_std": float(prediction.std()),
        })
    counterfactual = pd.DataFrame(counterfactual_rows)
    baseline_mean = float(counterfactual.loc[counterfactual["forced_season"] == 2024, "prediction_mean"].iloc[0])
    counterfactual["change_vs_2024_pp"] = (
        counterfactual["prediction_mean"] - baseline_mean
    ) * 100.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    season_summary.to_csv(args.output_dir / "season_summary.csv", index=False)
    game_type.to_csv(args.output_dir / "game_type_by_season.csv", index=False)
    counterfactual.to_csv(args.output_dir / "season_counterfactual.csv", index=False)
    diagnostics = {
        "target_grain": "one pitch",
        "target": "control_success",
        "season_target_pearson_correlation": season_target_correlation,
        "extra_trees_season_importance": importance,
        "extra_trees_trees": len(model.extra_model_.estimators_),
        "trees_using_season": trees_using_season,
        "season_split_count": split_count,
        "all_internal_split_count": internal_count,
        "season_split_share": split_count / internal_count,
        "season_split_median_depth": float(np.median(split_depths)),
        "season_split_min_depth": int(np.min(split_depths)),
        "season_split_max_depth": int(np.max(split_depths)),
        "counterfactual_note": "Only season was changed; results measure model sensitivity, not causal effect.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print(season_summary.to_string(index=False))
    print(counterfactual.to_string(index=False))


if __name__ == "__main__":
    main()
