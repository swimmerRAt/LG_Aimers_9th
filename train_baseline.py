#!/usr/bin/env python3
"""Train leakage-safe forward-validation baselines and save the best model."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.lg_aimers import ID_COL, TARGET_COL
from src.lg_aimers.metrics import brier_score, competition_score
from src.lg_aimers.modeling import make_model
from src.lg_aimers.validation import make_season_forward_splits


def positive_probability(model, frame: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(frame)
    classes = np.asarray(model.classes_)
    positions = np.flatnonzero(classes == 1)
    if len(positions) != 1:
        raise ValueError(f"class 1 missing from fitted model classes: {classes.tolist()}")
    return np.asarray(probabilities[:, int(positions[0])], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-output", type=Path, default=Path("model/final_model.pkl"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["constant", "logistic", "random_forest", "histgb"],
        default=["constant", "random_forest"],
    )
    parser.add_argument("--validation-seasons", nargs="+", type=int, default=[2024])
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"
    test_columns = pd.read_csv(test_path, encoding="utf-8-sig", nrows=0).columns.tolist()
    if ID_COL not in test_columns:
        raise ValueError(f"{test_path} missing {ID_COL}")
    feature_columns = [column for column in test_columns if column != ID_COL]
    train = pd.read_csv(
        train_path,
        encoding="utf-8-sig",
        usecols=[ID_COL, TARGET_COL, *feature_columns],
    )
    if train[ID_COL].duplicated().any():
        raise ValueError("train row_id is not unique")
    if not train[TARGET_COL].isin([0, 1]).all():
        raise ValueError("target must contain only 0 and 1")

    splits = make_season_forward_splits(train, args.validation_seasons)
    metrics: list[dict] = []
    oof_parts: list[pd.DataFrame] = []

    for model_name in args.models:
        for split in splits:
            fitted = make_model(model_name, feature_columns, args.random_state)
            train_rows = train.iloc[split.train_index]
            validation_rows = train.iloc[split.validation_index]
            fit_started = time.perf_counter()
            fitted.fit(train_rows[feature_columns], train_rows[TARGET_COL])
            fit_seconds = time.perf_counter() - fit_started
            infer_started = time.perf_counter()
            predictions = positive_probability(fitted, validation_rows[feature_columns])
            inference_seconds = time.perf_counter() - infer_started
            brier = brier_score(validation_rows[TARGET_COL].to_numpy(), predictions)
            score = competition_score(validation_rows[TARGET_COL].to_numpy(), predictions)
            row = {
                "model": model_name,
                "validation_season": split.validation_season,
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "train_target_rate": float(train_rows[TARGET_COL].mean()),
                "validation_target_rate": float(validation_rows[TARGET_COL].mean()),
                "brier": brier,
                "competition_score": score,
                "fit_seconds": fit_seconds,
                "inference_seconds": inference_seconds,
            }
            metrics.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            oof_parts.append(pd.DataFrame({
                ID_COL: validation_rows[ID_COL].to_numpy(),
                TARGET_COL: validation_rows[TARGET_COL].to_numpy(),
                "validation_season": split.validation_season,
                "model": model_name,
                "prediction": predictions,
            }))
            del fitted, train_rows, validation_rows, predictions
            gc.collect()

    metrics_frame = pd.DataFrame(metrics)
    mean_brier = metrics_frame.groupby("model")["brier"].mean().sort_values()
    selected_name = str(mean_brier.index[0])
    print(f"selected model: {selected_name} | mean Brier={mean_brier.iloc[0]:.8f}")

    final_model = make_model(selected_name, feature_columns, args.random_state)
    fit_started = time.perf_counter()
    final_model.fit(train[feature_columns], train[TARGET_COL])
    full_fit_seconds = time.perf_counter() - fit_started
    artifact = {
        "model": final_model,
        "feature_columns": feature_columns,
        "positive_class": 1,
        "selected_model": selected_name,
        "validation_seasons": args.validation_seasons,
        "full_train_rows": len(train),
        "full_train_target_rate": float(train[TARGET_COL].mean()),
        "random_state": args.random_state,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_output, compress=3)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(args.artifact_dir / "metrics.csv", index=False)
    pd.concat(oof_parts, ignore_index=True).to_csv(args.artifact_dir / "oof_predictions.csv", index=False)
    (args.artifact_dir / "run_summary.json").write_text(
        json.dumps({
            "selected_model": selected_name,
            "mean_brier_by_model": mean_brier.to_dict(),
            "full_fit_seconds": full_fit_seconds,
            "model_output": str(args.model_output),
            "versions": artifact["versions"],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved {args.model_output} | full fit={full_fit_seconds:.1f}s")


if __name__ == "__main__":
    main()
