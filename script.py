#!/usr/bin/env python3
"""Offline inference entrypoint executed by the DACON evaluation server."""

from __future__ import annotations

import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"


def find_data_dir(root: Path) -> Path:
    candidates = []
    if os.environ.get("DATA_DIR"):
        candidates.append(Path(os.environ["DATA_DIR"]))
    candidates.extend([root / "data", root / "open"])
    for candidate in candidates:
        if (candidate / "test.csv").is_file() and (candidate / "sample_submission.csv").is_file():
            return candidate
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"test.csv and sample_submission.csv not found; checked: {checked}")


def load_artifact(path: Path):
    artifact = joblib.load(path)
    if isinstance(artifact, dict):
        required = {"model", "feature_columns"}
        missing = required - artifact.keys()
        if missing:
            raise ValueError(f"model artifact missing keys: {sorted(missing)}")
        return artifact["model"], list(artifact["feature_columns"]), artifact.get("positive_class", 1)
    feature_columns = list(getattr(artifact, "feature_names_in_", []))
    if not feature_columns:
        raise ValueError("legacy model has no feature_names_in_; cannot guarantee feature order")
    return artifact, feature_columns, 1


def positive_class_probability(model, features: pd.DataFrame, positive_class=1) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    classes = np.asarray(model.classes_)
    positions = np.flatnonzero(classes == positive_class)
    if len(positions) != 1:
        raise ValueError(f"positive class {positive_class!r} not found exactly once in {classes.tolist()}")
    return probabilities[:, int(positions[0])]


def build_submission(test: pd.DataFrame, sample: pd.DataFrame, predictions) -> pd.DataFrame:
    if ID_COL not in test.columns:
        raise ValueError(f"test.csv missing {ID_COL}")
    if sample.columns.tolist() != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample submission columns must be {[ID_COL, TARGET_COL]}")
    if test[ID_COL].isna().any() or sample[ID_COL].isna().any():
        raise ValueError("row_id contains missing values")
    if test[ID_COL].duplicated().any() or sample[ID_COL].duplicated().any():
        raise ValueError("row_id must be unique in both test and sample submission")
    if len(test) != len(sample) or set(test[ID_COL]) != set(sample[ID_COL]):
        raise ValueError("test and sample submission row_id sets do not match exactly")

    probabilities = np.asarray(predictions, dtype=float)
    if probabilities.shape != (len(test),):
        raise ValueError(f"prediction shape {probabilities.shape} does not match test rows {len(test)}")
    if not np.isfinite(probabilities).all():
        raise ValueError("predictions contain NaN or infinity")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("predictions must be inside [0, 1]")

    pred_by_id = pd.Series(probabilities, index=test[ID_COL])
    aligned = pred_by_id.loc[sample[ID_COL]].to_numpy()
    return pd.DataFrame({ID_COL: sample[ID_COL].to_numpy(), TARGET_COL: aligned})


def main() -> None:
    started = time.perf_counter()
    root = Path(__file__).resolve().parent
    data_dir = find_data_dir(root)
    preferred_model = root / "model" / "final_model.pkl"
    model_path = preferred_model if preferred_model.is_file() else root / "model" / "rf.pkl"
    output_path = root / "output" / "submission.csv"

    model, feature_columns, positive_class = load_artifact(model_path)
    test = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")
    sample = pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")
    missing_features = [column for column in feature_columns if column not in test.columns]
    if missing_features:
        raise ValueError(f"test.csv missing model features: {missing_features}")

    predictions = positive_class_probability(model, test.loc[:, feature_columns], positive_class)
    submission = build_submission(test, sample, predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, encoding="utf-8")
    elapsed = time.perf_counter() - started
    print(f"saved {output_path} | rows={len(submission):,} | elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()
