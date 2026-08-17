#!/usr/bin/env python3
"""Offline inference entrypoint executed by the DACON evaluation server."""

from __future__ import annotations

import os
import time
from html import escape
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


def feature_importance_frame(model) -> pd.DataFrame | None:
    """Extract feature importance exposed by the fitted final model."""
    method = getattr(model, "feature_importance_frame", None)
    if method is None:
        return None
    names, importance = method()
    frame = pd.DataFrame({
        "feature": np.asarray(names, dtype=str),
        "importance": np.asarray(importance, dtype=float),
    })
    if frame.empty or not np.isfinite(frame["importance"]).all():
        raise ValueError("feature importance is empty or contains non-finite values")
    if (frame["importance"] < 0).any():
        raise ValueError("feature importance must be non-negative")
    frame = frame.sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    frame["importance_percent"] = 100.0 * frame["importance"]
    frame["source_component"] = getattr(
        model,
        "feature_importance_source",
        type(model).__name__,
    )
    return frame


def render_feature_importance_svg(frame: pd.DataFrame, top_n: int = 20) -> str:
    """Render a dependency-free horizontal importance chart as SVG."""
    required = {"feature", "importance", "importance_percent"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"feature importance frame missing columns: {sorted(missing)}")
    chart = frame.head(max(1, int(top_n))).reset_index(drop=True)
    source = str(chart["source_component"].iloc[0]) if "source_component" in chart else "Model"
    model_label = source.split(maxsplit=1)[0]
    width = 1400
    left = 470
    right = 155
    top = 150
    row_height = 56
    bar_height = 30
    bottom = 130
    plot_width = width - left - right
    height = top + len(chart) * row_height + bottom
    maximum = float(chart["importance"].max())
    if not np.isfinite(maximum) or maximum <= 0:
        maximum = 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;fill:#172033}'
        '.title{font-size:28px;font-weight:700}.subtitle{font-size:16px;fill:#566176}'
        '.label{font-size:15px}.value{font-size:14px;font-variant-numeric:tabular-nums}'
        '.tick{font-size:12px;fill:#6B7280}.note{font-size:14px;fill:#566176}</style>',
        f'<text class="title" x="48" y="48">{escape(model_label)} Feature Importance</text>',
        f'<text class="subtitle" x="48" y="78">Top {len(chart)} of {len(frame)} features · impurity-based importance</text>',
        '<text class="subtitle" x="48" y="104">Correlated features can split importance; use validation permutation before changing model weights.</text>',
    ]
    for tick in range(5):
        fraction = tick / 4
        x = left + fraction * plot_width
        value = fraction * maximum * 100.0
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" y2="{height - bottom + 8}" '
            'stroke="#E5E7EB" stroke-width="1"/>'
        )
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{top - 22}" text-anchor="middle">{value:.1f}%</text>'
        )
    for index, row in chart.iterrows():
        center_y = top + index * row_height + row_height / 2
        y = center_y - bar_height / 2
        bar_width = float(row["importance"]) / maximum * plot_width
        feature = escape(str(row["feature"]))
        percent = float(row["importance_percent"])
        parts.extend([
            f'<text class="label" x="{left - 16}" y="{center_y + 5:.1f}" text-anchor="end">{feature}</text>',
            f'<rect x="{left}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height}" rx="3" fill="#2563EB"/>',
            f'<text class="value" x="{left + bar_width + 10:.1f}" y="{center_y + 5:.1f}">{percent:.3f}%</text>',
        ])
    parts.extend([
        f'<text class="note" x="48" y="{height - 48}">Source: {escape(source)}. Values sum to 100% across all features.</text>',
        '</svg>',
    ])
    return "\n".join(parts)


def write_feature_importance_outputs(model, output_dir: Path) -> tuple[Path, Path] | None:
    frame = feature_importance_frame(model)
    if frame is None:
        return None
    csv_path = output_dir / "feature_importance.csv"
    svg_path = output_dir / "feature_importance.svg"
    frame.to_csv(csv_path, index=False, encoding="utf-8")
    svg_path.write_text(render_feature_importance_svg(frame), encoding="utf-8")
    return csv_path, svg_path


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
    importance_paths = write_feature_importance_outputs(model, output_path.parent)
    elapsed = time.perf_counter() - started
    print(f"saved {output_path} | rows={len(submission):,} | elapsed={elapsed:.2f}s")
    if importance_paths is not None:
        csv_path, svg_path = importance_paths
        print(f"saved {csv_path} and {svg_path}")


if __name__ == "__main__":
    main()
