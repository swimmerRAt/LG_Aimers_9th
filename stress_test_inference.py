#!/usr/bin/env python3
"""Estimate inference runtime on the published 245,789-row evaluation size."""

from __future__ import annotations

import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from script import build_submission, load_artifact, positive_class_probability

EXPECTED_TEST_ROWS = 245_789


def max_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return float(value / 1024 / 1024 if platform.system() == "Darwin" else value / 1024)


def main() -> None:
    root = Path(__file__).resolve().parent
    sample_test = pd.read_csv(root / "data" / "test.csv", encoding="utf-8-sig")
    repeats = int(np.ceil(EXPECTED_TEST_ROWS / len(sample_test)))
    test = pd.concat([sample_test] * repeats, ignore_index=True).iloc[:EXPECTED_TEST_ROWS].copy()
    test["row_id"] = [f"STRESS_{index:06d}" for index in range(EXPECTED_TEST_ROWS)]
    sample = pd.DataFrame({"row_id": test["row_id"], "control_success": 0.5})

    load_started = time.perf_counter()
    model, feature_columns, positive_class = load_artifact(root / "model" / "final_model.pkl")
    load_seconds = time.perf_counter() - load_started
    inference_started = time.perf_counter()
    predictions = positive_class_probability(model, test[feature_columns], positive_class)
    inference_seconds = time.perf_counter() - inference_started
    validation_started = time.perf_counter()
    submission = build_submission(test, sample, predictions)
    validation_seconds = time.perf_counter() - validation_started

    result = {
        "rows": len(test),
        "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "output_validation_seconds": validation_seconds,
        "total_core_seconds": load_seconds + inference_seconds + validation_seconds,
        "rows_per_second": len(test) / inference_seconds,
        "max_rss_mib": max_rss_mib(),
        "finite_probabilities": bool(np.isfinite(submission["control_success"]).all()),
        "probability_min": float(submission["control_success"].min()),
        "probability_max": float(submission["control_success"].max()),
        "note": "Rows repeat the 5-row schema sample; this measures scale, not distributional behavior.",
    }
    output = root / "artifacts" / "runtime_stress.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

