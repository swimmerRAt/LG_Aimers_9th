#!/usr/bin/env python3
"""Run and persist the chunked competition data-quality audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.lg_aimers.data_quality import run_data_quality_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data_quality"))
    args = parser.parse_args()

    report = run_data_quality_audit(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    null_rates = pd.Series(report["train"]["null_rates"], name="null_rate")
    null_rates.sort_values(ascending=False).rename_axis("column").to_csv(
        args.output_dir / "train_null_rates.csv"
    )
    seasonal = pd.DataFrame.from_dict(report["train"]["seasonal"], orient="index")
    seasonal.rename_axis("season").to_csv(args.output_dir / "train_season_summary.csv")
    print(f"saved: {json_path}")
    print(json.dumps({
        "schema": report["schema"],
        "train_rows": report["train"]["rows"],
        "target_rate": report["train"]["target_rate"],
        "duplicate_row_ids": report["train"]["duplicate_row_ids"],
        "invalid_counts": report["train"]["invalid_counts"],
        "trackman_rows": report["trackman"]["rows"],
        "trackman_invalid_counts": report["trackman"]["invalid_counts"],
        "linkage": report["linkage"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

