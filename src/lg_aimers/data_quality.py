"""Chunked data-quality checks for the competition CSV files."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import ID_COL, TARGET_COL

RATE_SUFFIXES = ("_rate",)
ASOF_COLS = [
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]


def _add_counts(target: Counter, values: pd.Series) -> None:
    target.update({str(k): int(v) for k, v in values.items()})


def _id_duplicate_count(values: pd.Series, seen: set[str]) -> int:
    as_text = values.astype(str)
    duplicates = int(as_text.duplicated().sum())
    first_occurrences = as_text[~as_text.duplicated()]
    duplicates += int(first_occurrences.isin(seen).sum())
    seen.update(first_occurrences.tolist())
    return duplicates


def profile_train(path: Path, chunksize: int = 200_000) -> dict[str, Any]:
    rows = 0
    nulls: Counter = Counter()
    target_counts: Counter = Counter()
    season_rows: Counter = Counter()
    season_success: Counter = Counter()
    missing_by_season: dict[str, Counter] = defaultdict(Counter)
    seen_ids: set[str] = set()
    duplicate_ids = 0
    invalid: Counter = Counter()
    distinct: dict[str, set[Any]] = {
        name: set() for name in ["top_bottom", "game_type", "base_state"]
    }
    dtypes: dict[str, str] = {}

    base_map = {
        "___": (0, 0, 0), "1__": (1, 0, 0), "_2_": (0, 1, 0),
        "__3": (0, 0, 1), "12_": (1, 1, 0), "1_3": (1, 0, 1),
        "_23": (0, 1, 1), "123": (1, 1, 1),
    }

    for chunk in pd.read_csv(path, encoding="utf-8-sig", chunksize=chunksize):
        if not dtypes:
            dtypes = {col: str(dtype) for col, dtype in chunk.dtypes.items()}
        rows += len(chunk)
        nulls.update(chunk.isna().sum().astype(int).to_dict())
        duplicate_ids += _id_duplicate_count(chunk[ID_COL], seen_ids)
        _add_counts(target_counts, chunk[TARGET_COL].value_counts(dropna=False))
        _add_counts(season_rows, chunk["season"].value_counts(dropna=False))
        _add_counts(
            season_success,
            chunk.groupby("season", dropna=False)[TARGET_COL].sum().astype(int),
        )
        for season, group in chunk.groupby("season", dropna=False):
            missing_by_season[str(season)].update(
                group[ASOF_COLS].isna().sum().astype(int).to_dict()
            )
        for col in distinct:
            distinct[col].update(chunk[col].dropna().unique().tolist())

        invalid["target_not_binary"] += int((~chunk[TARGET_COL].isin([0, 1])).sum())
        invalid["season_outside_2019_2024"] += int((~chunk["season"].between(2019, 2024)).sum())
        invalid["game_month_outside_1_12"] += int((~chunk["game_month"].between(1, 12)).sum())
        invalid["dayofweek_outside_0_6"] += int((~chunk["game_dayofweek"].between(0, 6)).sum())
        invalid["balls_outside_0_3"] += int((~chunk["balls_before"].between(0, 3)).sum())
        invalid["strikes_outside_0_2"] += int((~chunk["strikes_before"].between(0, 2)).sum())
        invalid["outs_outside_0_2"] += int((~chunk["outs_before"].between(0, 2)).sum())
        for col in ["runner_on_1b", "runner_on_2b", "runner_on_3b"]:
            invalid[f"{col}_not_binary"] += int((~chunk[col].isin([0, 1])).sum())
        runner_sum = chunk[["runner_on_1b", "runner_on_2b", "runner_on_3b"]].sum(axis=1)
        invalid["runner_count_mismatch"] += int((runner_sum != chunk["num_runners_on"]).sum())
        encoded_base = chunk["base_state"].map(base_map)
        expected_base = list(zip(chunk["runner_on_1b"], chunk["runner_on_2b"], chunk["runner_on_3b"]))
        invalid["base_state_mismatch"] += sum(
            value != expected for value, expected in zip(encoded_base, expected_base)
        )
        invalid["run_total_mismatch"] += int(
            (chunk["run_total_before"] != chunk["run_top_before"] + chunk["run_bot_before"]).sum()
        )
        invalid["score_diff_home_mismatch"] += int(
            (chunk["score_diff_home"] != chunk["run_bot_before"] - chunk["run_top_before"]).sum()
        )
        expectancy_sum = chunk["home_win_expectancy"] + chunk["away_win_expectancy"]
        invalid["win_expectancy_sum_not_100"] += int((~np.isclose(expectancy_sum, 100.0, atol=0.11)).sum())
        for col in [c for c in chunk.columns if c.endswith(RATE_SUFFIXES)]:
            values = chunk[col].dropna()
            invalid[f"{col}_outside_0_1"] += int((~values.between(0.0, 1.0)).sum())

    seasonal = {}
    for season, count in season_rows.items():
        seasonal[season] = {
            "rows": int(count),
            "target_rate": float(season_success[season] / count),
            "missing_asof_cells": int(sum(missing_by_season[season].values())),
        }
    return {
        "rows": rows,
        "columns": len(dtypes),
        "dtypes": dtypes,
        "duplicate_row_ids": duplicate_ids,
        "unique_row_ids": len(seen_ids),
        "target_counts": dict(target_counts),
        "target_rate": float(int(target_counts.get("1", 0)) / rows),
        "null_counts": dict(nulls),
        "null_rates": {col: float(count / rows) for col, count in nulls.items()},
        "distinct_values": {col: sorted(map(str, values)) for col, values in distinct.items()},
        "seasonal": seasonal,
        "invalid_counts": {key: int(value) for key, value in invalid.items()},
    }


def profile_trackman(path: Path, chunksize: int = 250_000) -> dict[str, Any]:
    rows = 0
    nulls: Counter = Counter()
    season_rows: Counter = Counter()
    seen_ids: set[str] = set()
    duplicate_ids = 0
    invalid: Counter = Counter()
    pitchers: set[int] = set()
    pitch_groups: set[str] = set()
    date_min = None
    date_max = None

    for chunk in pd.read_csv(path, encoding="utf-8-sig", chunksize=chunksize):
        rows += len(chunk)
        nulls.update(chunk.isna().sum().astype(int).to_dict())
        duplicate_ids += _id_duplicate_count(chunk["trackman_id"], seen_ids)
        _add_counts(season_rows, chunk["season"].value_counts(dropna=False))
        pitchers.update(chunk["pitcher_trackman_id"].dropna().astype(int).unique().tolist())
        pitch_groups.update(chunk["pitch_type_group"].dropna().astype(str).unique().tolist())

        # Source switches from M/D/YYYY to YYYY-MM-DD in later seasons.
        dates = pd.to_datetime(chunk["game_date"], format="mixed", errors="coerce")
        invalid["invalid_game_date"] += int(dates.isna().sum())
        valid_dates = dates.dropna()
        if not valid_dates.empty:
            local_min, local_max = valid_dates.min(), valid_dates.max()
            date_min = local_min if date_min is None else min(date_min, local_min)
            date_max = local_max if date_max is None else max(date_max, local_max)
            invalid["season_date_mismatch"] += int(
                (chunk.loc[valid_dates.index, "season"] != valid_dates.dt.year).sum()
            )
            invalid["month_date_mismatch"] += int(
                (chunk.loc[valid_dates.index, "game_month"] != valid_dates.dt.month).sum()
            )
            invalid["dayofweek_date_mismatch"] += int(
                (chunk.loc[valid_dates.index, "game_dayofweek"] != valid_dates.dt.dayofweek).sum()
            )
        invalid["balls_outside_0_3"] += int((~chunk["balls_before"].between(0, 3)).sum())
        invalid["strikes_outside_0_2"] += int((~chunk["strikes_before"].between(0, 2)).sum())
        invalid["outs_outside_0_2"] += int((~chunk["outs_before"].between(0, 2)).sum())
        invalid["pitch_no_not_positive"] += int((chunk["pitch_no"] < 1).sum())

    return {
        "rows": rows,
        "columns": len(pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns),
        "duplicate_trackman_ids": duplicate_ids,
        "unique_trackman_ids": len(seen_ids),
        "null_counts": dict(nulls),
        "null_rates": {col: float(count / rows) for col, count in nulls.items()},
        "season_rows": dict(season_rows),
        "date_min": None if date_min is None else str(date_min.date()),
        "date_max": None if date_max is None else str(date_max.date()),
        "unique_pitchers": len(pitchers),
        "pitcher_ids": pitchers,
        "pitch_type_groups": sorted(pitch_groups),
        "invalid_counts": {key: int(value) for key, value in invalid.items()},
    }


def run_data_quality_audit(data_dir: Path | str) -> dict[str, Any]:
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    trackman_path = data_dir / "trackman_history.csv"
    for path in [train_path, test_path, sample_path, trackman_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    train_columns = pd.read_csv(train_path, encoding="utf-8-sig", nrows=0).columns.tolist()
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    trackman = profile_trackman(trackman_path)
    trackman_pitchers = trackman.pop("pitcher_ids")
    train = profile_train(train_path)

    main_pitchers: set[int] = set()
    for chunk in pd.read_csv(train_path, encoding="utf-8-sig", usecols=["pitcher_id"], chunksize=300_000):
        main_pitchers.update(chunk["pitcher_id"].dropna().astype(int).unique().tolist())

    feature_columns = [col for col in train_columns if col not in [ID_COL, TARGET_COL]]
    return {
        "schema": {
            "train_columns": train_columns,
            "test_columns": test.columns.tolist(),
            "train_test_feature_schema_matches": train_columns[:-1] == test.columns.tolist(),
            "target_is_last_train_column": train_columns[-1] == TARGET_COL,
            "feature_count": len(feature_columns),
        },
        "train": train,
        "test_sample": {
            "rows": len(test),
            "columns": len(test.columns),
            "duplicate_row_ids": int(test[ID_COL].duplicated().sum()),
            "null_cells": int(test.isna().sum().sum()),
            "season_values": sorted(test["season"].unique().tolist()),
        },
        "sample_submission": {
            "rows": len(sample),
            "columns": sample.columns.tolist(),
            "duplicate_row_ids": int(sample[ID_COL].duplicated().sum()),
            "id_order_matches_test": sample[ID_COL].tolist() == test[ID_COL].tolist(),
            "id_set_matches_test": set(sample[ID_COL]) == set(test[ID_COL]),
        },
        "trackman": trackman,
        "linkage": {
            "main_unique_pitchers": len(main_pitchers),
            "trackman_unique_pitchers": len(trackman_pitchers),
            "direct_pitcher_id_overlap": len(main_pitchers & trackman_pitchers),
        },
    }
