"""Leakage-safe forward validation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForwardSplit:
    validation_season: int
    train_index: np.ndarray
    validation_index: np.ndarray


def make_season_forward_splits(
    frame: pd.DataFrame,
    validation_seasons: list[int] | tuple[int, ...],
    season_col: str = "season",
) -> list[ForwardSplit]:
    if season_col not in frame.columns:
        raise ValueError(f"missing season column: {season_col}")
    if frame[season_col].isna().any():
        raise ValueError("season contains missing values")

    splits: list[ForwardSplit] = []
    for season in validation_seasons:
        train_mask = frame[season_col] < season
        valid_mask = frame[season_col] == season
        if not train_mask.any():
            raise ValueError(f"validation season {season} has no earlier training rows")
        if not valid_mask.any():
            raise ValueError(f"validation season {season} has no validation rows")
        splits.append(
            ForwardSplit(
                validation_season=int(season),
                train_index=np.flatnonzero(train_mask.to_numpy()),
                validation_index=np.flatnonzero(valid_mask.to_numpy()),
            )
        )
    return splits

