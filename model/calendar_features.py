"""Calendar-derived features that are safe to reproduce at inference time."""

from __future__ import annotations

import numpy as np
import pandas as pd


MONTH_COLUMN = "game_month"
CLIMATE_SEASONS = ("spring", "summer", "autumn", "winter")
CLIMATE_SEASON_COLUMNS = tuple(
    f"game_season_{season}" for season in CLIMATE_SEASONS
)


def climate_season_from_month(month) -> pd.Series:
    """Map calendar months to the four Korean meteorological seasons."""
    values = pd.to_numeric(month, errors="coerce")
    invalid = values.isna() | ~np.isclose(values, np.round(values))
    invalid |= ~values.between(1, 12)
    if invalid.any():
        bad = pd.Series(month).loc[invalid].head(5).tolist()
        raise ValueError(
            "game_month must contain integer values from 1 through 12; "
            f"invalid examples: {bad}"
        )
    values = values.astype(int)
    labels = np.select(
        [
            values.isin([3, 4, 5]),
            values.isin([6, 7, 8]),
            values.isin([9, 10, 11]),
        ],
        ["spring", "summer", "autumn"],
        default="winter",
    )
    return pd.Series(labels, index=values.index, name="climate_season")


def add_climate_season_indicators(
    frame: pd.DataFrame,
    *,
    month_column: str = MONTH_COLUMN,
    drop_month: bool = True,
) -> pd.DataFrame:
    """Add one binary feature per season, optionally replacing ``game_month``."""
    if month_column not in frame.columns:
        raise ValueError(f"calendar feature input is missing {month_column}")
    season = climate_season_from_month(frame[month_column])
    result = frame.drop(columns=[month_column]).copy() if drop_month else frame.copy()
    for label, column in zip(CLIMATE_SEASONS, CLIMATE_SEASON_COLUMNS):
        result[column] = season.eq(label).astype(np.int8)
    return result
