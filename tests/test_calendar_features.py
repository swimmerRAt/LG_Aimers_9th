from __future__ import annotations

import unittest

import pandas as pd

from model.calendar_features import (
    CLIMATE_SEASON_COLUMNS,
    add_climate_season_indicators,
    climate_season_from_month,
)


class CalendarFeaturesTest(unittest.TestCase):
    def test_maps_all_months_to_korean_seasons(self):
        months = pd.Series(range(1, 13))
        self.assertEqual(
            climate_season_from_month(months).tolist(),
            [
                "winter",
                "winter",
                "spring",
                "spring",
                "spring",
                "summer",
                "summer",
                "summer",
                "autumn",
                "autumn",
                "autumn",
                "winter",
            ],
        )

    def test_replaces_month_with_four_one_hot_indicators(self):
        source = pd.DataFrame({"game_month": [3, 7, 10, 12], "inning": [1, 2, 3, 4]})
        result = add_climate_season_indicators(source)
        self.assertNotIn("game_month", result)
        self.assertEqual(result.columns.tolist(), ["inning", *CLIMATE_SEASON_COLUMNS])
        self.assertEqual(result.loc[:, CLIMATE_SEASON_COLUMNS].sum(axis=1).tolist(), [1, 1, 1, 1])

    def test_rejects_missing_or_invalid_months(self):
        for value in (None, 0, 13, 3.5, "bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                climate_season_from_month(pd.Series([value]))


if __name__ == "__main__":
    unittest.main()
