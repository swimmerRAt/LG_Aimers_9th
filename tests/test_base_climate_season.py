from __future__ import annotations

import unittest

import pandas as pd

from experiment_base_climate_season import (
    build_month_profile,
    climate_season_feature_columns,
)


class BaseClimateSeasonTest(unittest.TestCase):
    def test_adds_only_raw_month_to_official_model_inputs(self):
        self.assertEqual(
            climate_season_feature_columns(["season", "inning", "li"]),
            ["season", "inning", "li", "game_month"],
        )

    def test_rejects_existing_game_month(self):
        with self.assertRaises(ValueError):
            climate_season_feature_columns(["season", "game_month"])

    def test_builds_month_and_climate_profiles(self):
        frame = pd.DataFrame(
            {
                "season": [2024, 2024, 2024],
                "game_month": [3, 6, 10],
                "control_success": [1, 0, 1],
            }
        )
        month, climate = build_month_profile(frame)
        self.assertEqual(len(month), 3)
        self.assertEqual(
            set(climate["climate_season"]), {"spring", "summer", "autumn"}
        )


if __name__ == "__main__":
    unittest.main()
