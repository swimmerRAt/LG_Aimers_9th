from __future__ import annotations

import unittest

from experiment_base_season_ablation import without_season_feature_columns


class BaseSeasonAblationTest(unittest.TestCase):
    def test_removes_only_season(self):
        self.assertEqual(
            without_season_feature_columns(["season", "inning", "li"]),
            ["inning", "li"],
        )

    def test_requires_season_in_official_features(self):
        with self.assertRaises(ValueError):
            without_season_feature_columns(["inning", "li"])


if __name__ == "__main__":
    unittest.main()
