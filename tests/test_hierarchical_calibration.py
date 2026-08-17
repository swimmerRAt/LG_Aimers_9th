from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiment_hierarchical_calibration import add_pitcher_cohorts, calibration_features


class HierarchicalCalibrationTest(unittest.TestCase):
    def test_new_pitcher_uses_only_prior_seasons(self):
        train = pd.DataFrame(
            {
                "row_id": ["old_history", "old_current", "new_current"],
                "season": [2021, 2022, 2022],
                "pitcher_id": [10, 10, 20],
                "asof_pitcher_n": [100, 120, 0],
                "game_type": ["R", "R", "F"],
            }
        )
        oof = pd.DataFrame(
            {
                "row_id": ["old_current", "new_current"],
                "control_success": [1, 0],
                "validation_season": [2022, 2022],
                "development_blend": [0.6, 0.4],
            }
        )
        result = add_pitcher_cohorts(oof, train)
        self.assertEqual(result["is_new_pitcher"].tolist(), [0, 1])
        self.assertEqual(result["is_low_sample"].tolist(), [0, 1])

    def test_hierarchical_features_are_finite(self):
        frame = pd.DataFrame(
            {
                "development_blend": [0.0, 1.0],
                "game_type": ["F", "R"],
                "is_new_pitcher": [1, 0],
                "is_low_sample": [1, 0],
                "log_pitcher_n": [0.0, np.log1p(1000)],
            }
        )
        features = calibration_features(frame, "hierarchical_interaction")
        self.assertEqual(features.shape, (2, 8))
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
