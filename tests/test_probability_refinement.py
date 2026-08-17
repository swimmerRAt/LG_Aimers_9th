from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from model.probability_refinement import GameTypeLogitAdjuster, LogitInterceptCalibrator
from train_probability_refinement import rolling_refinement


class ProbabilityRefinementTest(unittest.TestCase):
    def test_game_type_adjuster_moves_underpredicted_group_up(self):
        probability = np.array([0.4, 0.4, 0.6, 0.6])
        target = np.array([1, 1, 0, 1])
        groups = np.array(["F", "F", "R", "R"])
        model = GameTypeLogitAdjuster(strength=1.0, shrinkage=0.0).fit(
            probability, target, groups
        )
        transformed = model.transform(np.array([0.4, 0.6]), np.array(["F", "R"]))
        self.assertGreater(transformed[0], 0.4)
        self.assertLess(transformed[1], transformed[0])

    def test_unseen_game_type_uses_global_intercept(self):
        model = GameTypeLogitAdjuster(strength=0.5, shrinkage=10.0).fit(
            [0.4, 0.6], [0, 1], ["F", "R"]
        )
        transformed = model.transform([0.5], ["OOV"])
        self.assertTrue(0.0 < transformed[0] < 1.0)

    def test_partial_calibration_moves_mean_without_full_jump(self):
        probability = np.full(100, 0.4)
        target = np.r_[np.ones(60), np.zeros(40)]
        calibrated = LogitInterceptCalibrator(strength=0.25).fit(
            probability, target
        ).transform(probability)
        self.assertGreater(calibrated.mean(), 0.4)
        self.assertLess(calibrated.mean(), 0.6)

    def test_current_season_targets_do_not_change_current_predictions(self):
        frame = pd.DataFrame(
            {
                "row_id": [f"r{index}" for index in range(12)],
                "control_success": [0, 1, 0, 1] * 3,
                "validation_season": np.repeat([2022, 2023, 2024], 4),
                "development_blend": [0.35, 0.65, 0.45, 0.55] * 3,
                "game_type": ["F", "F", "R", "R"] * 3,
                "branch": "test",
            }
        )
        changed = frame.copy()
        changed.loc[changed["validation_season"] == 2024, "control_success"] = [1, 0, 1, 0]
        first = rolling_refinement(frame, 0.1, 100.0, 0.25, 0.6)
        second = rolling_refinement(changed, 0.1, 100.0, 0.25, 0.6)
        mask = first["validation_season"] == 2024
        np.testing.assert_allclose(
            first.loc[mask, ["game_type_corrected", "rolling_calibrated"]],
            second.loc[mask, ["game_type_corrected", "rolling_calibrated"]],
        )


if __name__ == "__main__":
    unittest.main()
