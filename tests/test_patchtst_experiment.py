from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiment_patchtst import build_monthly_panel, sliding_windows


class PatchTSTExperimentTest(unittest.TestCase):
    def test_builds_regular_three_channel_panel(self):
        frame = pd.DataFrame(
            {
                "season": [2021, 2021, 2021, 2021],
                "game_month": [3, 3, 4, 4],
                "game_type": ["F", "R", "F", "R"],
                "control_success": [1, 0, 1, 1],
            }
        )
        panel, grid, priors = build_monthly_panel(
            frame, minimum_season=2021, maximum_season=2021
        )
        self.assertEqual(panel.shape, (8, 3))
        self.assertEqual(len(grid), 8)
        self.assertEqual(set(priors), {"overall", "game_type_F", "game_type_R"})
        self.assertTrue(np.isfinite(panel).all())

    def test_sliding_window_shapes(self):
        panel = np.random.default_rng(7).random((24, 3))
        inputs, targets = sliding_windows(panel, 12, 8)
        self.assertEqual(inputs.shape, (5, 3, 12))
        self.assertEqual(targets.shape, (5, 3, 8))

    def test_rejects_short_panel(self):
        with self.assertRaises(ValueError):
            sliding_windows(np.ones((19, 3)), 12, 8)


if __name__ == "__main__":
    unittest.main()
