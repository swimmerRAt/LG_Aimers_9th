from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from model.temporal_ensemble import TemporalWindowEnsemble
from train_temporal_ensemble import select_continuous_blend_weights


class TemporalWindowDefinitionTest(unittest.TestCase):
    def test_windows_follow_latest_training_season(self):
        seasons = np.array([2019, 2020, 2021, 2022, 2023, 2024])
        masks = TemporalWindowEnsemble.component_masks(seasons, latest_season=2024)
        self.assertEqual(seasons[masks["full"]].tolist(), seasons.tolist())
        self.assertEqual(seasons[masks["recent_3"]].tolist(), [2022, 2023, 2024])
        self.assertEqual(seasons[masks["recent_2"]].tolist(), [2023, 2024])

    def test_temporal_weights_decay_into_the_past(self):
        weights = TemporalWindowEnsemble.temporal_sample_weight(
            [2019, 2020, 2024], latest_season=2024, decay=0.8
        )
        np.testing.assert_allclose(weights, [0.8**5, 0.8**4, 1.0])

    def test_component_weights_are_normalized(self):
        weights = TemporalWindowEnsemble._validated_component_weights([1, 2, 3, 4])
        np.testing.assert_allclose(weights, [0.1, 0.2, 0.3, 0.4])


class TemporalBlendSearchTest(unittest.TestCase):
    def test_continuous_optimizer_excludes_time_weighted_component(self):
        oof = pd.DataFrame(
            {
                "validation_season": [2022, 2022, 2023, 2023],
                "control_success": [0.0, 1.0, 0.0, 1.0],
                "full": [0.0, 1.0, 0.0, 1.0],
                "recent_3": [0.4, 0.6, 0.4, 0.6],
                "recent_2": [0.3, 0.7, 0.3, 0.7],
            }
        )
        weights, diagnostics = select_continuous_blend_weights(
            oof,
            seasons=(2022, 2023),
            season_weights=(0.4, 0.6),
            stability_penalty=0.1,
        )
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertAlmostEqual(weights[0], 1.0, places=6)
        self.assertEqual(weights[3], 0.0)
        self.assertEqual(diagnostics.loc[0, "optimizer"], "SLSQP")


if __name__ == "__main__":
    unittest.main()
