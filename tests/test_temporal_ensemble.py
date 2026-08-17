from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from model.ensemble import RateSmoothingFeatureBuilder
from model.temporal_ensemble import TemporalWindowEnsemble
from train_temporal_ensemble import fixed_blend_weights, select_continuous_blend_weights


class TemporalWindowDefinitionTest(unittest.TestCase):
    def test_rate_smoothing_shrinks_small_samples_more(self):
        frame = pd.DataFrame(
            {
                "asof_pitcher_n": [10, 1000],
                "asof_pitcher_success_rate": [0.7, 0.7],
                "asof_batter_n": [0, 0],
                "asof_batter_success_rate": [np.nan, np.nan],
            }
        )
        builder = RateSmoothingFeatureBuilder((50,)).fit(frame, [0, 1])
        result = builder.transform(frame)
        self.assertAlmostEqual(result.loc[0, "pitcher_success_rate_smooth_50"], 0.5333333333)
        self.assertAlmostEqual(result.loc[1, "pitcher_success_rate_smooth_50"], 0.6904761905)
        self.assertEqual(result.loc[0, "batter_success_rate_smooth_50"], 0.5)

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

    def test_histgb_iteration_setting_reaches_component_base_model(self):
        ensemble = TemporalWindowEnsemble(hist_max_iter=175)
        component = ensemble._make_base_model(0)
        self.assertEqual(component.hist_max_iter, 175)


class TemporalBlendSearchTest(unittest.TestCase):
    def test_fixed_weights_are_normalized_and_exclude_time_weighted(self):
        weights, diagnostics = fixed_blend_weights([2, 1, 1], "test")
        np.testing.assert_allclose(weights, [0.5, 0.25, 0.25, 0.0])
        self.assertEqual(diagnostics.loc[0, "strategy"], "fixed_from_prior_temporal_ensemble")

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
