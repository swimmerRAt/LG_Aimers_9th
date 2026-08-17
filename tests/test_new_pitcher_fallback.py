from __future__ import annotations

import unittest

import numpy as np

from experiment_new_pitcher_fallback import (
    apply_logit_shift,
    fallback_feature_columns,
    gate_weight,
)


class NewPitcherFallbackTest(unittest.TestCase):
    def test_fallback_removes_pitcher_rates_but_keeps_count(self):
        columns = [
            "season",
            "asof_pitcher_n",
            "asof_pitcher_success_rate",
            "asof_pitcher_prev1_game_success_rate",
            "asof_batter_success_rate",
        ]
        self.assertEqual(
            fallback_feature_columns(columns),
            ["season", "asof_pitcher_n", "asof_batter_success_rate"],
        )

    def test_new_only_gate_leaves_existing_pitcher_unchanged(self):
        result = gate_weight([0, 100], [True, False], "new_only", 50.0)
        np.testing.assert_allclose(result, [0.0, 1.0])

    def test_more_history_increases_main_model_weight(self):
        result = gate_weight([0, 10, 100], [True, True, True], "new_only", 50.0)
        self.assertTrue(np.all(np.diff(result) > 0.0))

    def test_fixed_negative_shift_reduces_all_probabilities(self):
        probability = np.array([0.2, 0.5, 0.8])
        shifted = apply_logit_shift(probability, -0.0461)
        self.assertTrue((shifted < probability).all())


if __name__ == "__main__":
    unittest.main()
