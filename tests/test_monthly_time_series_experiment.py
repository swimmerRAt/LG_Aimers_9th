from __future__ import annotations

import unittest

import numpy as np

from experiment_monthly_time_series import add_time_series_offset, configuration_grid


class MonthlyTimeSeriesExperimentTest(unittest.TestCase):
    def test_zero_strength_preserves_base_probability(self):
        base = np.asarray([0.2, 0.5, 0.8])
        result = add_time_series_offset(base, np.asarray([-1.0, 0.0, 1.0]), 0.0)
        np.testing.assert_allclose(result, base)

    def test_positive_strength_moves_probability_in_offset_direction(self):
        base = np.asarray([0.5, 0.5])
        result = add_time_series_offset(base, np.asarray([-0.5, 0.5]), 0.2)
        self.assertLess(result[0], 0.5)
        self.assertGreater(result[1], 0.5)

    def test_configuration_grid_is_unique(self):
        grid = configuration_grid()
        keys = {
            (row["harmonic_order"], row["ridge"], row["recency_decay_per_year"])
            for row in grid
        }
        self.assertEqual(len(grid), 18)
        self.assertEqual(len(keys), len(grid))


if __name__ == "__main__":
    unittest.main()
