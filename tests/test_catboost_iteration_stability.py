from __future__ import annotations

import unittest

import numpy as np

from experiment_catboost_iteration_stability import iteration_grid


class CatBoostIterationStabilityTest(unittest.TestCase):
    def test_iteration_grid_includes_non_multiple_maximum(self):
        np.testing.assert_array_equal(iteration_grid(25, 10), [10, 20, 25])

    def test_iteration_grid_rejects_invalid_range(self):
        with self.assertRaises(ValueError):
            iteration_grid(5, 10)


if __name__ == "__main__":
    unittest.main()
