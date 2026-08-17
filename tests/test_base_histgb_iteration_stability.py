from __future__ import annotations

import unittest

import numpy as np

from experiment_base_histgb_iteration_stability import (
    validated_iteration_candidates,
)


class BaseHistGBIterationStabilityTest(unittest.TestCase):
    def test_candidates_are_sorted_and_include_official_baseline(self):
        result = validated_iteration_candidates([350, 300, 100])
        np.testing.assert_array_equal(result, [100, 300, 350])

    def test_candidates_require_official_300_iteration_baseline(self):
        with self.assertRaises(ValueError):
            validated_iteration_candidates([100, 200])


if __name__ == "__main__":
    unittest.main()
