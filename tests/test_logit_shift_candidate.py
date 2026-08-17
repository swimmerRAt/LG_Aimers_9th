from __future__ import annotations

import unittest

import numpy as np

from create_logit_shift_candidate import logit


class LogitShiftCandidateTest(unittest.TestCase):
    def test_logit_clips_boundary_probabilities(self):
        result = logit([0.0, 0.5, 1.0])
        self.assertTrue(np.isfinite(result).all())
        self.assertAlmostEqual(result[1], 0.0)

    def test_negative_shift_reduces_probability(self):
        probability = np.array([0.2, 0.5, 0.8])
        shifted = 1.0 / (1.0 + np.exp(-(logit(probability) - 0.02)))
        self.assertTrue((shifted < probability).all())


if __name__ == "__main__":
    unittest.main()
