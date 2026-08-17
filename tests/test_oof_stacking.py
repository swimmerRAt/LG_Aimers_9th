from __future__ import annotations

import unittest

import numpy as np

from model.oof_stacking import (
    constrained_stack_weights,
    error_correlation_long,
    select_simplex_stack_weights,
)


class OOFStackingTest(unittest.TestCase):
    def test_constrained_weights_keep_official_model_and_sum_to_one(self):
        weights = constrained_stack_weights(0.65, 0.4, minimum_official_weight=0.5)
        np.testing.assert_allclose(weights, [0.65, 0.14, 0.21])
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreaterEqual(weights[0], 0.5)

    def test_constrained_weights_reject_official_weight_below_minimum(self):
        with self.assertRaises(ValueError):
            constrained_stack_weights(0.49, 0.5, minimum_official_weight=0.5)

    def test_simplex_optimizer_selects_perfect_branch(self):
        truth = np.asarray([0.0, 1.0, 0.0, 1.0] * 2)
        seasons = np.asarray([2022] * 4 + [2023] * 4)
        baseline = np.full(len(truth), 0.5)
        perfect = truth.copy()
        noisy = np.asarray([0.8, 0.2, 0.8, 0.2] * 2)
        matrix = np.column_stack([baseline, perfect, noisy])
        weights, diagnostics = select_simplex_stack_weights(
            truth, matrix, seasons, final_logit_shift=0.0
        )
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreater(weights[1], 0.999)
        self.assertTrue(diagnostics["success"])

    def test_error_correlations_include_both_error_definitions(self):
        truth = np.asarray([0.0, 1.0, 0.0, 1.0])
        result = error_correlation_long(
            truth,
            {
                "a": np.asarray([0.1, 0.9, 0.2, 0.8]),
                "b": np.asarray([0.2, 0.8, 0.3, 0.7]),
            },
        )
        diagonal = result[
            result["model_left"].eq(result["model_right"])
        ]
        np.testing.assert_allclose(diagonal["signed_error_correlation"], 1.0)
        np.testing.assert_allclose(diagonal["squared_error_correlation"], 1.0)


if __name__ == "__main__":
    unittest.main()
