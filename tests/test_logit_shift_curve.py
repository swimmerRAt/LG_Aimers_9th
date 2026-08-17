from __future__ import annotations

import unittest

from fit_logit_shift_curve import fit_quadratic_vertex


class LogitShiftCurveTest(unittest.TestCase):
    def test_recovers_quadratic_maximum(self):
        result = fit_quadratic_vertex([-1.0, 0.0, 1.0], [1.0, 4.0, 3.0])
        self.assertAlmostEqual(result["optimal_logit_shift"], 0.25)
        self.assertAlmostEqual(result["estimated_optimal_score"], 4.125)

    def test_rejects_convex_score_curve(self):
        with self.assertRaises(ValueError):
            fit_quadratic_vertex([-1.0, 0.0, 1.0], [4.0, 1.0, 4.0])


if __name__ == "__main__":
    unittest.main()
