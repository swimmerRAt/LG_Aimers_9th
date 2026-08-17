from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from model.monthly_time_series import MonthlyRateTimeSeries


class MonthlyRateTimeSeriesTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "season": np.repeat([2021, 2022, 2023], 8),
                "game_month": np.tile(np.arange(3, 11), 3),
            }
        )
        rates = 0.60 - 0.03 * (self.frame["season"] - 2021) - 0.002 * self.frame["game_month"]
        self.target = (np.arange(len(self.frame)) / len(self.frame) < rates).astype(int)

    def test_forecasts_finite_probabilities_and_offsets(self):
        model = MonthlyRateTimeSeries(harmonic_order=1, ridge=1.0).fit(
            self.frame, self.target
        )
        future = pd.DataFrame({"season": [2024, 2024], "game_month": [3, 10]})
        probability = model.predict_proba(future)
        offset = model.predict_logit_offset(future)
        self.assertTrue(np.isfinite(probability).all())
        self.assertTrue(((probability > 0.0) & (probability < 1.0)).all())
        self.assertEqual(offset.shape, (2,))

    def test_rejects_invalid_month(self):
        model = MonthlyRateTimeSeries().fit(self.frame, self.target)
        with self.assertRaises(ValueError):
            model.predict_proba(pd.DataFrame({"season": [2024], "game_month": [13]}))

    def test_requires_binary_target(self):
        with self.assertRaises(ValueError):
            MonthlyRateTimeSeries().fit(self.frame, np.full(len(self.frame), 0.5))


if __name__ == "__main__":
    unittest.main()
