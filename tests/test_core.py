from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from script import build_submission, render_feature_importance_svg
from src.lg_aimers.metrics import brier_score, competition_score
from src.lg_aimers.validation import make_season_forward_splits


class MetricsTest(unittest.TestCase):
    def test_brier_score(self):
        self.assertAlmostEqual(brier_score([0, 1], [0.25, 0.75]), 0.0625)

    def test_perfect_competition_score(self):
        self.assertEqual(competition_score([0, 1], [0, 1]), 100000.0)

    def test_invalid_probability_fails(self):
        with self.assertRaises(ValueError):
            brier_score([0, 1], [0.2, 1.1])


class ForwardSplitTest(unittest.TestCase):
    def test_only_past_seasons_enter_training(self):
        frame = pd.DataFrame({"season": [2019, 2020, 2021, 2022]})
        split = make_season_forward_splits(frame, [2022])[0]
        self.assertEqual(split.train_index.tolist(), [0, 1, 2])
        self.assertEqual(split.validation_index.tolist(), [3])


class SubmissionTest(unittest.TestCase):
    def test_aligns_predictions_to_sample_order(self):
        test = pd.DataFrame({"row_id": ["b", "a"], "x": [1, 2]})
        sample = pd.DataFrame({"row_id": ["a", "b"], "control_success": [0.5, 0.5]})
        result = build_submission(test, sample, np.array([0.2, 0.8]))
        self.assertEqual(result["control_success"].tolist(), [0.8, 0.2])

    def test_missing_id_fails_instead_of_using_placeholder(self):
        test = pd.DataFrame({"row_id": ["a"]})
        sample = pd.DataFrame({"row_id": ["a", "b"], "control_success": [0.5, 0.5]})
        with self.assertRaises(ValueError):
            build_submission(test, sample, np.array([0.2]))

    def test_feature_importance_svg_has_zero_based_ranked_bars(self):
        frame = pd.DataFrame({
            "feature": ["long_feature_name", "other"],
            "importance": [0.7, 0.3],
            "importance_percent": [70.0, 30.0],
            "source_component": ["ExtraTrees impurity importance"] * 2,
        })
        svg = render_feature_importance_svg(frame, top_n=2)
        self.assertIn("ExtraTrees Feature Importance", svg)
        self.assertIn("long_feature_name", svg)
        self.assertEqual(svg.count("<rect "), 3)  # background plus two bars


if __name__ == "__main__":
    unittest.main()
