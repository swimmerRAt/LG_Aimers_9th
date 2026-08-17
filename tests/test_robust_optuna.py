from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from optimize_hyperparameters_robust import (
    assert_tuning_unlocked,
    normalized_weights,
    robust_objective,
    selection_signature,
)


class RobustObjectiveTest(unittest.TestCase):
    def test_weighted_mean_and_stability_penalty(self):
        objective, weighted_mean, weighted_std = robust_objective(
            [0.9, 1.0], [1.0, 3.0], stability_penalty=0.25
        )
        self.assertAlmostEqual(weighted_mean, 0.975)
        self.assertAlmostEqual(weighted_std, 0.04330127018922193)
        self.assertAlmostEqual(objective, 0.9858253175473055)

    def test_weights_are_normalized(self):
        self.assertEqual(
            normalized_weights([15, 30, 55], [2021, 2022, 2023]),
            [0.15, 0.3, 0.55],
        )

    def test_weight_count_must_match_folds(self):
        with self.assertRaises(ValueError):
            normalized_weights([1.0], [2021, 2022])


class OuterLockTest(unittest.TestCase):
    def test_outer_lock_blocks_additional_tuning(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            assert_tuning_unlocked(artifact_dir)
            (artifact_dir / "outer_lock.json").write_text(
                json.dumps({"locked": True}), encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                assert_tuning_unlocked(artifact_dir)

    def test_selection_signature_changes_with_parameters(self):
        selection = {
            "config": {"inner_seasons": [2021, 2022, 2023]},
            "feature_columns": ["season", "inning"],
            "histgb_params": {"max_iter": 300},
            "extra_trees_params": {"max_depth": 16},
            "hist_weight": 0.45,
        }
        first = selection_signature(selection)
        selection["hist_weight"] = 0.50
        self.assertNotEqual(first, selection_signature(selection))


if __name__ == "__main__":
    unittest.main()
