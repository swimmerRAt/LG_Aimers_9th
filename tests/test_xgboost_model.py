from __future__ import annotations

import unittest
import subprocess
import sys

import numpy as np
import pandas as pd

from model.xgboost_model import XGBoostProbabilityModel
from script import render_feature_importance_svg
from train_xgboost import optimal_xgboost_blend_weight


class XGBoostConfigurationTest(unittest.TestCase):
    def test_binary_probability_objective_and_brier_equivalent_metric(self):
        params = XGBoostProbabilityModel()._make_estimator().get_params()
        self.assertEqual(params["objective"], "binary:logistic")
        self.assertEqual(params["eval_metric"], "rmse")
        self.assertEqual(params["tree_method"], "hist")
        self.assertIsNone(params["scale_pos_weight"])

    def test_fit_predict_handles_unknown_category(self):
        # Exercise native XGBoost training in an isolated process so a native
        # runtime failure cannot terminate the main test runner.
        code = r'''
import numpy as np
import pandas as pd
from model.xgboost_model import XGBoostProbabilityModel

train = pd.DataFrame({
    "top_bottom": ["T", "B", "T", "B", "T", "B"],
    "game_type": ["R"] * 6,
    "base_state": ["___", "1__", "___", "1__", "___", "1__"],
    "season": [2023] * 6,
    "li": [0.1, 0.2, np.nan, 0.4, 0.5, 0.6],
})
y = np.array([0, 1, 0, 1, 0, 1])
model = XGBoostProbabilityModel(
    n_estimators=2,
    max_depth=2,
    min_child_weight=1,
    early_stopping_rounds=None,
    n_jobs=1,
).fit(train, y)
test = train.iloc[[0]].copy()
test["base_state"] = "OOV"
prediction = model.predict_proba(test)
assert prediction.shape == (1, 2)
assert np.isfinite(prediction).all()
assert abs(float(prediction.sum()) - 1.0) < 1e-7
'''
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)


class XGBoostBlendTest(unittest.TestCase):
    def test_optimal_weight_is_clipped_to_probability_blend_range(self):
        truth = np.array([0.0, 1.0, 1.0])
        baseline = np.array([0.4, 0.6, 0.6])
        candidate = np.array([0.2, 0.8, 0.8])
        self.assertEqual(optimal_xgboost_blend_weight(truth, baseline, candidate), 1.0)

    def test_importance_chart_uses_xgboost_label(self):
        frame = pd.DataFrame(
            {
                "feature": ["season"],
                "importance": [1.0],
                "importance_percent": [100.0],
                "source_component": ["XGBoost gain"],
            }
        )
        svg = render_feature_importance_svg(frame)
        self.assertIn("XGBoost Feature Importance", svg)


if __name__ == "__main__":
    unittest.main()
