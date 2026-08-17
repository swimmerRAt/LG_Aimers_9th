from __future__ import annotations

import unittest

import torch

from model.patchtst_model import PatchTST, parameter_count


class PatchTSTModelTest(unittest.TestCase):
    def test_output_shape_and_patch_count(self):
        model = PatchTST(
            channels=3,
            context_length=12,
            prediction_length=8,
            patch_length=4,
            stride=2,
            dropout=0.0,
        )
        values = torch.rand(5, 3, 12)
        output = model(values)
        self.assertEqual(output.shape, (5, 3, 8))
        self.assertEqual(model.patch_count, 6)
        self.assertGreater(parameter_count(model), 0)

    def test_channel_independent_inference(self):
        torch.manual_seed(7)
        model = PatchTST(
            channels=2,
            context_length=12,
            prediction_length=8,
            patch_length=4,
            stride=2,
            dropout=0.0,
        ).eval()
        source = torch.rand(1, 2, 12)
        changed = source.clone()
        changed[:, 1] += 10.0
        with torch.no_grad():
            first = model(source)[:, 0]
            second = model(changed)[:, 0]
        torch.testing.assert_close(first, second)

    def test_rejects_wrong_input_shape(self):
        model = PatchTST(channels=3, context_length=12, prediction_length=8)
        with self.assertRaises(ValueError):
            model(torch.rand(2, 12, 3))


if __name__ == "__main__":
    unittest.main()
