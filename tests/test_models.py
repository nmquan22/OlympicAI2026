"""Checks invariances and measurable behavior, beyond tensor output shapes."""
import unittest
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from audio.blocks.models import AudioCNN
from common.ensemble import fit_temperature, blend_probabilities
from ml.blocks.behavior import InstructionPolicy
from ml.blocks.tabular import tabular_pipeline
from ml.blocks.timeseries import TemporalCNN
from vision.blocks.fields import FourierField, signed_log, signed_expm1
from vision.blocks.restoration import ResidualRestorer, GuidedSR, tiled_restore
from vision.blocks.robustness import selective_perturbation, selective_success
from vision.blocks.self_supervised import nt_xent, update_ema
from vision_ml.blocks.multimodal import MaskedFusion, ridge_prototypes


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_padding_invariance_audio_and_temporal(self):
        audio, temporal = AudioCNN(3).eval(), TemporalCNN(4, 3).eval()
        spectrogram = torch.randn(1, 64, 19)
        sequence = torch.randn(1, 19, 4)
        lengths = torch.tensor([19])
        torch.testing.assert_close(audio(spectrogram, lengths), audio(F.pad(spectrogram, (0, 11)), lengths))
        torch.testing.assert_close(temporal(sequence, lengths), temporal(F.pad(sequence, (0, 0, 0, 11)), lengths))

    def test_missing_modality_nan_and_ridge_support(self):
        fusion = MaskedFusion([4, 5], 6, 2)
        first, second = torch.randn(3, 4), torch.randn(3, 5)
        available = torch.tensor([[True, False], [True, True], [False, True]])
        prediction = fusion([first, second], available)
        second[0] = float('nan')
        first[2] = float('nan')
        torch.testing.assert_close(prediction, fusion([first, second], available))
        supports = torch.eye(3)
        logits = ridge_prototypes(supports, torch.arange(3), supports, 3)
        self.assertEqual(logits.argmax(1).tolist(), [0, 1, 2])

    def test_tiling_matches_full_for_sufficient_halo(self):
        model = ResidualRestorer(width=8, depth=2).eval()
        image = torch.randn(1, 3, 31, 43)
        torch.testing.assert_close(model(image), tiled_restore(model, image, tile=12, halo=4), atol=1e-6, rtol=1e-5)

    def test_guided_sr_learns_both_branches(self):
        model = GuidedSR(width=8)
        low, rgb = torch.randn(2, 1, 3, 4), torch.randn(2, 3, 24, 32)
        prediction = model(low, rgb)
        self.assertEqual(tuple(prediction.shape), (2, 1, 24, 32))
        prediction.square().mean().backward()
        self.assertGreater(float(model.rgb[0].weight.grad.norm()), 0)
        self.assertGreater(float(model.thermal[0].weight.grad.norm()), 0)

    def test_selective_perturbation_keeps_one_flips_other(self):
        class Keep(nn.Module):
            def forward(self, x):
                signal = x[:, 0, 0, 0]
                return torch.stack([signal, 1 - signal], dim=1)
        class Fool(nn.Module):
            def forward(self, x):
                signal = x[:, 0, 0, 1]
                return torch.stack([signal, 1 - signal], dim=1)
        image = torch.tensor([[[[0.9, 0.51]]]])
        labels = torch.tensor([0])
        delta = selective_perturbation(Keep(), Fool(), image, labels, epsilon=0.1, step_size=0.02, steps=4)
        self.assertLessEqual(float(delta.abs().max()), 0.100001)
        self.assertTrue(selective_success(Keep(), Fool(), image, labels, delta).item())

    def test_contrastive_views_and_ema_buffers(self):
        features = torch.eye(5)
        self.assertLess(float(nt_xent(features, features)), float(nt_xent(features, features.roll(1, 0))))
        teacher, student = nn.BatchNorm1d(3), nn.BatchNorm1d(3)
        student.running_mean.fill_(2)
        update_ema(teacher, student)
        torch.testing.assert_close(teacher.running_mean, student.running_mean)

    def test_field_parameter_budget_and_transform(self):
        model = FourierField()
        self.assertLessEqual(sum(p.numel() for p in model.parameters()), 20260)
        values = torch.tensor([-1e10, -1., 0., 2026., 1e10])
        torch.testing.assert_close(signed_expm1(signed_log(values)), values, rtol=2e-6, atol=1e-6)

    def test_tabular_unseen_category_and_regression(self):
        data = pd.DataFrame({'n': [0., np.nan, 1., 2.], 'c': ['a', 'a', 'b', 'b']})
        query = pd.DataFrame({'n': [1.5], 'c': ['unseen']})
        model = tabular_pipeline(['n'], ['c']).fit(data, [0, 0, 1, 1])
        self.assertEqual(model.predict_proba(query).shape, (1, 2))
        regression = tabular_pipeline(['n'], ['c'], task='regression').fit(data, [0., 0.2, 1., 2.])
        self.assertTrue(np.isfinite(regression.predict(query)).all())

    def test_temperature_and_blend(self):
        logits = np.array([[5., 0.], [0., 5.], [5., 0.], [0., 5.]])
        temperature = fit_temperature(logits, np.array([0, 1, 1, 0]))
        self.assertGreater(temperature, 1)
        probabilities = np.array([[0.2, 0.8]])
        np.testing.assert_allclose(blend_probabilities([probabilities, probabilities], [1, 3]), probabilities)

    def test_behavior_policy_ignores_mission_padding(self):
        model = InstructionPolicy(vocabulary=20).eval()
        grid = torch.zeros(2, 8, 8, 2, dtype=torch.long)
        ids, directions, carrying = torch.tensor([0, 1]), torch.tensor([0, 2]), torch.zeros(2, 2, dtype=torch.long)
        mission, lengths = torch.tensor([[2, 3, 4], [5, 6, 0]]), torch.tensor([3, 2])
        first = model(grid, ids, directions, carrying, mission, lengths)
        second = model(grid, ids, directions, carrying, F.pad(mission, (0, 4)), lengths)
        torch.testing.assert_close(first, second)


if __name__ == '__main__':
    unittest.main()
