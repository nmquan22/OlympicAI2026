"""Behavioral checks for metric/offset, leakage and decision-making mistakes."""
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile
import torch

from common.submission import order_to_ranks, write_predictions
from ml.blocks.active_search import ComparisonSearch
from ml.blocks.graphs import GCN
from ml.blocks.rl import q_update, dqn_loss, QNetwork, reinforce_loss
from ml.blocks.timeseries import forecast_windows, seasonal_naive
from ml.blocks.validation import grouped_holdout, chronological_holdout
from nlp.blocks.changepoint import sentence_spans, locate_boundary, boundary_score
from nlp.blocks.ordering import beam_order, pairwise_order_accuracy
from nlp.blocks.retrieval import SparseRetriever, reciprocal_rank_fusion, ndcg_at_k
from nlp.blocks.structured import best_span, span_char_offsets
from audio.blocks.features import read_wav, log_mel
from audio.blocks.models import ctc_greedy, edit_distance
from audio.blocks.temporal import best_interval, temporal_iou


class ReasoningAndDataTests(unittest.TestCase):
    def test_beam_matches_exact_search_on_small_graph(self):
        scores = np.random.default_rng(7).normal(size=(5, 5))
        prefix = (3, 1)
        choices = [prefix + tail for tail in itertools.permutations([0, 2, 4])]
        objective = lambda order: sum(scores[i, j] for i, j in zip(order, order[1:]))
        expected = max(choices, key=objective)
        actual = beam_order(scores, prefix, width=100)
        self.assertEqual(tuple(actual), expected)
        self.assertEqual(order_to_ranks([2, 0, 3, 1]), [1, 3, 0, 2])
        self.assertEqual(pairwise_order_accuracy([1, 3, 0, 2], [1, 3, 0, 2]), 1)
        self.assertEqual(pairwise_order_accuracy([3, 2, 1, 0], [0, 1, 2, 3]), 0)
        self.assertEqual(pairwise_order_accuracy([0, 0, 1, 2], [0, 1, 2, 3]), 0)

    def test_original_unicode_offsets_and_single_change(self):
        text = 'Quân đi về.  Mưa rơi! Đây là phần máy. Cuối cùng.'
        spans = sentence_spans(text)
        boundary = text.index('Đây')
        self.assertEqual(locate_boundary([0.02, 0.1, 0.9, 0.97], [a for a, _ in spans]), boundary)
        self.assertEqual(''.join(text[a:b] for a, b in spans), text)
        self.assertAlmostEqual(boundary_score([boundary + 100], [boundary]), np.exp(-1))

    def test_no_future_in_forecasting_or_group_split(self):
        values = np.arange(40, dtype=float)
        x, y, origins = forecast_windows(values, 6, 3)
        changed = values.copy()
        changed[20:] = -100
        x2, _, _ = forecast_windows(changed, 6, 3)
        np.testing.assert_array_equal(x[origins <= 20], x2[origins <= 20])
        np.testing.assert_array_equal(y[0, :, 0], [6, 7, 8])
        np.testing.assert_array_equal(seasonal_naive(x[:1], 5, 2)[0, :, 0], [4, 5, 4, 5, 4])
        train, valid = grouped_holdout(np.repeat(np.arange(10), 4))
        groups = np.repeat(np.arange(10), 4)
        self.assertFalse(set(groups[train]) & set(groups[valid]))
        train, valid = chronological_holdout(origins, cutoff=25, gap=3)
        self.assertLess(origins[train].max(), 22)
        self.assertGreaterEqual(origins[valid].min(), 25)

    def test_ctc_blank_does_not_merge_distinct_repeats(self):
        logits = torch.nn.functional.one_hot(torch.tensor([[1, 1, 0, 1, 2, 2, 0]]), 3).float()
        self.assertEqual(ctc_greedy(logits, torch.tensor([7])), [[1, 1, 2]])
        self.assertEqual(edit_distance('the red cat'.split(), 'the cat'.split()), 1)

    def test_audio_rate_and_silence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'stereo.wav'
            tone = (np.sin(np.arange(8000) * 2 * np.pi * 440 / 8000) * 20000).astype(np.int16)
            wavfile.write(path, 8000, np.stack([tone, tone], axis=1))
            waveform = read_wav(path)
        self.assertEqual(len(waveform), 16000)
        self.assertLess(float(waveform.abs().max()), 1)
        self.assertTrue(torch.isfinite(log_mel(torch.zeros(80))).all())
        self.assertEqual(log_mel(waveform).shape[0], 64)

    def test_temporal_units_and_duration_penalty(self):
        prediction = best_interval(np.array([-1, 2, 2, -2]), np.arange(5) * 0.5, max_clips=4)
        self.assertEqual(prediction, (0.5, 1.5))
        self.assertEqual(temporal_iou(prediction, (0.5, 1.5)), 1)
        self.assertEqual(temporal_iou(prediction, (2, 3)), 0)

    def test_rank_fusion_and_retrieval(self):
        self.assertEqual(reciprocal_rank_fusion([[3, 1], [3, 2]])[0], 3)
        self.assertEqual(ndcg_at_k([1, 2], {1: 2, 2: 1}, k=2), 1)
        retrieval = SparseRetriever().fit(['red apple fruit', 'blue ocean water'])
        self.assertEqual(int(retrieval.search(['apple'], 1)[0][0, 0]), 0)

    def test_span_end_and_mask(self):
        start, end = best_span(np.array([100., 1., 3., 2.]), np.array([100., 1., 2., 4.]), np.array([False, True, True, True]), 2)
        self.assertEqual((start, end), (2, 3))
        self.assertEqual(span_char_offsets([(0, 0), (0, 4), (5, 8), (9, 12)], start, end), (5, 12))

    def test_graph_node_permutation_equivariance(self):
        torch.manual_seed(7)
        model = GCN(3, 5, 2).eval()
        adjacency = torch.tensor([[0., 1, 0], [1, 0, 1], [0, 1, 0]])
        features = torch.randn(3, 3)
        permutation = torch.tensor([2, 0, 1])
        torch.testing.assert_close(model(features, adjacency)[permutation], model(features[permutation], adjacency[permutation][:, permutation]))

    def test_terminal_vs_truncation_and_gradient(self):
        terminal, truncated = np.zeros((2, 2)), np.zeros((2, 2))
        terminal[1] = truncated[1] = [4, 2]
        q_update(terminal, 0, 0, 1, 1, True, lr=1, gamma=0.5)
        q_update(truncated, 0, 0, 1, 1, False, lr=1, gamma=0.5)
        self.assertEqual(terminal[0, 0], 1)
        self.assertEqual(truncated[0, 0], 3)
        online, target = QNetwork(3, 2), QNetwork(3, 2)
        loss = dqn_loss(online, target, torch.randn(4, 3), torch.tensor([0, 1, 0, 1]), torch.ones(4), torch.randn(4, 3), torch.tensor([0, 1, 0, 1]))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in target.parameters()))
        log_probability = torch.tensor([-0.5, -0.2], requires_grad=True)
        reinforce_loss(log_probability, [0, 1]).backward()
        self.assertTrue((log_probability.grad < 0).all())

    def test_interactive_search_handles_all_verdicts(self):
        search = ComparisonSearch(np.eye(8))
        for verdict in ['first', 'same', 'second']:
            search.observe(0, 1, verdict)
        proposal = search.propose(0)
        self.assertNotIn(proposal, [0, 1])
        self.assertTrue(np.isfinite(search.log_weights).all())

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                write_predictions(Path(temporary) / 'bad.csv', ['a', 'a'], [0, 1])


if __name__ == '__main__':
    unittest.main()
