import unittest

from attnlens.data import MARKER, QUERY
from attnlens.patterns import analyze, head_stats, heatmap
from attnlens.model import Cache, Params, forward
import random


class TestHeadStats(unittest.TestCase):
    def test_uniform_attention_has_high_entropy(self):
        attn = [[[0.25] * 4] * 4]
        stats = head_stats(attn, [1, 2, 3, 4])
        import math
        self.assertAlmostEqual(stats[0]["mean_entropy"], math.log(4), places=4)

    def test_peaked_attention_has_low_entropy(self):
        attn = [[[1.0, 0.0, 0.0, 0.0]] * 4]
        stats = head_stats(attn, [1, 2, 3, 4])
        self.assertEqual(stats[0]["mean_entropy"], 0.0)

    def test_prev_token_mass(self):
        attn = [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0]]
        stats = head_stats([attn], [1, 2, 3])
        self.assertAlmostEqual(stats[0]["prev_token_mass"], 0.75)

    def test_recall_attention_reported_with_marker(self):
        attn = [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.1, 0.8, 0.1]]
        tokens = [MARKER, 5, QUERY]
        stats = head_stats([attn], tokens)
        self.assertAlmostEqual(stats[0]["recall_attention"], 0.8)

    def test_no_marker_no_recall_key(self):
        attn = [[1.0, 0.0], [0.5, 0.5]]
        stats = head_stats([attn], [1, 2])
        self.assertNotIn("recall_attention", stats[0])


class TestAnalyze(unittest.TestCase):
    def test_layer_structure(self):
        p = Params(vocab=12, dim=4, heads=2, layers=2, rng=random.Random(1))
        result = analyze([1, 2, 3, QUERY], p, layers=2)
        self.assertEqual(len(result["layers"]), 2)
        self.assertTrue(all(len(layer) == 2 for layer in result["layers"]))

    def test_heatmap_shape(self):
        bars = heatmap([0.5, 0.5], width=4)
        self.assertIn("0.50", bars)
        self.assertIn("██", bars)


if __name__ == "__main__":
    unittest.main()
