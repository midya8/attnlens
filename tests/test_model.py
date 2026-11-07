import math
import random
import unittest

from attnlens.model import (AttentionLayer, Params, cross_entropy, forward,
                            loss_and_grads, positional_encoding, predict,
                            sgd_step, softmax)


class TestSoftmax(unittest.TestCase):
    def test_sums_to_one(self):
        probs = softmax([1.0, -2.0, 3.5, 0.0])
        self.assertAlmostEqual(sum(probs), 1.0)

    def test_monotonic(self):
        probs = softmax([0.0, 1.0, 2.0])
        self.assertLess(probs[0], probs[1])
        self.assertLess(probs[1], probs[2])

    def test_large_values_stable(self):
        probs = softmax([1000.0, 1000.0])
        self.assertAlmostEqual(probs[0], 0.5)


class TestPositionalEncoding(unittest.TestCase):
    def test_shape(self):
        pos = positional_encoding(5, 4)
        self.assertEqual(len(pos), 5)
        self.assertTrue(all(len(row) == 4 for row in pos))

    def test_first_position_is_sin0_cos0(self):
        pos = positional_encoding(3, 4)
        self.assertEqual(pos[0], [0.0, 1.0, 0.0, 1.0])

    def test_positions_differ(self):
        pos = positional_encoding(4, 8)
        self.assertNotEqual(pos[1], pos[2])


class TestForward(unittest.TestCase):
    def setUp(self):
        self.p = Params(vocab=6, dim=4, heads=2, layers=2, rng=random.Random(3))

    def test_logits_shape(self):
        logits = forward([1, 2, 3], self.p)
        self.assertEqual(len(logits), 3)
        self.assertTrue(all(len(row) == 6 for row in logits))

    def test_deterministic(self):
        self.assertEqual(forward([1, 2, 3], self.p), forward([1, 2, 3], self.p))

    def test_causal_masking(self):
        # changing a future token must not change earlier logits
        a = forward([1, 2, 3, 4], self.p)
        b = forward([1, 2, 3, 0], self.p)
        for i in range(3):
            self.assertEqual(a[i], b[i])

    def test_attention_rows_sum_to_one(self):
        from attnlens.model import Cache
        cache = Cache()
        forward([1, 2, 3, 4], self.p, cache)
        for lcache in cache.layer_caches:
            for attn in lcache["attn"]:
                for row in attn:
                    self.assertAlmostEqual(sum(row), 1.0)

    def test_residual_stack(self):
        # with two layers the cache records both layer inputs
        from attnlens.model import Cache
        cache = Cache()
        forward([0, 1], self.p, cache)
        self.assertEqual(len(cache.layer_caches), 2)
        self.assertEqual(len(cache.layer_inputs), 2)

    def test_dim_heads_validation(self):
        with self.assertRaises(ValueError):
            Params(vocab=6, dim=5, heads=2, layers=1, rng=random.Random(1))
        with self.assertRaises(ValueError):
            Params(vocab=6, dim=4, heads=2, layers=0, rng=random.Random(1))

    def test_predict_argmax(self):
        self.assertEqual(predict([0.1, 0.9, 0.2]), 1)

    def test_cross_entropy_perfect_is_small(self):
        self.assertLess(cross_entropy([0.0, 10.0], 1), 0.001)


class TestGradients(unittest.TestCase):
    """Hand-derived gradients against finite differences, every group."""

    def test_finite_difference_all_groups(self):
        rng = random.Random(3)
        p = Params(vocab=6, dim=4, heads=2, layers=2, rng=rng)
        tokens, target, eps = [1, 4, 2, 5, 3], 2, 1e-5
        _, grads = loss_and_grads(tokens, target, p)
        for name, group in p.groups().items():
            g = grads[name]
            for i in range(len(group)):
                for j in range(len(group[i])):
                    orig = group[i][j]
                    group[i][j] = orig + eps
                    lp, _ = loss_and_grads(tokens, target, p)
                    group[i][j] = orig - eps
                    lm, _ = loss_and_grads(tokens, target, p)
                    group[i][j] = orig
                    fd = (lp - lm) / (2 * eps)
                    self.assertAlmostEqual(fd, g[i][j], places=6,
                                           msg=f"{name}[{i}][{j}]")

    def test_sgd_step_decreases_loss(self):
        p = Params(vocab=6, dim=4, heads=2, layers=1, rng=random.Random(5))
        tokens, target = [1, 4, 2, 5], 3
        before, grads = loss_and_grads(tokens, target, p)
        for _ in range(10):
            sgd_step(p, grads, lr=0.05)
            _, grads = loss_and_grads(tokens, target, p)
        after, _ = loss_and_grads(tokens, target, p)
        self.assertLess(after, before)

    def test_gradient_clipping_scales_down(self):
        p = Params(vocab=6, dim=4, heads=2, layers=1, rng=random.Random(5))
        tokens, target = [1, 4, 2, 5], 3
        _, grads = loss_and_grads(tokens, target, p)
        huge = {name: [[x * 1000 for x in row] for row in g]
                for name, g in grads.items()}
        before = [row[:] for row in p.wout]
        sgd_step(p, huge, lr=1.0, max_norm=1.0)
        moved = math.sqrt(sum((p.wout[i][j] - before[i][j]) ** 2
                              for i in range(len(before)) for j in range(len(before[0]))))
        self.assertLess(moved, 1.0)  # clipped: total movement norm <= max_norm


if __name__ == "__main__":
    unittest.main()
