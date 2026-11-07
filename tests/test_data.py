import random
import unittest

from attnlens.data import MARKER, QUERY, VOCAB, make_dataset, make_sample


class TestData(unittest.TestCase):
    def test_sample_structure(self):
        rng = random.Random(1)
        tokens, key = make_sample(rng, length=8)
        self.assertEqual(len(tokens), 8)
        self.assertEqual(tokens[-1], QUERY)
        m = tokens.index(MARKER)
        self.assertEqual(tokens[m + 1], key)

    def test_marker_position_varies(self):
        positions = {make_sample(random.Random(s))[0].index(MARKER) for s in range(20)}
        self.assertGreater(len(positions), 3)

    def test_deterministic(self):
        self.assertEqual(make_dataset(5, seed=1), make_dataset(5, seed=1))

    def test_different_seed_differs(self):
        self.assertNotEqual(make_dataset(5, seed=1), make_dataset(5, seed=2))

    def test_validation(self):
        with self.assertRaises(ValueError):
            make_sample(random.Random(1), length=3)
        with self.assertRaises(ValueError):
            make_dataset(0, seed=1)

    def test_vocab_constants(self):
        self.assertEqual(VOCAB, 12)
        self.assertNotIn(MARKER, range(10))
        self.assertNotIn(QUERY, range(10))


if __name__ == "__main__":
    unittest.main()
