import unittest

from attnlens.train import train


class TestTrain(unittest.TestCase):
    def test_deterministic_same_seed(self):
        r1 = train(dim=4, heads=2, epochs=3, per_epoch=4, n_eval=8, seed=11)
        r2 = train(dim=4, heads=2, epochs=3, per_epoch=4, n_eval=8, seed=11)
        self.assertEqual(r1["losses"], r2["losses"])
        self.assertEqual(r1["accuracy"], r2["accuracy"])

    def test_different_seed_differs(self):
        r1 = train(dim=4, heads=2, epochs=3, per_epoch=4, n_eval=8, seed=11)
        r2 = train(dim=4, heads=2, epochs=3, per_epoch=4, n_eval=8, seed=12)
        self.assertNotEqual(r1["losses"], r2["losses"])

    def test_validation(self):
        for kwargs in ({"lr": 0}, {"epochs": 0}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                train(dim=4, heads=2, **kwargs)

    def test_loss_decreases(self):
        r = train(dim=8, heads=2, epochs=30, per_epoch=32, n_eval=16, seed=7)
        self.assertLess(r["losses"][-1], r["losses"][0])


class TestTrainedQuality(unittest.TestCase):
    """Pinned quality: 100 seeded epochs solve the recall task and the top
    layer attends to the key token (the induction pattern)."""

    @classmethod
    def setUpClass(cls):
        cls.result = train(dim=8, heads=2, epochs=100, per_epoch=64,
                           n_eval=64, seed=7, lr=0.1)

    def test_accuracy(self):
        self.assertGreaterEqual(self.result["accuracy"], 0.95)

    def test_final_loss(self):
        self.assertLess(self.result["final_loss"], 0.05)

    def test_recall_attention(self):
        self.assertGreaterEqual(self.result["mean_recall_attention"], 0.7)


if __name__ == "__main__":
    unittest.main()
