import json
import unittest

from attnlens.report import build_report, to_json, to_markdown, to_terminal


def sample_report():
    result = {"config": {"dim": 8, "seed": 7}, "final_loss": 0.01,
              "accuracy": 0.98, "mean_recall_attention": 0.85,
              "losses": [2.5, 1.0, 0.1, 0.01]}
    analysis = {"tokens": [10, 5, 11],
                "layers": [[{"head": 0, "mean_entropy": 0.4,
                             "prev_token_mass": 0.2, "recall_attention": 0.9}]]}
    return build_report(result, analysis)


class TestReport(unittest.TestCase):
    def test_json_round_trip(self):
        parsed = json.loads(to_json(sample_report()))
        self.assertEqual(parsed["accuracy"], 0.98)
        self.assertEqual(parsed["head_analysis"]["layers"][0][0]["recall_attention"], 0.9)

    def test_loss_curve_downsampled(self):
        report = sample_report()
        self.assertEqual(report["loss_curve"], [2.5, 1.0, 0.1, 0.01])

    def test_markdown_shape(self):
        md = to_markdown(sample_report())
        self.assertIn("# AttnLens report", md)
        self.assertIn("| Layer | Head | Mean entropy | Prev-token mass | Recall attention |", md)
        self.assertIn("| 0 | 0 | 0.4000 | 0.2000 | 0.9000 |", md)

    def test_terminal_shape(self):
        text = to_terminal(sample_report())
        self.assertIn("accuracy=0.98", text)
        self.assertIn("recall=0.900", text)


if __name__ == "__main__":
    unittest.main()
