import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent


def run_cli(*argv):
    return subprocess.run([sys.executable, "-m", "attnlens", *argv],
                          cwd=ROOT, capture_output=True, text=True)


FAST = ["--dim", "4", "--heads", "2", "--layers", "1", "--epochs", "5",
        "--per-epoch", "8", "--n-eval", "8"]


class TestCli(unittest.TestCase):
    def test_train_runs(self):
        proc = run_cli("train", *FAST)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("final_loss=", proc.stdout)

    def test_train_gate_fail(self):
        proc = run_cli("train", *FAST, "--min-accuracy", "1.01")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("GATE FAIL", proc.stderr)

    def test_train_gate_pass_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, m = Path(tmp) / "r.json", Path(tmp) / "r.md"
            proc = run_cli("train", *FAST, "--min-accuracy", "0.0",
                           "--json", str(j), "--markdown", str(m))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            parsed = json.loads(j.read_text())
            self.assertIn("accuracy", parsed)
            self.assertIn("# AttnLens report", m.read_text())

    def test_heatmap_runs(self):
        proc = run_cli("heatmap", "--dim", "4", "--heads", "2", "--layers", "1",
                       "--epochs", "3", "--per-epoch", "8")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("query row recall=", proc.stdout)
        self.assertIn("top head query row:", proc.stdout)

    def test_bad_lr_errors(self):
        proc = run_cli("train", *FAST, "--lr", "0")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("error:", proc.stderr)


if __name__ == "__main__":
    unittest.main()
