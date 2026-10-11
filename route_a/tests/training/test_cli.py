import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


class TrainingCliTests(unittest.TestCase):
    def test_help_does_not_require_gpu_packages(self):
        result = subprocess.run(
            [sys.executable, "-B", "-m", "route_a.training", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("train-planner", result.stdout)
        self.assertIn("train-realizer", result.stdout)


if __name__ == "__main__":
    unittest.main()
