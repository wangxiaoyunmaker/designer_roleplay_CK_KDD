import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


class InferenceCliTests(unittest.TestCase):
    def test_help_does_not_require_gpu_runtime(self):
        result = subprocess.run(
            [sys.executable, "-B", "-m", "route_a.inference", "--help"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("prepare-q7", result.stdout)
        self.assertIn("generate-q7", result.stdout)
        self.assertIn("merge-q7-folds", result.stdout)
        self.assertIn("export-q7-teacher-forced", result.stdout)
        self.assertIn("merge-teacher-forced", result.stdout)
        self.assertIn("calibrate-best-of-n", result.stdout)
        self.assertIn("merge-q7-seeds", result.stdout)


if __name__ == "__main__":
    unittest.main()
