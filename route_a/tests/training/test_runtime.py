import unittest

from route_a.training.models import load_torch_model_components
from route_a.training.runtime import TrainingRuntimeError, runtime_status


class RuntimeIsolationTests(unittest.TestCase):
    def test_status_can_be_read_without_importing_gpu_stack(self):
        status = runtime_status()
        self.assertIn("torch", status.packages)
        self.assertIn("transformers", status.packages)

    def test_missing_runtime_fails_with_actionable_message(self):
        status = runtime_status()
        if all(status.packages.values()):
            self.skipTest("GPU training packages are installed")
        with self.assertRaisesRegex(TrainingRuntimeError, "requirements-training.txt"):
            load_torch_model_components()


if __name__ == "__main__":
    unittest.main()
