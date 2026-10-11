"""Route A V10 training interfaces.

The configuration and dry-run layer use only the Python standard library. GPU
components import PyTorch/Transformers/PEFT lazily so data validation remains
available on machines without the training runtime.
"""

from .config import TrainingConfig, TrainingConfigError, load_training_config
from .prepare import PreparedFold, PrepareError, prepare_fold

__all__ = [
    "PreparedFold",
    "PrepareError",
    "TrainingConfig",
    "TrainingConfigError",
    "load_training_config",
    "prepare_fold",
]
