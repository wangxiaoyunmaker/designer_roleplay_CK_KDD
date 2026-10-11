"""Route A data preparation and training-record utilities."""

from .errors import DataContractError
from .build import build_dataset, inspect_dataset, validate_dataset

__all__ = [
    "DataContractError",
    "build_dataset",
    "inspect_dataset",
    "validate_dataset",
]
