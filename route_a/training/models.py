"""Lazy access to PyTorch model components."""

from __future__ import annotations

from typing import Any

from .runtime import require_training_runtime


def load_torch_model_components() -> dict[str, Any]:
    """Import actual nn.Module classes only after runtime validation."""
    require_training_runtime(require_cuda=False)
    from . import torch_models

    return {
        "PreferenceEncoder": torch_models.PreferenceEncoder,
        "StatePrototypeBank": torch_models.StatePrototypeBank,
        "AuxiliaryPreferenceModel": torch_models.AuxiliaryPreferenceModel,
        "PlannerModel": torch_models.PlannerModel,
        "RealizerModel": torch_models.RealizerModel,
        "initialize_semantic_vectors": torch_models.initialize_semantic_vectors,
    }
