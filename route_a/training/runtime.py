"""Lazy GPU runtime loading for Qwen3/PEFT training."""

from __future__ import annotations

import importlib
import importlib.util
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import TrainingConfig


REQUIRED_PACKAGES = ("torch", "transformers", "peft", "accelerate", "safetensors")


class TrainingRuntimeError(RuntimeError):
    """Raised when the optional GPU training runtime is unavailable."""


@dataclass(frozen=True)
class RuntimeStatus:
    packages: dict[str, bool]
    cuda_available: bool | None
    cuda_device_count: int | None

    @property
    def ready(self) -> bool:
        return all(self.packages.values()) and self.cuda_available is True


def runtime_status() -> RuntimeStatus:
    packages = {name: importlib.util.find_spec(name) is not None for name in REQUIRED_PACKAGES}
    if not packages["torch"]:
        return RuntimeStatus(packages, None, None)
    torch = importlib.import_module("torch")
    return RuntimeStatus(packages, bool(torch.cuda.is_available()), int(torch.cuda.device_count()))


def require_training_runtime(*, require_cuda: bool = True) -> dict[str, Any]:
    status = runtime_status()
    missing = sorted(name for name, present in status.packages.items() if not present)
    if missing:
        raise TrainingRuntimeError(
            "missing training packages: " + ", ".join(missing)
            + "; install route_a/training/requirements-training.txt on the GPU server"
        )
    modules = {name: importlib.import_module(name) for name in REQUIRED_PACKAGES}
    if require_cuda and not modules["torch"].cuda.is_available():
        raise TrainingRuntimeError("CUDA is required for Qwen3-8B training")
    return modules


def directory_sha256(directory: Any) -> str:
    root = Path(directory).resolve()
    if not root.is_dir():
        raise TrainingRuntimeError(f"model snapshot directory does not exist: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise TrainingRuntimeError("directory is empty")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
        content_digest = file_digest.digest()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(content_digest)
    return digest.hexdigest()


def load_tokenizer_and_causal_lm(
    config: TrainingConfig,
    *,
    model_path: Any,
    adapter_name: str | None,
    qlora: bool,
    device_index: int = 0,
    trainable: bool = True,
) -> tuple[Any, Any]:
    modules = require_training_runtime(require_cuda=True)
    torch = modules["torch"]
    transformers = modules["transformers"]
    peft = modules["peft"]
    snapshot = Path(model_path).resolve()
    if not snapshot.is_dir():
        raise TrainingRuntimeError(f"local model snapshot does not exist: {snapshot}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(snapshot),
        use_fast=True,
        trust_remote_code=False,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": device_index},
        "trust_remote_code": False,
    }
    reproducibility = config.raw.get("reproducibility")
    if reproducibility is not None:
        kwargs["attn_implementation"] = reproducibility["attention_implementation"]
    if qlora:
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    kwargs["local_files_only"] = True
    model = transformers.AutoModelForCausalLM.from_pretrained(str(snapshot), **kwargs)
    expected_architecture = {
        "model_type": "qwen3",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 151936,
        "eos_token_id": 151645,
    }
    mismatches = {
        field: (expected, getattr(model.config, field, None))
        for field, expected in expected_architecture.items()
        if getattr(model.config, field, None) != expected
    }
    if mismatches:
        raise TrainingRuntimeError(
            "local model snapshot does not match Qwen/Qwen3-8B architecture: "
            + repr(mismatches)
        )
    model.config.use_cache = False
    if trainable:
        model.gradient_checkpointing_enable()
    for parameter in model.parameters():
        parameter.requires_grad = False

    if adapter_name is not None:
        if qlora:
            model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora = config.raw["lora"]
        peft_config = peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            r=lora["rank"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=list(lora["target_modules"]),
            bias=lora["bias"],
            inference_mode=not trainable,
        )
        model = peft.get_peft_model(model, peft_config, adapter_name=adapter_name)
    return tokenizer, model


def load_causal_lm_from_adapter(
    config: TrainingConfig,
    *,
    model_path: Any,
    adapter_dir: Any,
    adapter_name: str,
    qlora: bool,
    device_index: int = 0,
    trainable: bool = False,
) -> tuple[Any, Any]:
    modules = require_training_runtime(require_cuda=True)
    tokenizer, base = load_tokenizer_and_causal_lm(
        config,
        model_path=model_path,
        adapter_name=None,
        qlora=qlora,
        device_index=device_index,
        trainable=trainable,
    )
    if qlora and trainable:
        base = modules["peft"].prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=True
        )
    model = modules["peft"].PeftModel.from_pretrained(
        base,
        str(adapter_dir),
        adapter_name=adapter_name,
        is_trainable=trainable,
    )
    if trainable:
        model.train()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.eval()
        model.config.use_cache = True
    return tokenizer, model
