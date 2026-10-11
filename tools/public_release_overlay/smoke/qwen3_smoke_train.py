#!/usr/bin/env python3
"""One-step, single-GPU Qwen3-8B BF16 LoRA delivery smoke test.

This is not the Route A research trainer.  It proves that the public data can
flow through the fixed base model, LoRA attachment, forward/backward pass,
optimizer step, and artifact writer without touching existing server checkouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Iterable


EXPECTED_MODEL_ID = "Qwen/Qwen3-8B"
MIN_TRANSFORMERS = "4.51.0"
REQUIRED_FIELDS = ("system", "prompt", "completion")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def require_runtime_paths(runtime_root: Path, paths: Iterable[Path]) -> None:
    runtime_root = Path(runtime_root).resolve()
    if str(runtime_root) == "/" or not str(runtime_root).startswith("/data/wy/"):
        raise RuntimeError("runtime root must be a dedicated directory under /data/wy")
    for path in paths:
        if not _is_within(path, runtime_root):
            raise RuntimeError(f"runtime artifact escapes runtime root: {path}")
    for name in ("TMPDIR", "HF_HOME", "TRANSFORMERS_CACHE", "PIP_CACHE_DIR"):
        value = os.environ.get(name)
        if not value or not _is_within(Path(value), runtime_root):
            raise RuntimeError(f"{name} must point under {runtime_root}")


def disk_free_gib(path: Path) -> float:
    return shutil.disk_usage(Path(path)).free / (1024 ** 3)


def require_disk_space(path: Path, minimum_gib: float) -> float:
    free = disk_free_gib(path)
    if free < minimum_gib:
        raise RuntimeError(
            f"insufficient disk space at {path}: {free:.2f} GiB free, "
            f"requires at least {minimum_gib:.2f} GiB"
        )
    return free


def load_public_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at line {line_number}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"dataset row {line_number} is not an object")
            for field in REQUIRED_FIELDS:
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise RuntimeError(f"dataset row {line_number} has invalid {field}")
            rows.append(row)
    if not rows:
        raise RuntimeError("public smoke dataset is empty")
    return rows


def import_and_version_check() -> dict[str, str]:
    import accelerate
    import huggingface_hub
    import packaging
    from packaging.version import Version
    import peft
    import safetensors
    import torch
    import transformers

    if Version(transformers.__version__) < Version(MIN_TRANSFORMERS):
        raise RuntimeError(
            f"transformers {transformers.__version__} is below required {MIN_TRANSFORMERS}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the isolated runtime")
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "safetensors": safetensors.__version__,
        "packaging": packaging.__version__,
    }


def gpu_status(minimum_free_gib: float) -> dict[str, Any]:
    import torch

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gib = free_bytes / (1024 ** 3)
    if free_gib < minimum_free_gib:
        raise RuntimeError(
            f"visible CUDA device has {free_gib:.2f} GiB free; "
            f"requires at least {minimum_free_gib:.2f} GiB"
        )
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_device_index": 0,
        "name": torch.cuda.get_device_name(0),
        "free_gib": round(free_gib, 3),
        "total_gib": round(total_bytes / (1024 ** 3), 3),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def require_exact_model_id(value: str) -> None:
    if value != EXPECTED_MODEL_ID:
        raise RuntimeError(
            f"smoke model must be exactly {EXPECTED_MODEL_ID}; got {value!r}. "
            "No legacy fallback is permitted."
        )


def preflight(args: argparse.Namespace) -> int:
    require_exact_model_id(args.model_id)
    runtime_root = Path(args.runtime_root)
    output_dir = runtime_root / "preflight"
    require_runtime_paths(runtime_root, (output_dir,))
    runtime_root.mkdir(parents=True, exist_ok=True)
    rows = load_public_rows(args.dataset)
    versions = import_and_version_check()
    gpu = gpu_status(args.minimum_gpu_free_gib)
    free = require_disk_space(runtime_root, args.minimum_disk_free_gib)
    probe = output_dir / ".write_probe"
    output_dir.mkdir(parents=True, exist_ok=True)
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()
    payload = {
        "status": "passed",
        "phase": "preflight",
        "model_id": args.model_id,
        "dataset": str(Path(args.dataset).resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "dataset_rows": len(rows),
        "runtime_root": str(runtime_root.resolve()),
        "disk_free_gib": round(free, 3),
        "gpu": gpu,
        "versions": versions,
        "thinking_mode": "disabled_required",
        "quantization": "none",
        "precision": "bf16",
    }
    write_json(output_dir / "preflight.json", payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def download(args: argparse.Namespace) -> int:
    require_exact_model_id(args.model_id)
    runtime_root = Path(args.runtime_root)
    manifest_path = runtime_root / "model" / "download_manifest.json"
    require_runtime_paths(runtime_root, (manifest_path,))
    require_disk_space(runtime_root, args.minimum_disk_free_gib)
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=args.model_id,
            cache_dir=os.environ["HF_HOME"],
        )
    ).resolve()
    config_path = snapshot / "config.json"
    if not config_path.is_file():
        raise RuntimeError("downloaded snapshot has no config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3":
        raise RuntimeError(f"downloaded model_type is not qwen3: {config.get('model_type')!r}")
    remaining = require_disk_space(runtime_root, args.minimum_remaining_gib)
    payload = {
        "status": "downloaded",
        "model_id": args.model_id,
        "model_type": config["model_type"],
        "architectures": config.get("architectures"),
        "snapshot_path": str(snapshot),
        "snapshot_revision": snapshot.name,
        "config_sha256": sha256_file(config_path),
        "disk_free_gib_after_download": round(remaining, 3),
    }
    write_json(manifest_path, payload)
    (manifest_path.parent / "snapshot_path.txt").write_text(str(snapshot) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _render_chat(tokenizer: Any, messages: list[dict[str, str]], *, generation: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=generation,
            enable_thinking=False,
        )
    except TypeError as exc:
        raise RuntimeError(
            "tokenizer chat template does not accept enable_thinking=False; "
            "refusing to run with ambiguous thinking mode"
        ) from exc


def encode_bounded_example(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_length: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    candidates = sorted(
        enumerate(rows),
        key=lambda item: (
            len(item[1]["system"]) + len(item[1]["prompt"]) + len(item[1]["completion"]),
            item[0],
        ),
    )
    for row_index, row in candidates:
        original_prompt = row["prompt"]
        low, high = 1, len(original_prompt)
        best: tuple[list[int], list[int], int] | None = None
        while low <= high:
            keep = (low + high) // 2
            messages = [
                {"role": "system", "content": row["system"]},
                {"role": "user", "content": original_prompt[:keep]},
                {"role": "assistant", "content": row["completion"]},
            ]
            full_text = _render_chat(tokenizer, messages, generation=False)
            prompt_text = _render_chat(tokenizer, messages[:2], generation=True)
            full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            if len(full_ids) <= max_length and len(prompt_ids) < len(full_ids):
                best = (full_ids, prompt_ids, keep)
                low = keep + 1
            else:
                high = keep - 1
        if best is None:
            continue
        full_ids, prompt_ids, kept_chars = best
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        if not any(value != -100 for value in labels):
            continue
        batch = {
            "input_ids": torch.tensor([full_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(full_ids)), dtype=torch.long),
            "labels": torch.tensor([labels], dtype=torch.long),
        }
        metadata = {
            "source_row_index": row_index,
            "source_user_id": row.get("user_id"),
            "source_qid": row.get("qid"),
            "input_tokens": len(full_ids),
            "completion_tokens": sum(value != -100 for value in labels),
            "original_prompt_chars": len(original_prompt),
            "kept_prompt_chars": kept_chars,
            "prompt_truncated_for_smoke": kept_chars < len(original_prompt),
        }
        return batch, metadata
    raise RuntimeError("no public dataset row can retain completion tokens within max_length")


def train(args: argparse.Namespace) -> int:
    require_exact_model_id(args.model_id)
    runtime_root = Path(args.runtime_root)
    output_dir = Path(args.output_dir)
    require_runtime_paths(runtime_root, (output_dir,))
    if output_dir.exists():
        raise RuntimeError(f"smoke output directory already exists: {output_dir}")
    free_before = require_disk_space(runtime_root, args.minimum_disk_free_gib)
    versions = import_and_version_check()
    gpu = gpu_status(args.minimum_gpu_free_gib)
    rows = load_public_rows(args.dataset)

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_path = Path(args.model_path).resolve()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if config.model_type != "qwen3" or EXPECTED_MODEL_ID.split("/")[-1] not in model_path.as_posix():
        raise RuntimeError("model snapshot identity does not match Qwen/Qwen3-8B")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch, sample_metadata = encode_bounded_example(
        tokenizer, rows, max_length=args.max_seq_length
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable <= 0 or trainable >= total:
        raise RuntimeError("LoRA trainable-parameter contract failed")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    device_batch = {key: value.to("cuda:0") for key, value in batch.items()}
    model.train()
    outputs = model(**device_batch, use_cache=False)
    loss = outputs.loss
    if loss is None or not math.isfinite(float(loss.detach().float().cpu())):
        raise RuntimeError("smoke forward pass produced a non-finite loss")
    loss.backward()
    gradient_tensors = sum(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if gradient_tensors == 0:
        raise RuntimeError("smoke backward pass produced no LoRA gradients")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    adapter_dir = output_dir / "adapter_step_1"
    adapter_dir.mkdir(parents=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    torch.save(optimizer.state_dict(), output_dir / "optimizer_step_1.pt")
    loss_value = float(loss.detach().float().cpu())
    payload = {
        "status": "passed",
        "purpose": "delivery_smoke_only_not_research_training",
        "optimizer_steps": 1,
        "model_id": args.model_id,
        "model_type": config.model_type,
        "model_snapshot": str(model_path),
        "precision": "bf16",
        "quantization": "none",
        "thinking_mode": "disabled",
        "gpu": gpu,
        "versions": versions,
        "dataset_sha256": sha256_file(args.dataset),
        "sample": sample_metadata,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "learning_rate": args.learning_rate,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "gradient_tensors": gradient_tensors,
        "loss": loss_value,
        "disk_free_gib_before_model_load": round(free_before, 3),
        "adapter_dir": str(adapter_dir),
        "optimizer_state": str(output_dir / "optimizer_step_1.pt"),
    }
    write_json(output_dir / "trainer_state.json", payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--runtime-root", type=Path, required=True)
    common.add_argument("--model-id", default=EXPECTED_MODEL_ID)

    check = subparsers.add_parser("preflight", parents=[common])
    check.add_argument("--dataset", type=Path, required=True)
    check.add_argument("--minimum-disk-free-gib", type=float, default=40.0)
    check.add_argument("--minimum-gpu-free-gib", type=float, default=60.0)
    check.set_defaults(func=preflight)

    fetch = subparsers.add_parser("download", parents=[common])
    fetch.add_argument("--minimum-disk-free-gib", type=float, default=40.0)
    fetch.add_argument("--minimum-remaining-gib", type=float, default=20.0)
    fetch.set_defaults(func=download)

    run = subparsers.add_parser("train", parents=[common])
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--minimum-disk-free-gib", type=float, default=20.0)
    run.add_argument("--minimum-gpu-free-gib", type=float, default=60.0)
    run.add_argument("--max-seq-length", type=int, default=1024)
    run.add_argument("--lora-r", type=int, default=8)
    run.add_argument("--learning-rate", type=float, default=1.0e-4)
    run.add_argument("--seed", type=int, default=20260721)
    run.set_defaults(func=train)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
