# Qwen3-8B delivery smoke test

This directory is a bounded infrastructure check, not a research-training implementation. It proves one real optimizer step with the public SFT JSONL, `Qwen/Qwen3-8B`, BF16, no quantization, LoRA, and thinking mode explicitly disabled.

The server workflow is intentionally fixed to:

- clean checkout: `/data/wy/roleplay_public_smoke`;
- isolated runtime: `/data/wy/roleplay_smoke_runtime`;
- physical GPU 0 only;
- one optimizer step;
- model/cache/temp/log/checkpoint writes only below the runtime root;
- no fallback to any Qwen2.5 path;
- no modification of existing server repositories or their environments.

Run only after reviewing `run_server_smoke.sh`:

```bash
cd /data/wy/roleplay_public_smoke
bash smoke/run_server_smoke.sh
```

Success requires all of the following:

- `preflight/preflight.json` reports Transformers `>=4.51.0`, CUDA, BF16, at least 60 GiB free GPU memory, and the public dataset hash;
- `model/download_manifest.json` reports `model_id=Qwen/Qwen3-8B` and `model_type=qwen3`;
- `checkpoints/qwen3_8b_one_step/trainer_state.json` reports `optimizer_steps=1` and a finite loss;
- `adapter_step_1/`, `optimizer_step_1.pt`, the full log, pip freeze, and exit-status file exist;
- the two pre-existing dirty repositories have byte-identical captured `git status` before and after the run.

The wrapper refuses to overwrite an earlier run. A repeated smoke requires a separately named runtime directory and run ID rather than deleting evidence. `RUNTIME_ROOT`, `RUN_ID`, `PIP_INDEX_URL`, and `HF_ENDPOINT` may be supplied explicitly for an isolated retry; the selected values are recorded in the run log.
