#!/usr/bin/env bash
set -Eeuo pipefail

EXPECTED_REPO="/data/wy/roleplay_public_smoke"
RUNTIME_ROOT="${RUNTIME_ROOT:-/data/wy/roleplay_smoke_runtime}"
BASE_PYTHON="/home/wy/RolePlay_Designer/0512技术方案/.venv/bin/python"
DATASET="$EXPECTED_REPO/data/processed/sft_train.jsonl"
VENV="$RUNTIME_ROOT/venv"
LOG_DIR="$RUNTIME_ROOT/logs"
RUN_ID="${RUN_ID:-qwen3_8b_one_step}"
RUN_DIR="$RUNTIME_ROOT/checkpoints/$RUN_ID"
LOG_FILE="$LOG_DIR/$RUN_ID.log"
STATUS_FILE="$LOG_DIR/$RUN_ID.exit_status"

if [[ "$(pwd -P)" != "$EXPECTED_REPO" ]]; then
  echo "error: run from $EXPECTED_REPO" >&2
  exit 2
fi
if [[ ! -x "$BASE_PYTHON" ]]; then
  echo "error: CUDA base Python is unavailable: $BASE_PYTHON" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" || -e "$STATUS_FILE" ]]; then
  echo "error: smoke output already exists; refusing to overwrite" >&2
  exit 2
fi

mkdir -p "$RUNTIME_ROOT/tmp" "$RUNTIME_ROOT/hf" "$RUNTIME_ROOT/transformers" \
  "$RUNTIME_ROOT/pip-cache" "$RUNTIME_ROOT/model" "$LOG_DIR" \
  "$RUNTIME_ROOT/checkpoints"
export TMPDIR="$RUNTIME_ROOT/tmp"
export HF_HOME="$RUNTIME_ROOT/hf"
export TRANSFORMERS_CACHE="$RUNTIME_ROOT/transformers"
export PIP_CACHE_DIR="$RUNTIME_ROOT/pip-cache"
export XDG_CACHE_HOME="$RUNTIME_ROOT/hf"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

exec > >(tee -a "$LOG_FILE") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$STATUS_FILE"; exit "$status"' EXIT

echo "command: smoke/run_server_smoke.sh"
echo "repo_commit: $(git rev-parse HEAD)"
echo "runtime_root: $RUNTIME_ROOT"
echo "pip_index_url: ${PIP_INDEX_URL:-default}"
echo "hf_endpoint: ${HF_ENDPOINT:-default}"
echo "started_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
df -h /data
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv,noheader

if [[ ! -x "$VENV/bin/python" ]]; then
  "$BASE_PYTHON" -m venv "$VENV"
fi
SITE_PACKAGES="$($VENV/bin/python -c 'import site; print(site.getsitepackages()[0])')"
BASE_SITE_PACKAGES="$($BASE_PYTHON -c 'import site; print(site.getsitepackages()[0])')"
printf "%s\n" "$BASE_SITE_PACKAGES" > "$SITE_PACKAGES/reuse_cuda_torch_read_only.pth"

"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install --upgrade -r smoke/requirements-smoke.txt
"$VENV/bin/python" -m pip freeze | sort > "$LOG_DIR/pip_freeze.txt"

"$VENV/bin/python" smoke/qwen3_smoke_train.py preflight \
  --runtime-root "$RUNTIME_ROOT" \
  --dataset "$DATASET" \
  --model-id Qwen/Qwen3-8B \
  --minimum-disk-free-gib 40 \
  --minimum-gpu-free-gib 60

"$VENV/bin/python" smoke/qwen3_smoke_train.py download \
  --runtime-root "$RUNTIME_ROOT" \
  --model-id Qwen/Qwen3-8B \
  --minimum-disk-free-gib 40 \
  --minimum-remaining-gib 20

MODEL_PATH="$(tr -d '\n' < "$RUNTIME_ROOT/model/snapshot_path.txt")"
"$VENV/bin/python" smoke/qwen3_smoke_train.py train \
  --runtime-root "$RUNTIME_ROOT" \
  --dataset "$DATASET" \
  --model-id Qwen/Qwen3-8B \
  --model-path "$MODEL_PATH" \
  --output-dir "$RUN_DIR" \
  --minimum-disk-free-gib 20 \
  --minimum-gpu-free-gib 60 \
  --max-seq-length 1024 \
  --lora-r 8 \
  --learning-rate 1e-4 \
  --seed 20260721

echo "completed_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
