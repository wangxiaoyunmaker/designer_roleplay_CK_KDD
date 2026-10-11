#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 || "$#" -gt 5 ]]; then
  echo "usage: $0 DATASET_DIR MODEL_PATH OUTPUT_DIR [GPU_CSV] [BASE_CONFIG]" >&2
  exit 2
fi

DATASET_DIR="$(realpath "$1")"
MODEL_PATH="$(realpath "$2")"
OUTPUT_DIR="$(realpath -m "$3")"
GPU_CSV="${4:-0,2,3,6,7}"
BASE_CONFIG="${5:-route_a/configs/training_v13_3_delta_sequence_farm.json}"
PYTHON_BIN="${PYTHON_BIN:-/data/wy/roleplay_smoke_runtime_v2/venv/bin/python}"
IFS=',' read -r -a GPUS <<< "$GPU_CSV"
VARIANTS=(semantic_baseline shared_r4 shared_r8 shared_r16 shared_r32)

if [[ "${#GPUS[@]}" -ne 5 ]]; then
  echo "exactly five physical GPUs are required" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "output directory already exists: $OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/tmp"
export TMPDIR="$OUTPUT_DIR/tmp"
export TEMP="$OUTPUT_DIR/tmp"
export TMP="$OUTPUT_DIR/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$PYTHON_BIN" -m route_a.training.prepare_v13_3_capacity_configs \
  --base-config "$BASE_CONFIG" \
  --output-dir "$OUTPUT_DIR/frozen"

if [[ -n "${STAGE_B_SOURCE:-}" ]]; then
  STAGE_B="$(realpath "$STAGE_B_SOURCE")"
  printf '%s\n' "$STAGE_B" >"$OUTPUT_DIR/reused_stage_b_source.txt"
else
  STAGE_B="$OUTPUT_DIR/stage_b"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON_BIN" \
    -m route_a.training.shared_farm_train \
    --dataset-dir "$DATASET_DIR" \
    --config "$OUTPUT_DIR/frozen/shared_r4.json" \
    --model-path "$MODEL_PATH" \
    --output-dir "$STAGE_B" \
    --outer-fold 0 \
    --fit-mode inner-selection \
    --inner-fold 0 \
    --device-index 0 \
    --seed 42 \
    --qlora \
    --stop-after-stage-b \
    >"$OUTPUT_DIR/stage_b.log" 2>&1
fi

if [[ ! -f "$STAGE_B/stage_b_manifest.json" ]]; then
  echo "Stage B checkpoint did not complete" >&2
  exit 1
fi

pids=()
for index in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$index]}"
  gpu="${GPUS[$index]}"
  (
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" \
      -m route_a.training.shared_farm_train \
      --dataset-dir "$DATASET_DIR" \
      --config "$OUTPUT_DIR/frozen/$variant.json" \
      --model-path "$MODEL_PATH" \
      --output-dir "$OUTPUT_DIR/$variant" \
      --outer-fold 0 \
      --fit-mode inner-selection \
      --inner-fold 0 \
      --device-index 0 \
      --seed 42 \
      --qlora \
      --resume-stage-b "$STAGE_B" \
      >"$OUTPUT_DIR/$variant.log" 2>&1
    status="$?"
    printf '%s\n' "$status" >"$OUTPUT_DIR/$variant.exit_status.txt"
    exit "$status"
  ) &
  pids+=("$!")
  printf '%s\n' "${pids[-1]}" >"$OUTPUT_DIR/$variant.pid"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  printf 'FAIL\n' >"$OUTPUT_DIR/pilot_result.txt"
  exit 1
fi
printf 'PASS\n' >"$OUTPUT_DIR/pilot_result.txt"
