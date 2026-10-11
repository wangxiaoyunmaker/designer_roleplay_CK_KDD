#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 DATASET_DIR MODEL_PATH OUTPUT_ROOT [SEED]" >&2
  exit 2
fi

DATASET_DIR="$1"
MODEL_PATH="$2"
OUTPUT_ROOT="$3"
SEED="${4:-42}"
CONFIG="route_a/configs/training_v1.json"
mkdir -p "${OUTPUT_ROOT}"

if [[ "${DATASET_DIR}" == *"/draft" ]]; then
  echo "formal training launcher refuses a draft dataset" >&2
  exit 2
fi

run_fold() {
  local fold="$1"
  local gpu="$2"
  local root="${OUTPUT_ROOT}/seed${SEED}/fold${fold}"
  mkdir -p "${root}"

  CUDA_VISIBLE_DEVICES="${gpu}" python -m route_a.training train-aux \
    --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${fold}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${root}/aux-selection" --device-index 0 --seed "${SEED}" \
    --fit-mode selection
  CUDA_VISIBLE_DEVICES="${gpu}" python -m route_a.training train-aux \
    --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${fold}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${root}/aux-refit" --device-index 0 --seed "${SEED}" \
    --fit-mode outer-refit --selection-checkpoint "${root}/aux-selection"

  CUDA_VISIBLE_DEVICES="${gpu}" python -m route_a.training train-planner \
    --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${fold}" \
    --model-path "${MODEL_PATH}" \
    --upstream-checkpoint "${root}/aux-selection" \
    --output-dir "${root}/planner-selection" --device-index 0 --seed "${SEED}" \
    --fit-mode selection
  CUDA_VISIBLE_DEVICES="${gpu}" python -m route_a.training train-planner \
    --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${fold}" \
    --model-path "${MODEL_PATH}" \
    --upstream-checkpoint "${root}/aux-refit" \
    --output-dir "${root}/planner-refit" --device-index 0 --seed "${SEED}" \
    --fit-mode outer-refit --selection-checkpoint "${root}/planner-selection"

  CUDA_VISIBLE_DEVICES="${gpu}" python -m route_a.training train-realizer \
    --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${fold}" \
    --model-path "${MODEL_PATH}" \
    --upstream-checkpoint "${root}/planner-selection" \
    --output-dir "${root}/realizer-selection" --device-index 0 --seed "${SEED}" \
    --fit-mode selection
  CUDA_VISIBLE_DEVICES="${gpu}" python -m route_a.training train-realizer \
    --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${fold}" \
    --model-path "${MODEL_PATH}" \
    --upstream-checkpoint "${root}/planner-refit" \
    --output-dir "${root}/realizer-refit" --device-index 0 --seed "${SEED}" \
    --fit-mode outer-refit --selection-checkpoint "${root}/realizer-selection"
}

pids=()
for fold in 0 1 2 3 4; do
  run_fold "${fold}" "${fold}" >"${OUTPUT_ROOT}/fold${fold}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "at least one fold failed; inspect ${OUTPUT_ROOT}/fold*.log" >&2
  exit 1
fi

echo "five folds completed; GPUs 5-7 remain available for smoke tests, ablations, or reruns"
