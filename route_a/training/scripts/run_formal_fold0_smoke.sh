#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 DATASET_DIR MODEL_PATH OUTPUT_ROOT [CONFIG]" >&2
  exit 2
fi

DATASET_DIR="$1"
MODEL_PATH="$2"
OUTPUT_ROOT="$3"
CONFIG="${4:-route_a/configs/training_smoke_v1.json}"
SEED=42
FOLD=0

if [[ "${DATASET_DIR}" != *"/formal" ]]; then
  echo "formal smoke refuses a non-formal dataset" >&2
  exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "output already exists; refusing to overwrite: ${OUTPUT_ROOT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"

python -m route_a.training train-aux \
  --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
  --model-path "${MODEL_PATH}" --output-dir "${OUTPUT_ROOT}/aux-selection" \
  --device-index 0 --seed "${SEED}" --fit-mode selection
python -m route_a.training train-aux \
  --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
  --model-path "${MODEL_PATH}" --output-dir "${OUTPUT_ROOT}/aux-refit" \
  --device-index 0 --seed "${SEED}" --fit-mode outer-refit \
  --selection-checkpoint "${OUTPUT_ROOT}/aux-selection"

python -m route_a.training train-planner \
  --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
  --model-path "${MODEL_PATH}" --upstream-checkpoint "${OUTPUT_ROOT}/aux-selection" \
  --output-dir "${OUTPUT_ROOT}/planner-selection" --device-index 0 --seed "${SEED}" \
  --fit-mode selection
python -m route_a.training train-planner \
  --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
  --model-path "${MODEL_PATH}" --upstream-checkpoint "${OUTPUT_ROOT}/aux-refit" \
  --output-dir "${OUTPUT_ROOT}/planner-refit" --device-index 0 --seed "${SEED}" \
  --fit-mode outer-refit --selection-checkpoint "${OUTPUT_ROOT}/planner-selection"

python -m route_a.training train-realizer \
  --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
  --model-path "${MODEL_PATH}" --upstream-checkpoint "${OUTPUT_ROOT}/planner-selection" \
  --output-dir "${OUTPUT_ROOT}/realizer-selection" --device-index 0 --seed "${SEED}" \
  --fit-mode selection
python -m route_a.training train-realizer \
  --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
  --model-path "${MODEL_PATH}" --upstream-checkpoint "${OUTPUT_ROOT}/planner-refit" \
  --output-dir "${OUTPUT_ROOT}/realizer-refit" --device-index 0 --seed "${SEED}" \
  --fit-mode outer-refit --selection-checkpoint "${OUTPUT_ROOT}/realizer-selection"

echo "formal Route A fold-0 one-update stage chain completed"
