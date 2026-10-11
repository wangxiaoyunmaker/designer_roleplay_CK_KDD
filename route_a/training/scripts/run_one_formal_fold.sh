#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 DATASET_DIR MODEL_PATH FOLD_OUTPUT_DIR OUTER_FOLD [SEED]" >&2
  exit 2
fi

DATASET_DIR="$1"
MODEL_PATH="$2"
ROOT="$3"
FOLD="$4"
SEED="${5:-42}"
CONFIG="route_a/configs/training_v1.json"

if [[ "${DATASET_DIR}" != *"/formal" ]]; then
  echo "formal fold launcher requires a /formal dataset" >&2
  exit 2
fi
if [[ ! "${FOLD}" =~ ^[0-4]$ ]]; then
  echo "OUTER_FOLD must be one of 0,1,2,3,4" >&2
  exit 2
fi
if [[ -e "${ROOT}" ]]; then
  echo "fold output already exists; refusing to overwrite: ${ROOT}" >&2
  exit 2
fi

mkdir -p "${ROOT}"

run_stage() {
  local label="$1"
  shift
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START ${label}"
  python -m route_a.training "$@"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] PASS ${label}"
}

COMMON=(
  --dataset-dir "${DATASET_DIR}"
  --config "${CONFIG}"
  --outer-fold "${FOLD}"
  --model-path "${MODEL_PATH}"
  --device-index 0
  --seed "${SEED}"
)

run_stage aux-selection train-aux "${COMMON[@]}" \
  --output-dir "${ROOT}/aux-selection" --fit-mode selection
run_stage aux-refit train-aux "${COMMON[@]}" \
  --output-dir "${ROOT}/aux-refit" --fit-mode outer-refit \
  --selection-checkpoint "${ROOT}/aux-selection"

run_stage planner-selection train-planner "${COMMON[@]}" \
  --upstream-checkpoint "${ROOT}/aux-selection" \
  --output-dir "${ROOT}/planner-selection" --fit-mode selection
run_stage planner-refit train-planner "${COMMON[@]}" \
  --upstream-checkpoint "${ROOT}/aux-refit" \
  --output-dir "${ROOT}/planner-refit" --fit-mode outer-refit \
  --selection-checkpoint "${ROOT}/planner-selection"

run_stage realizer-selection train-realizer "${COMMON[@]}" \
  --upstream-checkpoint "${ROOT}/planner-selection" \
  --output-dir "${ROOT}/realizer-selection" --fit-mode selection
run_stage realizer-refit train-realizer "${COMMON[@]}" \
  --upstream-checkpoint "${ROOT}/planner-refit" \
  --output-dir "${ROOT}/realizer-refit" --fit-mode outer-refit \
  --selection-checkpoint "${ROOT}/realizer-selection"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] COMPLETE fold=${FOLD} seed=${SEED}"
