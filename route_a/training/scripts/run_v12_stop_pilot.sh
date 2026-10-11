#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 DATASET_DIR MODEL_PATH OUTPUT_DIR OUTER_FOLD PHYSICAL_GPU" >&2
  exit 2
fi

DATASET_DIR="$1"
MODEL_PATH="$2"
ROOT="$3"
FOLD="$4"
PHYSICAL_GPU="$5"
BASE_CONFIG="route_a/configs/training_v12.json"

if [[ "${DATASET_DIR}" != *"/formal" || ! "${FOLD}" =~ ^[0-4]$ ]]; then
  echo "STOP pilot requires /formal data and outer fold 0-4" >&2
  exit 2
fi
if [[ ! "${PHYSICAL_GPU}" =~ ^[0-9]+$ ]]; then
  echo "PHYSICAL_GPU must be a non-negative integer" >&2
  exit 2
fi
mkdir -p "${ROOT}"
mkdir -p "${ROOT}/.tmp"
export TMPDIR="${ROOT}/.tmp"
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
if [[ ! -f "${ROOT}/configs/pilot_config_manifest.json" ]]; then
  python -m route_a.training build-stop-pilot-configs \
    --base-config "${BASE_CONFIG}" --output-dir "${ROOT}/configs"
fi

for LABEL in natural 20 25; do
  CONFIG="${ROOT}/configs/training_v12_stop_${LABEL}.json"
  if [[ ! -f "${ROOT}/aux-${LABEL}/checkpoint_manifest.json" ]]; then
    python -m route_a.training train-aux \
      --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
      --model-path "${MODEL_PATH}" --device-index 0 --seed 42 \
      --fit-mode inner-selection --inner-fold 0 --output-dir "${ROOT}/aux-${LABEL}"
  fi
  if [[ ! -f "${ROOT}/planner-${LABEL}/checkpoint_manifest.json" ]]; then
    python -m route_a.training train-planner \
      --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
      --model-path "${MODEL_PATH}" --device-index 0 --seed 42 \
      --fit-mode inner-selection --inner-fold 0 \
      --upstream-checkpoint "${ROOT}/aux-${LABEL}" \
      --output-dir "${ROOT}/planner-${LABEL}"
  fi
done

if [[ ! -f "${ROOT}/stop_pilot_selection.json" ]]; then
  python -m route_a.training select-stop-pilot \
    --checkpoint "${ROOT}/planner-natural" \
    --checkpoint "${ROOT}/planner-20" \
    --checkpoint "${ROOT}/planner-25" \
    --base-config "${BASE_CONFIG}" \
    --output-config "${ROOT}/frozen/training_v12.json" \
    --report "${ROOT}/stop_pilot_selection.json"
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] COMPLETE_STOP_PILOT fold=${FOLD}"
