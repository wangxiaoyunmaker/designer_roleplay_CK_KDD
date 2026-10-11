#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 4 || $# -gt 7 ]]; then
  echo "usage: $0 DATASET_DIR MODEL_PATH FINAL_ROOT OUTER_FOLD [FINAL_SEED] [CONFIG] [METHOD_ROOT]" >&2
  exit 2
fi

DATASET_DIR="$1"
MODEL_PATH="$2"
ROOT="$3"
FOLD="$4"
FINAL_SEED="${5:-42}"
METHOD_SEED=42
CONFIG="${6:-route_a/configs/training_v12.json}"
METHOD_ROOT="${7:-${ROOT}}"
# Keep the established V12 default (bf16 LoRA), while allowing a versioned
# experiment launcher to opt into QLoRA without duplicating this 24-stage
# nested-CV orchestrator.  The flag is recorded by every checkpoint manifest.
TRAINING_RUNTIME_FLAGS=()
if [[ "${ROUTE_A_QLORA:-0}" == "1" ]]; then
  TRAINING_RUNTIME_FLAGS=(--qlora)
elif [[ "${ROUTE_A_QLORA:-0}" != "0" ]]; then
  echo "ROUTE_A_QLORA must be unset, 0, or 1" >&2
  exit 2
fi

if [[ "${DATASET_DIR}" != *"/formal" ]]; then
  echo "V12 launcher requires a /formal dataset" >&2
  exit 2
fi
if [[ ! "${FOLD}" =~ ^[0-4]$ ]]; then
  echo "OUTER_FOLD must be one of 0,1,2,3,4" >&2
  exit 2
fi
if [[ ! "${FINAL_SEED}" =~ ^(17|42|2026)$ ]]; then
  echo "FINAL_SEED must be one of 17,42,2026" >&2
  exit 2
fi

mkdir -p "${ROOT}"
mkdir -p "${ROOT}/.tmp"
# The server's system /tmp may be full even when the training volume has
# capacity. Keep Python/safetensors temporary files beside this fold so a
# direct invocation is as robust as the managed launcher.
export TMPDIR="${ROOT}/.tmp"
if [[ "${METHOD_ROOT}" != "${ROOT}" && ! -d "${METHOD_ROOT}" ]]; then
  echo "shared METHOD_ROOT does not exist: ${METHOD_ROOT}" >&2
  exit 2
fi

run_stage() {
  local label="$1"
  local output="$2"
  shift 2
  if [[ -f "${output}/checkpoint_manifest.json" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP_COMPLETE ${label}"
    return
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START ${label}"
  python -m route_a.training "$@"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] PASS ${label}"
}

if [[ "${METHOD_ROOT}" == "${ROOT}" ]]; then
  for INNER in 0 1 2; do
    run_stage "aux-inner-${INNER}" "${METHOD_ROOT}/aux-inner-${INNER}" train-aux \
      --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
      --model-path "${MODEL_PATH}" --device-index 0 --seed "${METHOD_SEED}" \
      --fit-mode inner-selection --inner-fold "${INNER}" \
      --output-dir "${METHOD_ROOT}/aux-inner-${INNER}" "${TRAINING_RUNTIME_FLAGS[@]}"
    run_stage "planner-inner-${INNER}" "${METHOD_ROOT}/planner-inner-${INNER}" train-planner \
      --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
      --model-path "${MODEL_PATH}" --device-index 0 --seed "${METHOD_SEED}" \
      --fit-mode inner-selection --inner-fold "${INNER}" \
      --upstream-checkpoint "${METHOD_ROOT}/aux-inner-${INNER}" \
      --output-dir "${METHOD_ROOT}/planner-inner-${INNER}" "${TRAINING_RUNTIME_FLAGS[@]}"
    run_stage "realizer-inner-${INNER}" "${METHOD_ROOT}/realizer-inner-${INNER}" train-realizer \
      --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
      --model-path "${MODEL_PATH}" --device-index 0 --seed "${METHOD_SEED}" \
      --fit-mode inner-selection --inner-fold "${INNER}" \
      --upstream-checkpoint "${METHOD_ROOT}/planner-inner-${INNER}" \
      --output-dir "${METHOD_ROOT}/realizer-inner-${INNER}" "${TRAINING_RUNTIME_FLAGS[@]}"
  done

  # Strict OOF generators never validate or early-stop on their target fold.
  for INNER in 0 1 2; do
    run_stage "aux-oof-${INNER}" "${METHOD_ROOT}/aux-oof-${INNER}" train-aux \
      --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
      --model-path "${MODEL_PATH}" --device-index 0 --seed "${METHOD_SEED}" \
      --fit-mode oof-refit --inner-fold "${INNER}" \
      --output-dir "${METHOD_ROOT}/aux-oof-${INNER}" "${TRAINING_RUNTIME_FLAGS[@]}"
    run_stage "planner-oof-${INNER}" "${METHOD_ROOT}/planner-oof-${INNER}" train-planner \
      --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
      --model-path "${MODEL_PATH}" --device-index 0 --seed "${METHOD_SEED}" \
      --fit-mode oof-refit --inner-fold "${INNER}" \
      --upstream-checkpoint "${METHOD_ROOT}/aux-oof-${INNER}" \
      --output-dir "${METHOD_ROOT}/planner-oof-${INNER}" "${TRAINING_RUNTIME_FLAGS[@]}"
    run_stage "realizer-oof-${INNER}" "${METHOD_ROOT}/realizer-oof-${INNER}" train-realizer \
      --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
      --model-path "${MODEL_PATH}" --device-index 0 --seed "${METHOD_SEED}" \
      --fit-mode oof-refit --inner-fold "${INNER}" \
      --upstream-checkpoint "${METHOD_ROOT}/planner-oof-${INNER}" \
      --output-dir "${METHOD_ROOT}/realizer-oof-${INNER}" "${TRAINING_RUNTIME_FLAGS[@]}"
  done

  AUX_SELECTION=(
    --selection-checkpoint "${METHOD_ROOT}/aux-inner-0"
    --selection-checkpoint "${METHOD_ROOT}/aux-inner-1"
    --selection-checkpoint "${METHOD_ROOT}/aux-inner-2"
  )
  run_stage aux-provisional "${METHOD_ROOT}/aux-provisional" train-aux \
    --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
    --model-path "${MODEL_PATH}" --device-index 0 --seed "${METHOD_SEED}" \
    --fit-mode provisional "${AUX_SELECTION[@]}" \
    --output-dir "${METHOD_ROOT}/aux-provisional" "${TRAINING_RUNTIME_FLAGS[@]}"
  PLANNER_SELECTION=(
    --selection-checkpoint "${METHOD_ROOT}/planner-inner-0"
    --selection-checkpoint "${METHOD_ROOT}/planner-inner-1"
    --selection-checkpoint "${METHOD_ROOT}/planner-inner-2"
  )
  run_stage planner-provisional "${METHOD_ROOT}/planner-provisional" train-planner \
    --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
    --model-path "${MODEL_PATH}" --device-index 0 --seed "${METHOD_SEED}" \
    --fit-mode provisional "${PLANNER_SELECTION[@]}" \
    --upstream-checkpoint "${METHOD_ROOT}/aux-provisional" \
    --output-dir "${METHOD_ROOT}/planner-provisional" "${TRAINING_RUNTIME_FLAGS[@]}"
  REALIZER_SELECTION=(
    --selection-checkpoint "${METHOD_ROOT}/realizer-inner-0"
    --selection-checkpoint "${METHOD_ROOT}/realizer-inner-1"
    --selection-checkpoint "${METHOD_ROOT}/realizer-inner-2"
  )
  run_stage realizer-provisional "${METHOD_ROOT}/realizer-provisional" train-realizer \
    --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
    --model-path "${MODEL_PATH}" --device-index 0 --seed "${METHOD_SEED}" \
    --fit-mode provisional "${REALIZER_SELECTION[@]}" \
    --upstream-checkpoint "${METHOD_ROOT}/planner-provisional" \
    --output-dir "${METHOD_ROOT}/realizer-provisional" "${TRAINING_RUNTIME_FLAGS[@]}"
fi

AUX_SELECTION=(
  --selection-checkpoint "${METHOD_ROOT}/aux-inner-0"
  --selection-checkpoint "${METHOD_ROOT}/aux-inner-1"
  --selection-checkpoint "${METHOD_ROOT}/aux-inner-2"
)
run_stage aux-refit "${ROOT}/aux-refit" train-aux \
  --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
  --model-path "${MODEL_PATH}" --device-index 0 --seed "${FINAL_SEED}" \
  --fit-mode outer-refit "${AUX_SELECTION[@]}" --output-dir "${ROOT}/aux-refit" \
  "${TRAINING_RUNTIME_FLAGS[@]}"

PLANNER_SELECTION=(
  --selection-checkpoint "${METHOD_ROOT}/planner-inner-0"
  --selection-checkpoint "${METHOD_ROOT}/planner-inner-1"
  --selection-checkpoint "${METHOD_ROOT}/planner-inner-2"
)
run_stage planner-refit "${ROOT}/planner-refit" train-planner \
  --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
  --model-path "${MODEL_PATH}" --device-index 0 --seed "${FINAL_SEED}" \
  --fit-mode outer-refit "${PLANNER_SELECTION[@]}" \
  --upstream-checkpoint "${ROOT}/aux-refit" --output-dir "${ROOT}/planner-refit" \
  "${TRAINING_RUNTIME_FLAGS[@]}"

REALIZER_SELECTION=(
  --selection-checkpoint "${METHOD_ROOT}/realizer-inner-0"
  --selection-checkpoint "${METHOD_ROOT}/realizer-inner-1"
  --selection-checkpoint "${METHOD_ROOT}/realizer-inner-2"
)
run_stage realizer-refit "${ROOT}/realizer-refit" train-realizer \
  --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
  --model-path "${MODEL_PATH}" --device-index 0 --seed "${FINAL_SEED}" \
  --fit-mode outer-refit "${REALIZER_SELECTION[@]}" \
  --upstream-checkpoint "${ROOT}/planner-refit" --output-dir "${ROOT}/realizer-refit" \
  "${TRAINING_RUNTIME_FLAGS[@]}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] COMPLETE_V12 fold=${FOLD} seed=${FINAL_SEED} method_root=${METHOD_ROOT}"
