#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 DATASET_DIR MODEL_PATH TRAINING_ROOT OUTPUT_ROOT INSTRUMENT PHYSICAL_GPU MIN_FREE_MIB SEEDS_CSV" >&2
  exit 2
fi

DATASET_DIR="$1"
MODEL_PATH="$2"
TRAINING_ROOT="$3"
OUTPUT_ROOT="$4"
INSTRUMENT="$5"
PHYSICAL_GPU="$6"
MIN_FREE_MIB="$7"
SEEDS_CSV="$8"
TRAINING_CONFIG="route_a/configs/training_v1.json"
INFERENCE_CONFIG="route_a/configs/inference_v1.json"
ONTOLOGY="route_a/configs/state_ontology_v2.json"

if [[ "${DATASET_DIR}" != *"/formal" ]]; then
  echo "V10 Q7 inference requires a /formal dataset" >&2
  exit 2
fi
if [[ ! "${PHYSICAL_GPU}" =~ ^[0-9]+$ ]] || [[ ! "${MIN_FREE_MIB}" =~ ^[0-9]+$ ]]; then
  echo "PHYSICAL_GPU and MIN_FREE_MIB must be non-negative integers" >&2
  exit 2
fi
if [[ "${MIN_FREE_MIB}" -lt 8192 ]]; then
  echo "MIN_FREE_MIB must reserve at least 8192 MiB" >&2
  exit 2
fi
if [[ ! -f "${INSTRUMENT}" || ! -d "${MODEL_PATH}" || ! -f "${TRAINING_CONFIG}" || ! -f "${INFERENCE_CONFIG}" ]]; then
  echo "required V10 Q7 input is missing" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/.tmp"
export TMPDIR="${OUTPUT_ROOT}/.tmp"

require_disk_reserve() {
  local free_mib
  free_mib="$(df -Pm "${OUTPUT_ROOT}" | awk 'NR == 2 {print $4}')"
  if [[ ! "${free_mib}" =~ ^[0-9]+$ ]] || [[ "${free_mib}" -lt "${MIN_FREE_MIB}" ]]; then
    echo "insufficient disk reserve: free=${free_mib:-unknown}MiB required=${MIN_FREE_MIB}MiB" >&2
    exit 1
  fi
}

run_if_missing() {
  local artifact="$1"
  shift
  if [[ -f "${artifact}" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP ${artifact}"
    return
  fi
  "$@"
}

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
if [[ "${#SEEDS[@]}" -ne 3 || "${SEEDS[*]}" != "42 43 44" ]]; then
  echo "SEEDS_CSV must be exactly 42,43,44 for V10" >&2
  exit 2
fi

for SEED in "${SEEDS[@]}"; do
  SEED_ROOT="${OUTPUT_ROOT}/seed${SEED}"
  for FOLD in 0 1 2 3 4; do
    TRAIN_ROOT="${TRAINING_ROOT}/seed${SEED}/fold${FOLD}"
    CALIBRATION="${SEED_ROOT}/calibration/fold${FOLD}"
    PREDICTION="${SEED_ROOT}/fold${FOLD}"
    require_disk_reserve
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START_CALIBRATION seed=${SEED} fold=${FOLD}"
    run_if_missing "${CALIBRATION}/calibration_manifest.json" \
      env CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" python -m route_a.inference calibrate-best-of-n \
        --dataset-dir "${DATASET_DIR}" --training-config "${TRAINING_CONFIG}" \
        --inference-config "${INFERENCE_CONFIG}" --outer-fold "${FOLD}" \
        --planner-checkpoint "${TRAIN_ROOT}/planner-selection" \
        --realizer-checkpoint "${TRAIN_ROOT}/realizer-selection" \
        --model-path "${MODEL_PATH}" --output-dir "${CALIBRATION}" \
        --device-index 0 --run-seed "${SEED}"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START_Q7 seed=${SEED} fold=${FOLD}"
    run_if_missing "${PREDICTION}/q7_predictions_v1.sha256" \
      env CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" python -m route_a.inference generate-q7 \
        --dataset-dir "${DATASET_DIR}" --training-config "${TRAINING_CONFIG}" \
        --inference-config "${INFERENCE_CONFIG}" --instrument "${INSTRUMENT}" \
        --outer-fold "${FOLD}" --planner-checkpoint "${TRAIN_ROOT}/planner-refit" \
        --realizer-checkpoint "${TRAIN_ROOT}/realizer-refit" --model-path "${MODEL_PATH}" \
        --selection-calibration "${CALIBRATION}" --output-dir "${PREDICTION}" \
        --device-index 0 --run-seed "${SEED}"
  done
  run_if_missing "${SEED_ROOT}/merged/q7_predictions_v1.sha256" \
    python -m route_a.inference merge-q7-folds \
      --fold-dir "${SEED_ROOT}/fold0" --fold-dir "${SEED_ROOT}/fold1" \
      --fold-dir "${SEED_ROOT}/fold2" --fold-dir "${SEED_ROOT}/fold3" \
      --fold-dir "${SEED_ROOT}/fold4" --dataset-dir "${DATASET_DIR}" \
      --ontology "${ONTOLOGY}" --output-dir "${SEED_ROOT}/merged"
done

run_if_missing "${OUTPUT_ROOT}/merged-seeds/q7_predictions_v1.sha256" \
  python -m route_a.inference merge-q7-seeds \
    --seed-dir "${OUTPUT_ROOT}/seed42/merged" --seed-dir "${OUTPUT_ROOT}/seed43/merged" \
    --seed-dir "${OUTPUT_ROOT}/seed44/merged" --ontology "${ONTOLOGY}" \
    --training-config "${TRAINING_CONFIG}" --output-dir "${OUTPUT_ROOT}/merged-seeds"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] COMPLETE_V10_Q7_SINGLE_GPU output=${OUTPUT_ROOT}"
