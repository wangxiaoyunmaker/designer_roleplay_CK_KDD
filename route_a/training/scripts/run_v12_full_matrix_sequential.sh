#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 DATASET_DIR MODEL_PATH OUTPUT_ROOT BGE_MODEL_PATH BGE_MODEL_SHA256 FROZEN_CONFIG PHYSICAL_GPU MIN_FREE_MIB" >&2
  exit 2
fi

DATASET_DIR="$1"
MODEL_PATH="$2"
OUTPUT_ROOT="$3"
BGE_MODEL_PATH="$4"
BGE_MODEL_SHA256="$5"
FROZEN_CONFIG="$6"
PHYSICAL_GPU="$7"
MIN_FREE_MIB="$8"
INFERENCE_CONFIG="route_a/configs/inference_v1.json"

if [[ "${DATASET_DIR}" != *"/formal" ]]; then
  echo "V12 full matrix requires a /formal dataset" >&2
  exit 2
fi
if [[ ! -f "${FROZEN_CONFIG}" ]]; then
  echo "frozen STOP-pilot config does not exist: ${FROZEN_CONFIG}" >&2
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
if [[ ! -d "${MODEL_PATH}" || ! -d "${BGE_MODEL_PATH}" || ! -f "${BGE_MODEL_SHA256}" ]]; then
  echo "model/BGE artifact path is missing" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}/.tmp"
export TMPDIR="${OUTPUT_ROOT}/.tmp"

require_disk_reserve() {
  local free_mib
  free_mib="$(df -Pm "${OUTPUT_ROOT}" | awk 'NR == 2 {print $4}')"
  if [[ ! "${free_mib}" =~ ^[0-9]+$ ]] || [[ "${free_mib}" -lt "${MIN_FREE_MIB}" ]]; then
    echo "insufficient disk reserve: free=${free_mib:-unknown}MiB required=${MIN_FREE_MIB}MiB" >&2
    exit 1
  fi
}

run_fold_seed() {
  local fold="$1"
  local seed="$2"
  local method_root="$3"
  local root="${OUTPUT_ROOT}/seed${seed}/fold${fold}"
  require_disk_reserve
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START_TRAIN fold=${fold} seed=${seed}"
  CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" bash route_a/training/scripts/run_one_v12_fold.sh \
    "${DATASET_DIR}" "${MODEL_PATH}" "${root}" "${fold}" "${seed}" \
    "${FROZEN_CONFIG}" "${method_root}"
  require_disk_reserve
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START_SELECTION_RFT fold=${fold} seed=${seed}"
  CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" bash route_a/training/scripts/run_v12_selection_and_rft.sh \
    "${DATASET_DIR}" "${MODEL_PATH}" "${root}" "${fold}" \
    "${BGE_MODEL_PATH}" "${BGE_MODEL_SHA256}" "${INFERENCE_CONFIG}" "${seed}" \
    "${FROZEN_CONFIG}" "${method_root}"
}

for FOLD in 0 1 2 3 4; do
  METHOD_ROOT="${OUTPUT_ROOT}/seed42/fold${FOLD}"
  run_fold_seed "${FOLD}" 42 "${METHOD_ROOT}"
  run_fold_seed "${FOLD}" 17 "${METHOD_ROOT}"
  run_fold_seed "${FOLD}" 2026 "${METHOD_ROOT}"
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] COMPLETE_V12_FULL_MATRIX output=${OUTPUT_ROOT}"
