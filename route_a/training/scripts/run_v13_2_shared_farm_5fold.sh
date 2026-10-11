#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 DATASET_DIR MODEL_PATH OUTPUT_ROOT [GPU_CSV]" >&2
  exit 2
fi

DATASET_DIR="$1"
MODEL_PATH="$2"
ROOT="$3"
GPU_CSV="${4:-0,2,3,6,7}"
PYTHON_BIN="${ROUTE_A_PYTHON:-python}"
CONFIG="route_a/configs/training_v13_2_shared_farm.json"
IFS=',' read -r -a GPUS <<<"${GPU_CSV}"

if [[ "${#GPUS[@]}" -ne 5 ]]; then
  echo "GPU_CSV must contain exactly five physical GPU indices" >&2
  exit 2
fi
if [[ "${DATASET_DIR}" != *"/formal" ]]; then
  echo "Shared FaRM five-fold training requires a /formal dataset" >&2
  exit 2
fi
if [[ -e "${ROOT}" ]]; then
  echo "output root already exists: ${ROOT}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "training interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l | tr -d ' ')" -ne 5 ]]; then
  echo "GPU_CSV contains duplicate indices" >&2
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "invalid GPU index: ${gpu}" >&2
    exit 2
  fi
  free="$(nvidia-smi --id="${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if [[ ! "${free}" =~ ^[0-9]+$ || "${free}" -lt 60000 ]]; then
    echo "GPU ${gpu} has only ${free:-unknown} MiB free; require >=60000 MiB" >&2
    exit 2
  fi
done

mkdir -p "${ROOT}"
{
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "physical_gpus=${GPU_CSV}"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "config=${CONFIG}"
  echo "fit_mode=outer-refit"
  echo "q7_gold_used=false"
} >"${ROOT}/provenance.txt"

run_fold() {
  local fold="$1"
  local gpu="$2"
  local output="${ROOT}/fold${fold}"
  local temp_root="${ROOT}/fold${fold}-runtime"
  mkdir -p "${temp_root}/tmp" "${temp_root}/torch-cache"
  TMPDIR="${temp_root}/tmp" TEMP="${temp_root}/tmp" TMP="${temp_root}/tmp" \
  TORCHINDUCTOR_CACHE_DIR="${temp_root}/torch-cache" \
  CUBLAS_WORKSPACE_CONFIG=":4096:8" \
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    -m route_a.training.shared_farm_train \
    --dataset-dir "${DATASET_DIR}" \
    --config "${CONFIG}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${output}" \
    --outer-fold "${fold}" \
    --fit-mode outer-refit \
    --device-index 0 \
    --seed 42 \
    --qlora >"${ROOT}/fold${fold}.log" 2>&1
}

pids=()
for fold in 0 1 2 3 4; do
  run_fold "${fold}" "${GPUS[$fold]}" &
  pids+=("$!")
  echo "${pids[-1]}" >"${ROOT}/fold${fold}.pid"
done

failed=0
for index in 0 1 2 3 4; do
  if wait "${pids[$index]}"; then
    echo "PASS" >"${ROOT}/fold${index}.status"
  else
    echo "FAIL" >"${ROOT}/fold${index}.status"
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "at least one Shared FaRM fold failed; inspect ${ROOT}/fold*.log" >&2
  exit 1
fi

echo "PASS" >"${ROOT}/five_fold_result.txt"
echo "V13_2_SHARED_FARM_5FOLD_PASS output=${ROOT}"
