#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 DATASET_DIR MODEL_PATH OUTPUT_ROOT [PHYSICAL_GPU]" >&2
  exit 2
fi

DATASET_DIR="$1"
MODEL_PATH="$2"
ROOT="$3"
PHYSICAL_GPU="${4:-0}"
CONFIG="route_a/configs/training_v12_smoke.json"

if [[ "${DATASET_DIR}" != *"/formal" ]]; then
  echo "V12 GPU smoke requires a /formal dataset" >&2
  exit 2
fi
if [[ ! "${PHYSICAL_GPU}" =~ ^[0-9]+$ ]]; then
  echo "PHYSICAL_GPU must be a non-negative integer" >&2
  exit 2
fi
if [[ -e "${ROOT}" ]]; then
  echo "smoke output already exists; use a new OUTPUT_ROOT: ${ROOT}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "model path does not exist: ${MODEL_PATH}" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable" >&2
  exit 2
fi

GPU_FREE_MIB="$(nvidia-smi --id="${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${GPU_FREE_MIB}" =~ ^[0-9]+$ || "${GPU_FREE_MIB}" -lt 60000 ]]; then
  echo "GPU ${PHYSICAL_GPU} has only ${GPU_FREE_MIB:-unknown} MiB free; require >=60000 MiB" >&2
  exit 2
fi

mkdir -p "${ROOT}"
mkdir -p "${ROOT}/.tmp"
export TMPDIR="${ROOT}/.tmp"
{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "physical_gpu=${PHYSICAL_GPU}"
  echo "gpu_free_mib=${GPU_FREE_MIB}"
  echo "dataset_dir=${DATASET_DIR}"
  echo "model_path=${MODEL_PATH}"
  echo "config=${CONFIG}"
  sha256sum "${CONFIG}"
  nvidia-smi --id="${PHYSICAL_GPU}" --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader
} > "${ROOT}/preflight.txt"

set +e
CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" bash route_a/training/scripts/run_one_v12_fold.sh \
  "${DATASET_DIR}" "${MODEL_PATH}" "${ROOT}/fold0" 0 42 "${CONFIG}" \
  > "${ROOT}/training.log" 2>&1
STATUS=$?
set -e

printf '%s\n' "${STATUS}" > "${ROOT}/exit_status.txt"
if [[ "${STATUS}" -ne 0 ]]; then
  echo "V12 one-GPU smoke failed; inspect ${ROOT}/training.log" >&2
  exit "${STATUS}"
fi

MANIFEST_COUNT="$(find "${ROOT}/fold0" -name checkpoint_manifest.json -type f | wc -l | tr -d ' ')"
if [[ "${MANIFEST_COUNT}" -ne 24 ]]; then
  echo "expected 24 checkpoint manifests, found ${MANIFEST_COUNT}" >&2
  exit 1
fi

{
  echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "checkpoint_manifests=${MANIFEST_COUNT}"
  echo "status=PASS"
} > "${ROOT}/smoke_result.txt"
echo "V12_ONE_GPU_SMOKE_PASS output=${ROOT} manifests=${MANIFEST_COUNT}"
