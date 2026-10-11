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
PYTHON_BIN="${ROUTE_A_PYTHON:-python}"
CONFIG="route_a/configs/training_v13_2_shared_farm_smoke.json"

if [[ "${DATASET_DIR}" != *"/formal" ]]; then
  echo "Shared FaRM smoke requires a /formal dataset" >&2
  exit 2
fi
if [[ ! "${PHYSICAL_GPU}" =~ ^[0-9]+$ ]]; then
  echo "PHYSICAL_GPU must be a non-negative integer" >&2
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

GPU_FREE_MIB="$(nvidia-smi --id="${PHYSICAL_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${GPU_FREE_MIB}" =~ ^[0-9]+$ || "${GPU_FREE_MIB}" -lt 60000 ]]; then
  echo "GPU ${PHYSICAL_GPU} has only ${GPU_FREE_MIB:-unknown} MiB free; require >=60000 MiB" >&2
  exit 2
fi

mkdir -p "${ROOT}"
mkdir -p "${ROOT}/tmp" "${ROOT}/torch-cache"
export TMPDIR="${ROOT}/tmp"
export TEMP="${ROOT}/tmp"
export TMP="${ROOT}/tmp"
export TORCHINDUCTOR_CACHE_DIR="${ROOT}/torch-cache"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
{
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "physical_gpu=${PHYSICAL_GPU}"
  echo "gpu_free_mib=${GPU_FREE_MIB}"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "config=${CONFIG}"
  echo "q7_gold_used=false"
  echo "tmpdir=${TMPDIR}"
} >"${ROOT}/provenance.txt"

set +e
CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" "${PYTHON_BIN}" \
  -m route_a.training.shared_farm_train \
  --dataset-dir "${DATASET_DIR}" \
  --config "${CONFIG}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${ROOT}/checkpoint" \
  --outer-fold 0 \
  --fit-mode inner-selection \
  --inner-fold 0 \
  --device-index 0 \
  --seed 42 \
  --qlora >"${ROOT}/training.log" 2>&1
STATUS="$?"
set -e
echo "${STATUS}" >"${ROOT}/exit_status.txt"
if [[ "${STATUS}" -ne 0 ]]; then
  echo "V13.2 Shared FaRM smoke failed; inspect ${ROOT}/training.log" >&2
  exit "${STATUS}"
fi

"${PYTHON_BIN}" - "${ROOT}/checkpoint" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "checkpoint_manifest.json").read_text(encoding="utf-8"))
assert manifest["schema_version"] == "1.5.0"
assert manifest["stage"] == "shared-planner"
assert manifest["planner_conditioning"] == "shared_farm_only"
assert manifest["farm_addition_count"] == 1
assert manifest["q7_gold_used"] is False
events = [
    json.loads(line)
    for line in (root / "training_history.jsonl").read_text(encoding="utf-8").splitlines()
]
for phase in ("base", "farm", "joint"):
    assert len([row for row in events if row["phase"] == phase]) == 2
joint = [row for row in events if row["phase"] == "joint"]
assert joint[-1]["mean_abs_personal_contrib"] > 0
PY

echo "PASS" >"${ROOT}/smoke_result.txt"
echo "V13_2_SHARED_FARM_SMOKE_PASS output=${ROOT}"
