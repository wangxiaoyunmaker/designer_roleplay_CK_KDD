#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 5 || $# -gt 8 ]]; then
  echo "usage: $0 DATASET_DIR MODEL_PATH V10_ROOT V11_ROOT EVIDENCE_ROOT [GPU_A] [GPU_B] [POLL_SECONDS]" >&2
  exit 2
fi

DATASET_DIR="$1"
MODEL_PATH="$2"
V10_ROOT="$3"
V11_ROOT="$4"
EVIDENCE_ROOT="$5"
GPU_A="${6:-0}"
GPU_B="${7:-1}"
POLL_SECONDS="${8:-300}"

if [[ "${DATASET_DIR}" != *"/formal" ]]; then
  echo "V11 queue requires a formal dataset" >&2
  exit 2
fi
if ! [[ "${POLL_SECONDS}" =~ ^[0-9]+$ ]] || [[ "${POLL_SECONDS}" -lt 30 ]]; then
  echo "POLL_SECONDS must be an integer of at least 30" >&2
  exit 2
fi
if [[ -e "${V11_ROOT}" || -e "${EVIDENCE_ROOT}" ]]; then
  echo "V11 output/evidence root already exists; refusing to overwrite" >&2
  exit 2
fi

mkdir -p "${EVIDENCE_ROOT}"
while true; do
  if python -m route_a.workflows inspect-training \
      --training-root "${V10_ROOT}" --seeds 42 43 44 --require-complete \
      --report "${EVIDENCE_ROOT}/v10_matrix_latest.json" \
      >"${EVIDENCE_ROOT}/v10_preflight_latest.log" 2>&1; then
    printf '[%s] V10_COMPLETE_V11_START_ALLOWED\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >>"${EVIDENCE_ROOT}/events.log"
    break
  fi
  printf '[%s] WAITING_FOR_V10\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>"${EVIDENCE_ROOT}/events.log"
  sleep "${POLL_SECONDS}"
done

mkdir -p "${V11_ROOT}" "${EVIDENCE_ROOT}/logs"

worker() {
  local gpu="$1"
  local parity="$2"
  local index=0
  for seed in 42 43 44; do
    for fold in 0 1 2 3 4; do
      if (( index % 2 == parity )); then
        local target="${V11_ROOT}/seed${seed}/fold${fold}"
        local log="${EVIDENCE_ROOT}/logs/seed${seed}_fold${fold}.log"
        printf '[%s] START seed=%s fold=%s gpu=%s\n' \
          "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${seed}" "${fold}" "${gpu}" \
          >>"${EVIDENCE_ROOT}/events.log"
        if CUDA_VISIBLE_DEVICES="${gpu}" bash route_a/training/scripts/run_one_v11_fold.sh \
            "${DATASET_DIR}" "${MODEL_PATH}" "${target}" "${fold}" "${seed}" \
            >"${log}" 2>&1; then
          printf '[%s] COMPLETE seed=%s fold=%s gpu=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${seed}" "${fold}" "${gpu}" \
            >>"${EVIDENCE_ROOT}/events.log"
        else
          local status=$?
          printf '[%s] FAILED seed=%s fold=%s gpu=%s exit=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${seed}" "${fold}" "${gpu}" "${status}" \
            >>"${EVIDENCE_ROOT}/events.log"
          return "${status}"
        fi
      fi
      index=$((index + 1))
    done
  done
}

worker "${GPU_A}" 0 &
PID_A=$!
worker "${GPU_B}" 1 &
PID_B=$!
failed=0
wait "${PID_A}" || failed=1
wait "${PID_B}" || failed=1
if [[ "${failed}" -ne 0 ]]; then
  printf '[%s] V11_MATRIX_FAILED\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >>"${EVIDENCE_ROOT}/events.log"
  exit 1
fi

python -m route_a.workflows inspect-training \
  --training-root "${V11_ROOT}" --seeds 42 43 44 \
  --verify-artifacts --require-complete \
  --report "${EVIDENCE_ROOT}/v11_matrix_complete.json"
printf '[%s] V11_MATRIX_COMPLETE\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >>"${EVIDENCE_ROOT}/events.log"
