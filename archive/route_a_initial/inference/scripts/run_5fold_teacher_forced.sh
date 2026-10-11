#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 || $# -gt 8 ]]; then
  echo "usage: $0 DATASET_DIR MODEL_PATH TRAINING_ROOT MERGED_PREDICTION_DIR GOLD OUTPUT_ROOT INSTRUMENT [SEED]" >&2
  exit 2
fi

DATASET_DIR="$1"
MODEL_PATH="$2"
TRAINING_ROOT="$3"
PREDICTION_DIR="$4"
GOLD="$5"
OUTPUT_ROOT="$6"
INSTRUMENT="$7"
SEED="${8:-42}"
TRAINING_CONFIG="route_a/configs/training_v1.json"
INFERENCE_CONFIG="route_a/configs/inference_v1.json"
ONTOLOGY="route_a/configs/state_ontology_v2.json"
mkdir -p "${OUTPUT_ROOT}"

run_fold() {
  local fold="$1"
  local gpu="$2"
  local train="${TRAINING_ROOT}/seed${SEED}/fold${fold}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -m route_a.inference export-q7-teacher-forced \
    --dataset-dir "${DATASET_DIR}" \
    --training-config "${TRAINING_CONFIG}" \
    --inference-config "${INFERENCE_CONFIG}" \
    --instrument "${INSTRUMENT}" \
    --outer-fold "${fold}" \
    --planner-checkpoint "${train}/planner-refit" \
    --model-path "${MODEL_PATH}" \
    --predictions "${PREDICTION_DIR}/q7_predictions_v1.jsonl" \
    --prediction-checksum "${PREDICTION_DIR}/q7_predictions_v1.sha256" \
    --gold "${GOLD}" \
    --system-id route_a_full \
    --output-dir "${OUTPUT_ROOT}/fold${fold}" \
    --device-index 0 --run-seed "${SEED}"
}

pids=()
for fold in 0 1 2 3 4; do
  run_fold "${fold}" "${fold}" >"${OUTPUT_ROOT}/fold${fold}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "at least one teacher-forced fold failed; inspect logs" >&2
  exit 1
fi

python -m route_a.inference merge-teacher-forced \
  --fold-dir "${OUTPUT_ROOT}/fold0" \
  --fold-dir "${OUTPUT_ROOT}/fold1" \
  --fold-dir "${OUTPUT_ROOT}/fold2" \
  --fold-dir "${OUTPUT_ROOT}/fold3" \
  --fold-dir "${OUTPUT_ROOT}/fold4" \
  --dataset-dir "${DATASET_DIR}" \
  --ontology "${ONTOLOGY}" \
  --output-dir "${OUTPUT_ROOT}/merged"

echo "teacher-forced probabilities exported post-seal and merged"
