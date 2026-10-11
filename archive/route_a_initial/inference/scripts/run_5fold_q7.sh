#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 DATASET_DIR MODEL_PATH TRAINING_ROOT OUTPUT_ROOT INSTRUMENT [SEED]" >&2
  exit 2
fi

DATASET_DIR="$1"
MODEL_PATH="$2"
TRAINING_ROOT="$3"
OUTPUT_ROOT="$4"
INSTRUMENT="$5"
SEED="${6:-42}"
TRAINING_CONFIG="route_a/configs/training_v1.json"
INFERENCE_CONFIG="route_a/configs/inference_v1.json"
ONTOLOGY="route_a/configs/state_ontology_v2.json"

if [[ "${DATASET_DIR}" == *"/draft" ]]; then
  echo "formal Q7 launcher refuses a draft dataset" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}"

run_fold() {
  local fold="$1"
  local gpu="$2"
  local train="${TRAINING_ROOT}/seed${SEED}/fold${fold}"
  local calibration="${OUTPUT_ROOT}/calibration/fold${fold}"
  local prediction="${OUTPUT_ROOT}/fold${fold}"

  CUDA_VISIBLE_DEVICES="${gpu}" python -m route_a.inference calibrate-best-of-n \
    --dataset-dir "${DATASET_DIR}" \
    --training-config "${TRAINING_CONFIG}" \
    --inference-config "${INFERENCE_CONFIG}" \
    --outer-fold "${fold}" \
    --planner-checkpoint "${train}/planner-selection" \
    --realizer-checkpoint "${train}/realizer-selection" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${calibration}" \
    --device-index 0 --run-seed "${SEED}"

  CUDA_VISIBLE_DEVICES="${gpu}" python -m route_a.inference generate-q7 \
    --dataset-dir "${DATASET_DIR}" \
    --training-config "${TRAINING_CONFIG}" \
    --inference-config "${INFERENCE_CONFIG}" \
    --instrument "${INSTRUMENT}" \
    --outer-fold "${fold}" \
    --planner-checkpoint "${train}/planner-refit" \
    --realizer-checkpoint "${train}/realizer-refit" \
    --model-path "${MODEL_PATH}" \
    --selection-calibration "${calibration}" \
    --output-dir "${prediction}" \
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
  echo "at least one Q7 fold failed; inspect ${OUTPUT_ROOT}/fold*.log" >&2
  exit 1
fi

python -m route_a.inference merge-q7-folds \
  --fold-dir "${OUTPUT_ROOT}/fold0" \
  --fold-dir "${OUTPUT_ROOT}/fold1" \
  --fold-dir "${OUTPUT_ROOT}/fold2" \
  --fold-dir "${OUTPUT_ROOT}/fold3" \
  --fold-dir "${OUTPUT_ROOT}/fold4" \
  --dataset-dir "${DATASET_DIR}" \
  --ontology "${ONTOLOGY}" \
  --output-dir "${OUTPUT_ROOT}/merged"

echo "five Q7 folds calibrated, generated, verified, merged, and resealed"
