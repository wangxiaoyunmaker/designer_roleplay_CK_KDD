#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 10 || $# -gt 12 ]]; then
  echo "usage: $0 DATASET MODEL INSTRUMENT FINAL_FOLD_ROOT FOLD OUTPUT BGE_MODEL BGE_MANIFEST INFERENCE_CONFIG SEED [TRAINING_CONFIG] [METHOD_FOLD_ROOT]" >&2
  exit 2
fi

DATASET="$1"
MODEL="$2"
INSTRUMENT="$3"
FINAL_ROOT="$4"
FOLD="$5"
OUTPUT="$6"
BGE_MODEL="$7"
BGE_MANIFEST="$8"
INFERENCE_CONFIG="$9"
SEED="${10}"
TRAINING_CONFIG="${11:-route_a/configs/training_v12.json}"
METHOD_ROOT="${12:-${FINAL_ROOT}}"
METHOD="${METHOD_ROOT}/method-selection"
REALIZER="${FINAL_ROOT}/realizer-refit"
RFT_ARGS=()

case "${SEED}" in
  17|42|2026) ;;
  *) echo "SEED must be one of 17, 42, 2026" >&2; exit 2 ;;
esac

if [[ -f "${FINAL_ROOT}/realizer-rft-refit/checkpoint_manifest.json" ]]; then
  if ! python -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["enabled"] else 1)' "${METHOD}/rft_gate_report.json"; then
    echo "RFT checkpoint exists but gate is not enabled" >&2
    exit 2
  fi
  REALIZER="${FINAL_ROOT}/realizer-rft-refit"
  RFT_ARGS=(--rft-gate-report "${METHOD}/rft_gate_report.json")
fi

python -m route_a.inference generate-q7 \
  --dataset-dir "${DATASET}" --training-config "${TRAINING_CONFIG}" \
  --inference-config "${INFERENCE_CONFIG}" --instrument "${INSTRUMENT}" \
  --outer-fold "${FOLD}" --planner-checkpoint "${FINAL_ROOT}/planner-refit" \
  --realizer-checkpoint "${REALIZER}" --model-path "${MODEL}" \
  --selection-calibration "${METHOD}/best-of-n-calibration" \
  --bge-model-path "${BGE_MODEL}" --bge-snapshot-manifest "${BGE_MANIFEST}" \
  --output-dir "${OUTPUT}" --run-seed "${SEED}" "${RFT_ARGS[@]}"
