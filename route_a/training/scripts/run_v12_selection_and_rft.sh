#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 8 || $# -gt 10 ]]; then
  echo "usage: $0 DATASET_DIR MODEL_PATH FINAL_FOLD_ROOT OUTER_FOLD BGE_MODEL BGE_MANIFEST INFERENCE_CONFIG FINAL_SEED [TRAINING_CONFIG] [METHOD_FOLD_ROOT]" >&2
  exit 2
fi

DATASET_DIR="$1"
MODEL_PATH="$2"
FINAL_ROOT="$3"
FOLD="$4"
BGE_MODEL="$5"
BGE_MANIFEST="$6"
INFERENCE_CONFIG="$7"
FINAL_SEED="$8"
CONFIG="${9:-route_a/configs/training_v12.json}"
METHOD_ROOT="${10:-${FINAL_ROOT}}"
METHOD="${METHOD_ROOT}/method-selection"

case "${FINAL_SEED}" in
  17|42|2026) ;;
  *) echo "FINAL_SEED must be one of 17, 42, 2026" >&2; exit 2 ;;
esac

for CHECKPOINT in \
  "${FINAL_ROOT}/realizer-refit" \
  "${METHOD_ROOT}/planner-provisional" \
  "${METHOD_ROOT}/realizer-provisional" \
  "${METHOD_ROOT}/realizer-oof-0" \
  "${METHOD_ROOT}/realizer-oof-1" \
  "${METHOD_ROOT}/realizer-oof-2"; do
  if [[ ! -f "${CHECKPOINT}/checkpoint_manifest.json" ]]; then
    echo "required checkpoint is incomplete: ${CHECKPOINT}" >&2
    exit 2
  fi
done
mkdir -p "${METHOD}"
mkdir -p "${METHOD}/.tmp"
export TMPDIR="${METHOD}/.tmp"

run_if_missing() {
  local marker="$1"
  shift
  if [[ -f "${marker}" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP completed ${marker}"
  else
    "$@"
  fi
}

run_if_missing "${METHOD}/calibration-inputs/generation_manifest.json" \
  python -m route_a.inference generate-v12-calibration-inputs \
    --dataset-dir "${DATASET_DIR}" --training-config "${CONFIG}" \
    --inference-config "${INFERENCE_CONFIG}" --outer-fold "${FOLD}" \
    --planner-checkpoint "${METHOD_ROOT}/planner-provisional" \
    --realizer-checkpoint "${METHOD_ROOT}/realizer-provisional" \
    --model-path "${MODEL_PATH}" --bge-model-path "${BGE_MODEL}" \
    --bge-snapshot-manifest "${BGE_MANIFEST}" --run-seed 42 \
    --output-dir "${METHOD}/calibration-inputs"

run_if_missing "${METHOD}/best-of-n-calibration/best_of_n_calibration.json" \
  python -m route_a.inference freeze-best-of-n \
    --dataset-dir "${DATASET_DIR}" --training-config "${CONFIG}" \
    --outer-fold "${FOLD}" --planner-checkpoint "${METHOD_ROOT}/planner-provisional" \
    --realizer-checkpoint "${METHOD_ROOT}/realizer-provisional" \
    --consistency-pairs "${METHOD}/calibration-inputs/consistency_pairs.jsonl" \
    --selection-metrics "${METHOD}/calibration-inputs/selection_metrics.jsonl" \
    --output-dir "${METHOD}/best-of-n-calibration"

for INNER in 0 1 2; do
  run_if_missing "${METHOD}/rft-oof-${INNER}/generation_manifest.json" \
    python -m route_a.inference generate-rft-candidates \
      --dataset-dir "${DATASET_DIR}" --training-config "${CONFIG}" \
      --inference-config "${INFERENCE_CONFIG}" --outer-fold "${FOLD}" \
      --realizer-checkpoint "${METHOD_ROOT}/realizer-oof-${INNER}" \
      --source-model-role development_oof --inner-fold "${INNER}" \
      --model-path "${MODEL_PATH}" --bge-model-path "${BGE_MODEL}" \
      --bge-snapshot-manifest "${BGE_MANIFEST}" --run-seed 42 \
      --output-dir "${METHOD}/rft-oof-${INNER}"
done

run_if_missing "${METHOD}/rft-selection/generation_manifest.json" \
  python -m route_a.inference generate-rft-candidates \
    --dataset-dir "${DATASET_DIR}" --training-config "${CONFIG}" \
    --inference-config "${INFERENCE_CONFIG}" --outer-fold "${FOLD}" \
    --realizer-checkpoint "${METHOD_ROOT}/realizer-provisional" \
    --source-model-role development_provisional \
    --model-path "${MODEL_PATH}" --bge-model-path "${BGE_MODEL}" \
    --bge-snapshot-manifest "${BGE_MANIFEST}" --run-seed 42 \
    --output-dir "${METHOD}/rft-selection"

if [[ ! -f "${METHOD}/rft_candidates_merge.json" ]]; then
  python -m route_a.training merge-rft-candidates \
    --input-dir "${METHOD}/rft-oof-0" --input-dir "${METHOD}/rft-oof-1" \
    --input-dir "${METHOD}/rft-oof-2" --input-dir "${METHOD}/rft-selection" \
    --output-file "${METHOD}/rft_candidates_all.jsonl" \
    --report "${METHOD}/rft_candidates_merge.json"
fi

run_if_missing "${METHOD}/rft-data/rft_audit.json" \
  python -m route_a.training build-rft-data \
    --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
    --candidates "${METHOD}/rft_candidates_all.jsonl" \
    --output-dir "${METHOD}/rft-data"

run_if_missing "${METHOD}/realizer-rft-provisional/checkpoint_manifest.json" \
  python -m route_a.training train-rft \
    --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
    --model-path "${MODEL_PATH}" \
    --base-realizer-checkpoint "${METHOD_ROOT}/realizer-provisional" \
    --accepted-candidates "${METHOD}/rft-data/accepted_candidates.jsonl" \
    --rft-audit "${METHOD}/rft-data/rft_audit.json" --fit-scope provisional \
    --output-dir "${METHOD}/realizer-rft-provisional" --seed 42

if [[ ! -f "${METHOD}/rft_selection_metrics.manifest.json" ]]; then
  python -m route_a.inference evaluate-provisional-rft \
    --dataset-dir "${DATASET_DIR}" --training-config "${CONFIG}" \
    --inference-config "${INFERENCE_CONFIG}" --outer-fold "${FOLD}" \
    --base-realizer-checkpoint "${METHOD_ROOT}/realizer-provisional" \
    --rft-realizer-checkpoint "${METHOD}/realizer-rft-provisional" \
    --rft-audit "${METHOD}/rft-data/rft_audit.json" --model-path "${MODEL_PATH}" \
    --bge-model-path "${BGE_MODEL}" --bge-snapshot-manifest "${BGE_MANIFEST}" \
    --output-metrics "${METHOD}/rft_selection_metrics.json" --run-seed 42
fi

if [[ ! -f "${METHOD}/rft_gate_report.json" ]]; then
  python -m route_a.training gate-rft \
    --config "${CONFIG}" --metrics "${METHOD}/rft_selection_metrics.json" \
    --report "${METHOD}/rft_gate_report.json"
fi

if python -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["enabled"] else 1)' "${METHOD}/rft_gate_report.json"; then
  run_if_missing "${FINAL_ROOT}/realizer-rft-refit/checkpoint_manifest.json" \
    python -m route_a.training train-rft \
      --dataset-dir "${DATASET_DIR}" --config "${CONFIG}" --outer-fold "${FOLD}" \
      --model-path "${MODEL_PATH}" --base-realizer-checkpoint "${FINAL_ROOT}/realizer-refit" \
      --accepted-candidates "${METHOD}/rft-data/accepted_candidates.jsonl" \
      --rft-audit "${METHOD}/rft-data/rft_audit.json" \
      --rft-gate-report "${METHOD}/rft_gate_report.json" --fit-scope outer-refit \
      --output-dir "${FINAL_ROOT}/realizer-rft-refit" --seed "${FINAL_SEED}"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RFT_ENABLED fold=${FOLD} seed=${FINAL_SEED}"
else
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RFT_REJECTED_USE_STAGE_C fold=${FOLD} seed=${FINAL_SEED}"
fi
