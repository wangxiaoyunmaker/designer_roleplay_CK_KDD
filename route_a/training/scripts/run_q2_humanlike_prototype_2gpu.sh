#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 CHECKOUT DATASET MODEL SEMANTIC_MODEL OUTPUT_ROOT" >&2
  exit 2
fi

CHECKOUT="$(cd "$1" && pwd)"
DATASET="$(cd "$2" && pwd)"
MODEL="$(cd "$3" && pwd)"
SEMANTIC_MODEL="$(cd "$4" && pwd)"
OUTPUT_ROOT="$5"
PYTHON_BIN="${PYTHON_BIN:-/data/wy/roleplay_smoke_runtime_v2/venv/bin/python}"
GPU_LOW="${GPU_LOW:-3}"
GPU_HIGH="${GPU_HIGH:-7}"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "output root already exists: $OUTPUT_ROOT" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "runtime is not executable: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/tmp"
export PYTHONPATH="$CHECKOUT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$OUTPUT_ROOT/tmp"
export TEMP="$OUTPUT_ROOT/tmp"
export TMP="$OUTPUT_ROOT/tmp"

run_arm() {
  local gpu="$1"
  local name="$2"
  local config="$3"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m route_a.training.multitask_realizer_train \
    --dataset-dir "$DATASET" \
    --allow-draft-uncleaned \
    --base-config "$CHECKOUT/route_a/configs/training_v13_3_delta_sequence_farm.json" \
    --method-config "$config" \
    --model-path "$MODEL" \
    --semantic-model-path "$SEMANTIC_MODEL" \
    --output-dir "$OUTPUT_ROOT/$name" \
    --outer-fold 0 --inner-fold 0 --device-index 0 --seed 42 --qlora \
    >"$OUTPUT_ROOT/logs/$name.log" 2>&1
  local status=$?
  set -e
  printf '%s\n' "$status" >"$OUTPUT_ROOT/logs/$name.exit_status.txt"
  return "$status"
}

run_arm "$GPU_LOW" lr3e5 "$CHECKOUT/route_a/configs/realizer_q2_humanlike_prototype_lr3e5.json" &
PID_LOW=$!
run_arm "$GPU_HIGH" lr6e5 "$CHECKOUT/route_a/configs/realizer_q2_humanlike_prototype_lr6e5.json" &
PID_HIGH=$!
printf '%s\n' "$PID_LOW" >"$OUTPUT_ROOT/logs/lr3e5.pid"
printf '%s\n' "$PID_HIGH" >"$OUTPUT_ROOT/logs/lr6e5.pid"

LOW=0
HIGH=0
wait "$PID_LOW" || LOW=$?
wait "$PID_HIGH" || HIGH=$?
"$PYTHON_BIN" - "$OUTPUT_ROOT" "$LOW" "$HIGH" "$GPU_LOW" "$GPU_HIGH" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": "1.0.0",
    "status": "PASS" if int(sys.argv[2]) == 0 and int(sys.argv[3]) == 0 else "FAIL",
    "prototype_scope": "Q2-selected humanlike generation only; not a Q7 effectiveness claim",
    "exit_status": {"lr3e5": int(sys.argv[2]), "lr6e5": int(sys.argv[3])},
    "physical_gpus": {"lr3e5": int(sys.argv[4]), "lr6e5": int(sys.argv[5])},
    "q7_gold_used": False,
}
(root / "launcher_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

[[ "$LOW" -eq 0 && "$HIGH" -eq 0 ]]
