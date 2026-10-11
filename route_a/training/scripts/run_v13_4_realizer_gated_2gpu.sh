#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 CHECKOUT DATASET MODEL SEMANTIC_MODEL OUTPUT_ROOT smoke|formal" >&2
  exit 2
fi

CHECKOUT="$(cd "$1" && pwd)"
DATASET="$(cd "$2" && pwd)"
MODEL="$(cd "$3" && pwd)"
SEMANTIC_MODEL="$(cd "$4" && pwd)"
OUTPUT_ROOT="$5"
MODE="$6"
Q2_GPU="${Q2_GPU:-0}"
MULTI_GPU="${MULTI_GPU:-1}"

if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "mode must be smoke or formal" >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "output root must not already exist: $OUTPUT_ROOT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/tmp"
export PYTHONPATH="$CHECKOUT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$OUTPUT_ROOT/tmp"
export TEMP="$OUTPUT_ROOT/tmp"
export TMP="$OUTPUT_ROOT/tmp"

BASE_CONFIG="$CHECKOUT/route_a/configs/training_v13_3_delta_sequence_farm.json"
if [[ "$MODE" == "smoke" ]]; then
  Q2_CONFIG="$CHECKOUT/route_a/configs/realizer_q2_only_smoke_v2.json"
  MULTI_CONFIG="$CHECKOUT/route_a/configs/realizer_multitask_smoke_v2.json"
else
  Q2_CONFIG="$CHECKOUT/route_a/configs/realizer_q2_only_v2.json"
  MULTI_CONFIG="$CHECKOUT/route_a/configs/realizer_multitask_v2.json"
fi

run_variant() {
  local gpu="$1"
  local name="$2"
  local method_config="$3"
  local output="$OUTPUT_ROOT/$name"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" \
    python -m route_a.training.multitask_realizer_train \
      --dataset-dir "$DATASET" \
      --base-config "$BASE_CONFIG" \
      --method-config "$method_config" \
      --model-path "$MODEL" \
      --semantic-model-path "$SEMANTIC_MODEL" \
      --output-dir "$output" \
      --outer-fold 0 \
      --inner-fold 0 \
      --device-index 0 \
      --seed 42 \
      --qlora \
      >"$OUTPUT_ROOT/logs/$name.log" 2>&1
  local status=$?
  set -e
  echo "$status" >"$OUTPUT_ROOT/logs/$name.exit_status.txt"
  return "$status"
}

run_variant "$Q2_GPU" q2_only "$Q2_CONFIG" &
PID_Q2=$!
run_variant "$MULTI_GPU" q2_plus_aux "$MULTI_CONFIG" &
PID_MULTI=$!
echo "$PID_Q2" >"$OUTPUT_ROOT/logs/q2_only.pid"
echo "$PID_MULTI" >"$OUTPUT_ROOT/logs/q2_plus_aux.pid"

Q2_STATUS=0
MULTI_STATUS=0
wait "$PID_Q2" || Q2_STATUS=$?
wait "$PID_MULTI" || MULTI_STATUS=$?

python - "$OUTPUT_ROOT" "$MODE" "$Q2_STATUS" "$MULTI_STATUS" "$Q2_GPU" "$MULTI_GPU" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
mode = sys.argv[2]
statuses = {
    "q2_only": int(sys.argv[3]),
    "q2_plus_aux": int(sys.argv[4]),
}
payload = {
    "schema_version": "1.0.0",
    "mode": mode,
    "status": "PASS" if all(value == 0 for value in statuses.values()) else "FAIL",
    "exit_status": statuses,
    "physical_gpus": {
        "q2_only": int(sys.argv[5]),
        "q2_plus_aux": int(sys.argv[6]),
    },
    "selection_protocol": "heldout_q2_gated_v1",
    "q7_gold_used": False,
}
(root / "launcher_status.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

if [[ "$Q2_STATUS" -ne 0 || "$MULTI_STATUS" -ne 0 ]]; then
  exit 1
fi
