#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 CHECKOUT DATASET MODEL SEMANTIC_MODEL V7_SYNTHESIS OUTPUT_ROOT smoke|formal" >&2
  exit 2
fi

CHECKOUT="$(cd "$1" && pwd)"
DATASET="$(cd "$2" && pwd)"
MODEL="$(cd "$3" && pwd)"
SEMANTIC_MODEL="$(cd "$4" && pwd)"
V7_SYNTHESIS="$(cd "$(dirname "$5")" && pwd)/$(basename "$5")"
OUTPUT_ROOT="$6"
MODE="$7"
GPU="${GPU:-7}"
PYTHON_BIN="${PYTHON_BIN:-/data/wy/roleplay_smoke_runtime_v2/venv/bin/python}"

if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "mode must be smoke or formal" >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "output root must not already exist: $OUTPUT_ROOT" >&2
  exit 2
fi
if [[ ! -f "$V7_SYNTHESIS" ]]; then
  echo "V7 synthesis file is missing: $V7_SYNTHESIS" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python runtime is missing or not executable: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/tmp"
export PYTHONPATH="$CHECKOUT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$OUTPUT_ROOT/tmp"
export TEMP="$OUTPUT_ROOT/tmp"
export TMP="$OUTPUT_ROOT/tmp"

METHOD="$CHECKOUT/route_a/configs/realizer_unified_v3.json"
if [[ "$MODE" == "smoke" ]]; then
  METHOD="$CHECKOUT/route_a/configs/realizer_unified_smoke_v3.json"
fi

set +e
CUDA_VISIBLE_DEVICES="$GPU" \
  "$PYTHON_BIN" -m route_a.training.multitask_realizer_train \
    --dataset-dir "$DATASET" \
    --base-config "$CHECKOUT/route_a/configs/training_v13_3_delta_sequence_farm.json" \
    --method-config "$METHOD" \
    --model-path "$MODEL" \
    --semantic-model-path "$SEMANTIC_MODEL" \
    --v7-synthesis-path "$V7_SYNTHESIS" \
    --output-dir "$OUTPUT_ROOT/run" \
    --outer-fold 0 \
    --inner-fold 0 \
    --device-index 0 \
    --seed 42 \
    --qlora \
    >"$OUTPUT_ROOT/training.log" 2>&1
STATUS=$?
set -e

printf '%s\n' "$STATUS" >"$OUTPUT_ROOT/exit_status.txt"
"$PYTHON_BIN" - "$OUTPUT_ROOT" "$MODE" "$GPU" "$STATUS" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": "1.0.0",
    "mode": sys.argv[2],
    "physical_gpu": int(sys.argv[3]),
    "exit_status": int(sys.argv[4]),
    "status": "PASS" if int(sys.argv[4]) == 0 else "FAIL",
    "q7_gold_used": False,
}
(root / "launcher_status.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
exit "$STATUS"
