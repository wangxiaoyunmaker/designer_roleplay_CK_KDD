#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/data/wy/main_method_r2_ablation_v1
PYTHON_BIN=/data/wy/roleplay_smoke_runtime_v2/venv/bin/python
MODEL_PATH=/data/wy/roleplay_smoke_runtime_v2/hf/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
REALIZER_ROOT=/data/wy/route_a_v13_6_unified_checkout
SCRIPT=tools/prompt_baselines/run_main_method_r2_counterfactual.py
TARGET_IDS=diagnostics/counterfactual_v1/rulebank_ablation_pilot_v1_participants.json

cd "$RUN_ROOT"

COMMON=(
  --raw-data-root input
  --instrument input/design_task_instrument_q1_q8.json
  --participants input/participants.jsonl
  --planner input/planner_records.jsonl
  --uid-map input/user_id_mapping.json
  --gold input/q7_human_gold_adjudicated_v2.json
  --realizer-root "$REALIZER_ROOT"
  --long-lexicon input/v30_value_profile_lexicon_v1.json
  --compact-lexicon input/v30_value_profile_short_lexicon_v1.json
  --relevance-map input/v13_8_q2_relevance_v1.json
  --semantic-prior input/q2_semantic_value_relevance_v2.json
  --q1-rules input/q1_rules.json
  --q2-single-rules input/q2_single_rules.json
  --q2-double-rules input/q2_double_rules.json
  --aux-rules input/rules_by_arm.json
  --conditions c_dual
  --target-ids "$TARGET_IDS"
  --model "$MODEL_PATH"
)

nohup env \
  TMPDIR="$RUN_ROOT/tmp" \
  TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/cache/torchinductor" \
  CUDA_VISIBLE_DEVICES=6 \
  "$PYTHON_BIN" "$SCRIPT" "${COMMON[@]}" \
  --static-profile-assignment shuffled \
  --output-condition-label c_static_shuffled \
  --output diagnostics/counterfactual_v1/c_static_shuffled.jsonl \
  >diagnostics/counterfactual_v1/c_static_shuffled.log 2>&1 &
echo "c_static_shuffled pid=$! gpu=6"

nohup env \
  TMPDIR="$RUN_ROOT/tmp" \
  TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/cache/torchinductor" \
  CUDA_VISIBLE_DEVICES=7 \
  "$PYTHON_BIN" "$SCRIPT" "${COMMON[@]}" \
  --dynamic-profile-assignment shuffled \
  --frozen-sequences output/formal_v13_c_dual.jsonl \
  --output-condition-label c_dynamic_shuffled_frozen \
  --output diagnostics/counterfactual_v1/c_dynamic_shuffled_frozen.jsonl \
  >diagnostics/counterfactual_v1/c_dynamic_shuffled_frozen.log 2>&1 &
echo "c_dynamic_shuffled_frozen pid=$! gpu=7"
