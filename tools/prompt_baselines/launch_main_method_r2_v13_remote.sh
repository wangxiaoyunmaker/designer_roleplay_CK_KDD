#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/data/wy/main_method_r2_ablation_v1
PYTHON_BIN=/data/wy/roleplay_smoke_runtime_v2/venv/bin/python
MODEL_PATH=/data/wy/roleplay_smoke_runtime_v2/hf/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
REALIZER_ROOT=/data/wy/route_a_v13_6_unified_checkout

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
  --model "$MODEL_PATH"
)

launch_condition() {
  local condition=$1
  local gpu=$2
  local output="output/formal_v13_${condition}.jsonl"
  local log="logs/formal_v13_${condition}.log"
  nohup env \
    TMPDIR="$RUN_ROOT/tmp" \
    TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/cache/torchinductor" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON_BIN" tools/prompt_baselines/run_main_method_r2_ablation.py \
      "${COMMON[@]}" \
      --conditions "$condition" \
      --output "$output" \
      >"$log" 2>&1 &
  echo "$condition pid=$! gpu=$gpu output=$output log=$log"
}

launch_condition b_dynamic 0
launch_condition c_dual 1
launch_condition d_rules 2
launch_condition e_near 3
launch_condition f_far 4
