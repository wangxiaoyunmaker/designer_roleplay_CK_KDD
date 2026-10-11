#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m tools.prompt_baselines.run_main_method_r4_aligned \
  --raw-data-root raw_data/questionnaire_v3 \
  --instrument raw_data/design_task_instrument_q1_q8.json \
  --participants data/route_a_training_v1/formal/participants.jsonl \
  --planner data/route_a_training_v1/formal/planner_records.jsonl \
  --uid-map processed_data/route_a_training_v1/private/user_id_mapping.json \
  --gold processed_data/q7_human_gold_adjudicated_v2.json \
  --realizer-root . \
  --long-lexicon route_a/configs/v30_value_profile_lexicon_v1.json \
  --compact-lexicon route_a/configs/v30_value_profile_short_lexicon_v1.json \
  --relevance-map route_a/configs/v13_10_q2_relevance_all_v30_v1.json \
  --semantic-prior artifacts/q7_main_r4_public_semantic_prior_v1.json \
  --q2-single-rules artifacts/kc_operational_ontology_v1_uniform/rules.json \
  --q2-double-rules artifacts/kc_v30_relation_ontology_v1_uniform/rules.json \
  --auxiliary-rules artifacts/kc_aux_preference_sweep_v1_uniform/rules_by_arm.json \
  --conditions ours --segments seg2 --start-target D001 "$@"
