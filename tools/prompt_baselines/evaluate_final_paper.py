"""Evaluate the final active4-only paper cohort using the original metric functions.

Run from repository root: python -m tools.prompt_baselines.evaluate_final_paper
This performs no model/API calls. BGE needs the original encoder/cache separately.
"""
import argparse
import json
from pathlib import Path

from tools.prompt_baselines.evaluate_main_method_r2 import (
    load_gold, read_jsonl, sequence_metric_rows, summarize_sequences,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'Qwen3-8B/Ours': 'full162_ours_rules_qwen3_8b_final.jsonl',
    'Llama-3.1-8B/Ours': 'full162_ours_rules_llama_3_1_8b_instruct_final.jsonl',
    'GPT-4o/Ours': 'full162_ours_rules_gpt4o_final.jsonl',
    'GPT-4o/NoRules': 'full162_ours_no_trait_rules_gpt4o.jsonl',
    'GPT-4o/Full30': 'full162_ours_plus_full30_gpt4o_final.jsonl',
    'GPT-4o/NoQ2': 'full162_ours_no_q2_gpt4o_final.jsonl',
    'GPT-4o/NoActive4': 'full162_ours_no_active4_gpt4o_final.jsonl',
    'GPT-4o/ShuffledRules': 'full162_ours_shuffled_rules_gpt4o_final.jsonl',
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/paper_sequence_metrics.json')
    args = parser.parse_args()
    gold, _, _ = load_gold(ROOT/'processed_data/q7_human_gold_adjudicated_v2.json',
                          ROOT/'processed_data/route_a_training_v1/private/user_id_mapping.json')
    assert len(gold) == 162
    summaries = {}
    for label, filename in SOURCES.items():
        raw = read_jsonl(ROOT/'outputs/main_method_r4_aligned_v1'/filename)
        assert len(raw) == 162 and {r['target_id'] for r in raw} == set(gold), label
        normalized = []
        for row in raw:
            assert not row.get('q7_gold_used_for_target_generation'), label
            assert row['target_id'] not in row.get('q7_icl_ids', []), label
            normalized.append({'condition': label, 'target_id': row['target_id'],
                               'sequence': row.get('state_sequence'), 'text': row.get('nl_answer'),
                               'generation_ok': row['ok'], 'error': row.get('error')})
        summaries[label] = summarize_sequences(sequence_metric_rows(normalized, gold))
        print(label, summaries[label]['n'], summaries[label]['unique_sequences'],
              round(summaries[label]['pairwise_kc_diversity_retention'], 4), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'cohort': 162, 'sources': SOURCES,
        'bge': 'Not recomputed: requires original BGE encoder and gold cache',
        'conditions': summaries}, ensure_ascii=False, indent=2)+'\n')

if __name__ == '__main__':
    main()
