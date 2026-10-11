import unittest

from tools.prompt_baselines.run_rulebank_ablation_pilot import (
    _relation_active_and_strength,
    build_prompt,
    derange,
    normalized_edit_similarity,
    parse_response,
    render_routes,
)


class RulebankAblationPilotTests(unittest.TestCase):
    def test_derangement_has_no_fixed_points(self):
        ids = [f"D{i:03d}" for i in range(30)]
        mapping = derange(ids, 17)
        self.assertEqual(set(mapping), set(ids))
        self.assertTrue(all(pid != donor for pid, donor in mapping.items()))

    def test_relation_activation_and_strength(self):
        relation = {
            "relation_type": "left_dominant",
            "left_trait": {"trait_index": 0},
            "right_trait": {"trait_index": 1},
        }
        active, strength = _relation_active_and_strength([6, 2] + [3] * 28, relation)
        self.assertTrue(active)
        self.assertEqual(strength, 3)

    def test_prompt_renders_initial_rule_without_profile_scores(self):
        routes = [{
            "layer": "q1_single",
            "rule_id": "R1",
            "action": {"initial_state": "KtoK", "semantic_operation": "推导限制"},
        }]
        rendered = render_routes(routes, "seg1")
        self.assertIn("第一反应优先采用 KtoK", rendered)
        self.assertNotIn("分数", rendered)
        system, user = build_prompt("seg1_q1_single", routes)
        self.assertIn("80-180", system)
        self.assertIn("个体化认知操作软路由", user)

    def test_response_parser_and_kc_nes(self):
        sequence, answer = parse_response('{"state_sequence":["K","CtoK"],"nl_answer":"文本"}')
        self.assertEqual(sequence, ["K", "CtoK"])
        self.assertEqual(answer, "文本")
        self.assertEqual(normalized_edit_similarity(sequence, sequence), 1.0)


if __name__ == "__main__":
    unittest.main()
