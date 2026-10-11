import unittest
from pathlib import Path

from route_a.training.config import load_training_config
from route_a.training.prompts import render_planner_prompt, render_realizer_prompt
from route_a.training.value_profile import (
    load_compact_value_profile_lexicon,
    load_value_profile_lexicon,
    render_compact_value_profile,
    render_value_profile,
)


ROOT = Path(__file__).resolve().parents[3]


class ValueProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lexicon = load_value_profile_lexicon(
            ROOT / "route_a/configs/v30_value_profile_lexicon_v1.json"
        )
        cls.definitions = {"K": "知识判断", "C": "情境权衡"}

    def test_approved_lexicon_renders_all_items_in_frozen_order(self):
        profile = render_value_profile([1, 2, 3, 4, 5, 6] * 5, self.lexicon)
        lines = profile.splitlines()
        self.assertEqual(len(lines), 30)
        self.assertTrue(lines[0].startswith("- 数据 vs 直觉：强烈倾向："))
        self.assertIn("AI 替代低技能岗位", lines[2])
        self.assertIn("技术支持", lines[3])
        self.assertTrue(lines[-1].startswith("- 救世主情结：强烈倾向："))

    def test_approved_compact_lexicon_renders_thirty_no_strength_phrases(self):
        compact = load_compact_value_profile_lexicon(
            ROOT / "route_a/configs/v30_value_profile_short_lexicon_v1.json",
        )
        profile = render_compact_value_profile(
            [1, 2, 3, 4, 5, 6] * 5,
            long_lexicon=self.lexicon,
            compact_lexicon=compact,
        )
        lines = profile.splitlines()
        self.assertEqual(len(lines), 30)
        self.assertEqual(lines[0], "- 重视外部证据")
        self.assertEqual(lines[3], "- 尽早整合新技术")
        self.assertEqual(lines[-1], "- 设计介入系统议题")
        self.assertNotIn("强烈倾向", profile)
        self.assertNotIn("倾向：", profile)

    def test_planner_and_realizer_receive_the_identical_profile_block(self):
        profile = render_value_profile([6] * 30, self.lexicon)
        planner = render_planner_prompt("公共情境", ["K"], self.definitions, profile)
        realizer = render_realizer_prompt(
            "公共情境", "PROCESS_SEQUENCE", ["K"], self.definitions, profile
        )
        expected = f"[DESIGN_VALUE_PROFILE]\n{profile}\n\n"
        self.assertIn(expected, planner)
        self.assertIn(expected, realizer)
        self.assertLess(planner.index(expected), planner.index("[STATE_LEGEND]"))
        self.assertLess(realizer.index(expected), realizer.index("[PROCESS_SEQUENCE]"))

    def test_v13_config_binds_the_reviewed_lexicon(self):
        config = load_training_config(ROOT / "route_a/configs/training_v13_value_profile.json")
        self.assertTrue(config.value_profile_enabled)
        self.assertEqual(config.value_profile_path, ROOT / "route_a/configs/v30_value_profile_lexicon_v1.json")


if __name__ == "__main__":
    unittest.main()
