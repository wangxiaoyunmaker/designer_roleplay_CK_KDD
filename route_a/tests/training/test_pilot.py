import unittest

from route_a.training.pilot import choose_stop_fraction


class StopPilotTests(unittest.TestCase):
    def test_selects_smallest_setting_passing_both_gates(self):
        selected, audit = choose_stop_fraction({
            0.1275: {"rollout_gate_passed": False, "macro_f1_six_states": 0.7},
            0.2: {"rollout_gate_passed": True, "macro_f1_six_states": 0.695},
            0.25: {"rollout_gate_passed": True, "macro_f1_six_states": 0.71},
        })
        self.assertEqual(selected, 0.2)
        self.assertFalse(audit["q7_gold_used"])

    def test_rejects_incomplete_search(self):
        with self.assertRaises(ValueError):
            choose_stop_fraction({0.2: {}})

    def test_six_state_f1_cannot_be_hidden_by_stop_performance(self):
        with self.assertRaisesRegex(ValueError, "no STOP pilot setting"):
            choose_stop_fraction({
                0.1275: {"rollout_gate_passed": False, "macro_f1_six_states": 0.8},
                0.2: {"rollout_gate_passed": True, "macro_f1_six_states": 0.1},
                0.25: {"rollout_gate_passed": False, "macro_f1_six_states": 0.8},
            })


if __name__ == "__main__":
    unittest.main()
