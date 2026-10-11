import unittest

from route_a.state_ontology import load_kc_state_contract


class StateOntologyTests(unittest.TestCase):
    def test_kc_contract_is_six_state_and_excludes_v(self):
        order, contract = load_kc_state_contract()
        self.assertEqual(order, ("K", "C", "KtoC", "CtoK", "KtoK", "CtoC"))
        self.assertNotIn("V", contract)

    def test_transition_contracts_preserve_directional_criteria(self):
        _, contract = load_kc_state_contract()
        self.assertIn("向外查、问、测或要解释", contract["CtoK"]["core_criterion"])
        self.assertIn("已知→概念", contract["KtoC"]["core_criterion"])
        self.assertIn("已存在的方案", contract["CtoC"]["core_criterion"])
        self.assertIn("无新概念", contract["KtoK"]["core_criterion"])
