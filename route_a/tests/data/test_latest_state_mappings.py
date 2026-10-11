import unittest
from pathlib import Path

from route_a.data.build import _source_statistics
from route_a.data.records import build_aux_records, build_planner_records
from route_a.data.sources import load_sources


class LatestStateMappingTests(unittest.TestCase):
    def test_latest_instrument_mappings_preserve_v1_and_v2_transitions(self):
        project_root = Path(__file__).resolve().parents[3]
        raw_root = Path("/Users/wy/Desktop/KDD")
        if not (raw_root / "raw_data/design_task_instrument_q1_q8.json").is_file():
            self.skipTest("latest private KDD raw data is unavailable")
        bundle = load_sources(
            raw_root,
            project_root / "route_a/configs/choice_state_mapping_v2.json",
            project_root / "route_a/configs/data_build_v2.json",
        )
        stats = _source_statistics(bundle, build_aux_records(bundle), build_planner_records(bundle))
        self.assertEqual(stats["q2_kc_steps"], 1188)
        self.assertEqual(stats["q2_kc_state_counts"]["KtoK"], 110)
        self.assertEqual(stats["q2_removed_v"], 94)
        self.assertEqual(stats["aux_records"], 771)
