from types import SimpleNamespace
from pathlib import Path
import unittest

from route_a.training.config import load_training_config
from route_a.training.gpu_train import _bind_refit_to_selection, _fit_participants
from route_a.training.prepare import prepare_fold
from route_a.training.splitting import nested_selection_split


ROOT = Path(__file__).resolve().parents[3]


class FixedFitScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_training_config(ROOT / "route_a/configs/training_v12.json")
        cls.prepared = prepare_fold(
            ROOT / "data/route_a_training_v1/formal", cls.config, outer_fold=0
        )
        selection = cls.config.raw["selection"]
        cls.split = nested_selection_split(
            cls.prepared.train_participants,
            salt=f"{selection['split_salt']}|outer=0",
            development_fraction=selection["development_fraction"],
            n_folds=selection["inner_folds"],
        )

    def test_oof_refit_never_returns_target_fold_as_validation(self):
        for fold in range(3):
            args = SimpleNamespace(
                fit_mode="oof-refit", inner_fold=fold, selected_updates=10
            )
            train, validation, updates = _fit_participants(
                args, self.config, self.prepared
            )
            self.assertIsNone(validation)
            self.assertEqual(updates, 10)
            self.assertEqual(set(train), set(self.split.inner_train(fold)))
            self.assertFalse(set(train) & set(self.split.inner_validation(fold)))

    def test_provisional_refit_does_not_validate_on_selection_set(self):
        args = SimpleNamespace(
            fit_mode="provisional", inner_fold=None, selected_updates=10
        )
        train, validation, updates = _fit_participants(
            args, self.config, self.prepared
        )
        self.assertIsNone(validation)
        self.assertEqual(updates, 10)
        self.assertEqual(set(train), set(self.split.development_train))
        self.assertFalse(set(train) & set(self.split.selection_validation))

    def test_oof_refit_binds_frozen_updates_for_all_stage_names(self):
        for stage, config_section in (
            ("aux", "auxiliary"),
            ("planner", "planner"),
            ("realizer", "realizer"),
        ):
            with self.subTest(stage=stage):
                args = SimpleNamespace(fit_mode="oof-refit", inner_fold=0)
                _bind_refit_to_selection(
                    args, stage, self.config, self.prepared
                )
                self.assertEqual(
                    args.selected_updates,
                    self.config.raw[config_section]["max_updates"],
                )
                self.assertIsNone(args.selection_checkpoint_manifest_sha256)
                self.assertIsNone(args.selection_checkpoint_manifest_sha256s)
                self.assertEqual(args.stop_bias_offset, 0.0)


if __name__ == "__main__":
    unittest.main()
