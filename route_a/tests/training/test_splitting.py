import unittest

from route_a.training.splitting import (
    assign_inner_folds,
    nested_selection_split,
    selection_participants,
)


class InnerSplittingTests(unittest.TestCase):
    def test_inner_folds_are_deterministic_order_independent_and_balanced(self):
        ids = [f"D{i:03d}" for i in range(1, 131)]
        first = assign_inner_folds(ids, "frozen", 3)
        second = assign_inner_folds(reversed(ids), "frozen", 3)
        self.assertEqual(first, second)
        counts = [sum(fold == index for fold in first.values()) for index in range(3)]
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_selection_is_disjoint_and_complete(self):
        ids = [f"D{i:03d}" for i in range(1, 20)]
        train, validation = selection_participants(
            ids, salt="frozen", n_folds=3, validation_fold=0
        )
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(set(train) | set(validation), set(ids))

    def test_nested_split_is_deterministic_complete_and_oof(self):
        ids = [f"D{i:03d}" for i in range(1, 130)]
        first = nested_selection_split(
            ids, salt="frozen", development_fraction=0.8, n_folds=3
        )
        second = nested_selection_split(
            reversed(ids), salt="frozen", development_fraction=0.8, n_folds=3
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first.development_train), 103)
        self.assertEqual(len(first.selection_validation), 26)
        self.assertFalse(
            set(first.development_train) & set(first.selection_validation)
        )
        self.assertEqual(
            set(first.development_train) | set(first.selection_validation), set(ids)
        )
        for fold in range(3):
            train = first.inner_train(fold)
            held_out = first.inner_validation(fold)
            self.assertFalse(set(train) & set(held_out))
            self.assertEqual(set(train) | set(held_out), set(first.development_train))

    def test_nested_split_rejects_selection_val_as_inner_fold(self):
        split = nested_selection_split(
            [f"D{i:03d}" for i in range(20)], salt="frozen"
        )
        self.assertFalse(set(split.selection_validation) & set(split.development_fold))


if __name__ == "__main__":
    unittest.main()
