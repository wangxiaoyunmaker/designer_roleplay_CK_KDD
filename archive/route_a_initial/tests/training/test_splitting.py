import unittest

from route_a.training.splitting import assign_inner_folds, selection_participants


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


if __name__ == "__main__":
    unittest.main()
