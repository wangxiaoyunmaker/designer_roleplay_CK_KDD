import importlib.util
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch is only required on the training server")
class SharedFarmModelTests(unittest.TestCase):
    def setUp(self):
        import torch
        from route_a.training.torch_models import SharedFaRM

        self.torch = torch
        self.SharedFaRM = SharedFaRM
        torch.manual_seed(7)
        self.semantics = torch.randn(7, 4096)
        self.context = torch.randn(3, 4096)
        self.v30 = torch.randn(3, 30)
        self.structural = torch.randn(3, 7, 67)
        self.allowed = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
        ], dtype=torch.bool)

    def test_shared_centering_parameter_budget_and_gradient(self):
        model = self.SharedFaRM(self.semantics, variant="shared")
        output = model(self.context, self.v30, self.structural, self.allowed)
        self.assertEqual(tuple(output.centered_logits.shape), (3, 7))
        for row, mask in zip(output.centered_logits, self.allowed):
            self.assertAlmostEqual(float(row[mask].mean().item()), 0.0, places=5)
            self.assertTrue(bool((row[~mask] == 0).all().item()))
        parameters = sum(parameter.numel() for parameter in model.parameters())
        self.assertLess(parameters, 500_000)
        self.assertTrue(bool((output.centered_logits == 0).all().item()))
        targets = self.torch.tensor([0, 1, 2])
        logits = output.centered_logits.masked_fill(~self.allowed, -1e9)
        self.torch.nn.functional.cross_entropy(logits, targets).backward()
        self.assertIsNotNone(model.interaction_projection.weight.grad)
        self.assertGreater(
            float(model.interaction_projection.weight.grad.abs().sum().item()), 0.0
        )

    def test_residual_warmup_multiplier_is_bounded(self):
        model = self.SharedFaRM(self.semantics)
        model.set_residual_multiplier(0.25)
        self.assertAlmostEqual(float(model.alpha.item()), 0.025, places=5)
        with self.assertRaises(ValueError):
            model.set_residual_multiplier(1.1)

    def test_nonpersonal_variants_ignore_v30(self):
        for variant in ("structural", "semantic"):
            with self.subTest(variant=variant):
                model = self.SharedFaRM(self.semantics, variant=variant)
                model.eval()
                left = model(
                    self.context, self.v30, self.structural, self.allowed
                ).centered_logits
                right = model(
                    self.context, self.v30.flip(0), self.structural, self.allowed
                ).centered_logits
                self.assertTrue(bool(self.torch.allclose(left, right)))

    def test_invalid_mask_is_rejected(self):
        model = self.SharedFaRM(self.semantics)
        invalid = self.allowed.clone()
        invalid[0] = False
        with self.assertRaises(ValueError):
            model(self.context, self.v30, self.structural, invalid)

    def test_independent_branch_scales_are_centered_and_bounded(self):
        model = self.SharedFaRM(
            self.semantics,
            variant="shared",
            independent_scales=True,
            max_structural_scale=2.0,
            max_semantic_scale=2.0,
            max_personal_scale=5.0,
            initial_structural_scale=1.0,
            initial_semantic_scale=1.0,
            initial_personal_scale=1.0,
        )
        output = model(
            self.context, self.v30, self.structural, self.allowed
        )
        for branch in (
            output.structural_centered,
            output.semantic_centered,
            output.personal_centered,
        ):
            for row, mask in zip(branch, self.allowed):
                self.assertAlmostEqual(
                    float(row[mask].mean().item()), 0.0, places=5
                )
        self.assertAlmostEqual(
            float(output.alpha_structural.item()), 1.0, places=5
        )
        self.assertAlmostEqual(
            float(output.alpha_semantic.item()), 1.0, places=5
        )
        self.assertAlmostEqual(
            float(output.alpha_personal.item()), 1.0, places=5
        )
        expected = (
            output.alpha_structural * output.structural_centered
            + output.alpha_semantic * output.semantic_centered
            + output.alpha_personal * output.personal_centered
        )
        self.assertTrue(
            bool(self.torch.allclose(output.joint_residual, expected))
        )


if __name__ == "__main__":
    unittest.main()
