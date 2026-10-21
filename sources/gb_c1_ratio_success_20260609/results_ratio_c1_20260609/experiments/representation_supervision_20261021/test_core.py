from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))


class CoreTests(unittest.TestCase):
    def core(self):
        import core
        return core

    def test_loo_and_tie(self):
        c = self.core()
        z = torch.tensor([0, 1, 1, 2])
        t, alpha = c.make_targets(z, torch.zeros(4, dtype=torch.long), 3)
        expected_r = torch.tensor([0., 2/3, 1/3])
        expected_q = .5 * expected_r + .5 * F.one_hot(torch.tensor(1), 3)
        torch.testing.assert_close(t[0], .5 * F.one_hot(z[0], 3) + .5 * expected_q)
        self.assertEqual(alpha[1].item(), 0.)
        torch.testing.assert_close(t[1], F.one_hot(z[1], 3).float())

    def test_loo_is_independent_of_own_vote(self):
        c = self.core()
        z = torch.tensor([0, 1, 1, 2])
        z2 = z.clone(); z2[0] = 2
        t, a = c.make_targets(z, torch.zeros(4, dtype=torch.long), 3)
        t2, a2 = c.make_targets(z2, torch.zeros(4, dtype=torch.long), 3)
        torch.testing.assert_close(a[0], a2[0])
        q = (t[0] - (1-a[0])*F.one_hot(z[0], 3))/a[0]
        q2 = (t2[0] - (1-a2[0])*F.one_hot(z2[0], 3))/a2[0]
        torch.testing.assert_close(q, q2)

    def test_insufficient_votes_and_legacy_difference(self):
        c = self.core()
        z = torch.tensor([0, 1])
        b = torch.zeros(2, dtype=torch.long)
        t, a = c.make_targets(z, b, 3)
        torch.testing.assert_close(t, F.one_hot(z, 3).float())
        self.assertEqual(a.sum().item(), 0.)
        legacy, _ = c.make_targets(z, b, 3, leave_one_out=False, tie_safe=False)
        self.assertFalse(torch.equal(t, legacy))

    def test_target_rejects_nontraining_sentinel(self):
        with self.assertRaises(ValueError):
            self.core().make_targets(torch.tensor([0, -1]), torch.zeros(2, dtype=torch.long), 3)

    def test_backbone_matches_reference_and_rng(self):
        c = self.core()
        from gblcss.models import MLP
        torch.manual_seed(7)
        expected = MLP(8, 64, 3, 2, .5)
        after = torch.get_rng_state().clone()
        torch.manual_seed(7)
        model = c.AlignedMLP(8, 3, 5)
        torch.testing.assert_close(torch.get_rng_state(), after)
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(model.backbone.state_dict()[key], value)
        x = torch.randn(9, 8)
        model.eval(); expected.eval()
        torch.testing.assert_close(model(x)[0], expected(x), rtol=0, atol=0)

    def test_target_alignment_detached(self):
        c = self.core()
        projected = torch.randn(7, 5, requires_grad=True)
        target = torch.randn(7, 5, requires_grad=True)
        loss = c.alignment_loss(projected, target)
        loss.backward()
        self.assertIsNotNone(projected.grad)
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(projected.grad).all())

    def test_reference_small_loss_schedule(self):
        c = self.core()
        self.assertEqual(c.keep_count(100, 0, .4), 96)
        self.assertEqual(c.keep_count(100, 9, .4), 60)
        self.assertEqual(c.keep_count(100, 199, 0), 100)
        self.assertEqual(c.keep_count(1, 199, .6), 1)
        self.assertEqual(c.keep_count(0, 0, .4), 0)

    def test_small_loss_uses_soft_ce_and_selected_mean(self):
        c = self.core()
        logits = torch.tensor([[5., 0.], [0., 5.], [2., 1.]], requires_grad=True)
        target = torch.tensor([[1., 0.], [.5, .5], [1., 0.]])
        loss, selected = c.supervised_loss(logits, target, 'T3', 199, .4)
        losses = -(target * logits.log_softmax(1)).sum(1)
        torch.testing.assert_close(loss, losses.min())
        self.assertEqual(selected.tolist(), [0])

    def test_frozen_encoder_includes_bn_buffers(self):
        c = self.core()
        model = c.AlignedMLP(8, 3, 5)
        model.freeze_encoder()
        before = c.state_hash(model.encoder_state())
        opt = torch.optim.Adam(model.classifier_parameters(), lr=.01)
        for _ in range(3):
            model.train_head()
            logits, _ = model(torch.randn(12, 8))
            opt.zero_grad(); logits.square().mean().backward(); opt.step()
        self.assertEqual(before, c.state_hash(model.encoder_state()))
        self.assertTrue(all(p.grad is None for p in model.encoder_parameters()))

    def test_deployment_exports_only_original_mlp(self):
        c = self.core()
        from gblcss.models import MLP
        model = c.AlignedMLP(8, 3, 5).eval()
        exported = model.deployment_state()
        reference = MLP(8, 64, 3, 2, .5).eval()
        reference.load_state_dict(exported, strict=True)
        self.assertFalse(any('projection' in key for key in exported))
        x = torch.randn(11, 8)
        torch.testing.assert_close(model(x)[0], reference(x), atol=0, rtol=0)

    def test_validation_selection_uses_validation_only(self):
        c = self.core()
        rows = [dict(weight=.1, validation=.7, test=.9), dict(weight=1., validation=.7, test=.1)]
        self.assertEqual(c.select_weight(rows), .1)
        rows[0]['test'], rows[1]['test'] = 0., 1.
        self.assertEqual(c.select_weight(rows), .1)

    def test_alignment_does_not_use_labels(self):
        c = self.core()
        p = torch.randn(12, 7)
        h = torch.randn(12, 7)
        expected = (1-F.cosine_similarity(p, h, dim=1)).mean()
        torch.testing.assert_close(c.alignment_loss(p, h), expected)

    def test_original_target_function_parity(self):
        c = self.core()
        functions = c.reference_target_functions()
        torch.manual_seed(3)
        z = torch.randint(0, 5, (71,))
        assignment = torch.arange(71)//8
        for loo in (True, False):
            for safe in (True, False):
                actual, _ = c.make_targets(z, assignment, 5, leave_one_out=loo, tie_safe=safe)
                expected, _, _ = functions['apply_ball_soft_targets'](
                    z, assignment.numpy(), torch.arange(len(z)), 5, alpha_baseline='multiclass',
                    target_mode='distribution', target_beta=.5, leave_one_out=loo,
                    tie_mode='distribution_only' if safe else 'min_index')
                torch.testing.assert_close(actual, expected, atol=2e-7, rtol=0)

    def test_original_selection_schedule_all_epochs(self):
        c = self.core()
        original = c.reference_remember_count()
        for n in (1, 541, 3958):
            for rate in (0., .4, .6):
                for epoch in range(200):
                    self.assertEqual(c.keep_count(n, epoch, rate), original(n, epoch, rate))


if __name__ == '__main__':
    unittest.main()
