from pathlib import Path
import importlib
import sys
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


class AdapterTests(unittest.TestCase):
    def adapter(self):
        self.assertTrue((ROOT / 'adapter.py').is_file(), 'matched adapter not implemented')
        return importlib.import_module('adapter')

    def test_common_forward_and_two_fc(self):
        a = self.adapter()
        cfg = a.effective_config('cora')
        torch.manual_seed(3)
        model = a.MatchedContrast(9, 64, 3, cfg)
        reference = a.MLP(9, 64, 3, 2, .5)
        reference.load_state_dict(model.backbone.state_dict())
        x = torch.randn(12, 9)
        for training in (False, True):
            model.train(training)
            reference.train(training)
            torch.manual_seed(7)
            expected = reference(x)
            torch.manual_seed(7)
            actual, auxiliary = model(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(auxiliary.shape, (12, 3))
        self.assertEqual(sum(isinstance(m, nn.Linear) for m in model.backbone.modules()), 2)
        self.assertEqual(model.inference_parameters(), sum(p.numel() for p in reference.parameters()))

    def test_auxiliary_is_trainable_and_not_deployed(self):
        a = self.adapter()
        model = a.MatchedContrast(9, 64, 3, a.effective_config('cora'))
        logits, auxiliary = model(torch.randn(8, 9))
        (logits.square().mean() + auxiliary.square().mean()).backward()
        for p in model.inf.parameters():
            self.assertIsNotNone(p.grad)
        self.assertGreater(sum(p.numel() for p in model.parameters()), model.inference_parameters())

    def test_budget_and_seed(self):
        a = self.adapter()
        for name in ('cora', 'citeseer', 'corafull', 'coauthorcs', 'amazonpho'):
            cfg = a.effective_config(name, seed=1)
            self.assertEqual(cfg['seed'], 1)
            self.assertEqual(cfg['phase_budgets'], dict(edge_pretrain=100, classifier_pretrain=100, main=100))
            self.assertEqual(cfg['supervised_epochs_total'], 200)
            self.assertIn('first tie', cfg['validation_selection'])
        self.assertEqual(a.effective_config('cora', smoke=True)['supervised_epochs_total'], 2)
        with self.assertRaises(ValueError):
            a.effective_config('cora', seed=5)

    def test_frozen_core_and_narrow_transformation(self):
        a = self.adapter()
        self.assertEqual(a.TRANSFORM_COUNTS, dict(first_tie=1, inference_diagnostic=1, inference_return=1))
        self.assertIs(a.run.__globals__['_train_classifier_epoch'], a.core._train_classifier_epoch)
        self.assertIs(a.run.__globals__['EdgeSampler'], a.core.EdgeSampler)
        self.assertIs(a.run.__globals__['MixupScale'], a.core.MixupScale)
        self.assertIs(a.core.NeighContrast, a.ORIGINAL_CLASS)


if __name__ == '__main__':
    unittest.main()
