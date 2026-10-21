from __future__ import annotations

import copy
import importlib
from pathlib import Path
import unittest

import torch
from torch import nn
from torch_geometric.data import Data


class MatchedTests(unittest.TestCase):
    def test_adapter_exists(self):
        self.assertTrue((Path(__file__).parent / 'matched_adapter.py').is_file(),
                        'Matched two-layer adapter must be implemented')

    def test_exact_common_backbone(self):
        adapter = importlib.import_module('matched_adapter')
        from gblcss.models import MLP
        torch.manual_seed(3)
        baseline = MLP(11, 64, 5, 2, .5)
        torch.manual_seed(3)
        model, head, _ = adapter.build_model(Data(x=torch.zeros(9, 11), num_classes=5),
                                            adapter.effective_config('cora', True))
        x = torch.randn(9, 11)
        self.assertEqual(sum(isinstance(m, nn.Linear) for m in model.encoder.modules()), 1)
        self.assertIsInstance(head, nn.Linear)
        deployed = sum(p.numel() for p in model.encoder.parameters()) + sum(p.numel() for p in head.parameters())
        self.assertEqual(deployed, sum(p.numel() for p in baseline.parameters()))
        for training in (True, False):
            baseline.train(training)
            model.encoder.train(training)
            torch.manual_seed(17)
            expected = baseline(x)
            torch.manual_seed(17)
            actual = head(model.encode(x))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(any(isinstance(m, nn.PReLU) for m in model.encoder.modules()))

    def test_fixed_configuration(self):
        adapter = importlib.import_module('matched_adapter')
        for dataset in ('cora', 'citeseer', 'corafull', 'coauthorcs', 'amazonpho'):
            cfg = adapter.effective_config(dataset, False)
            self.assertEqual(cfg['optimizer'], 'Adam')
            self.assertFalse(cfg['use_scheduler'])
            self.assertEqual(cfg['epochs'], 200)
            self.assertEqual(cfg['lr_epochs'], 200)
            self.assertEqual(cfg['learning_rate'], .01)
            self.assertEqual(cfg['weight_decay'], .0005)
            self.assertEqual(cfg['probe_selection'], 'clean validation accuracy; first tie')

    def test_self_supervision_and_frozen_probe(self):
        adapter = importlib.import_module('matched_adapter')
        cfg = adapter.effective_config('cora', True)
        data = Data(x=torch.rand(8, 11), edge_index=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                                                                [1, 2, 3, 4, 5, 6, 7, 0]]),
                    num_classes=3)
        model, head, obj = adapter.build_model(data, cfg)
        before = model.encoder.layer.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.0005)
        criterion = obj['Bootstrap'](cfg['eta'], aux_pos_ratio=cfg['aux_pos_ratio'])
        loss = obj['train'](model, criterion, data, None, optimizer, None,
                            cfg['aug_rounds'], cfg['recon_lambda'])
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(before, model.encoder.layer.weight))
        model.eval()
        model.encoder.requires_grad_(False)
        frozen = copy.deepcopy(model.encoder.state_dict())
        with torch.no_grad():
            embedding = model.encode(data.x).detach()
        optimizer = torch.optim.Adam(head.parameters(), lr=.01, weight_decay=.0005)
        nn.functional.cross_entropy(head(embedding), torch.arange(8) % 3).backward()
        optimizer.step()
        for key, value in frozen.items():
            torch.testing.assert_close(model.encoder.state_dict()[key], value, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None or not p.requires_grad for p in model.encoder.parameters()))


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
