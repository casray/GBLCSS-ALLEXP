from __future__ import annotations

import importlib
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F


class GCNAdapterTests(unittest.TestCase):
    def test_adapter_exists(self):
        self.assertTrue((Path(__file__).parent / 'gcn_adapter.py').is_file(),
                        'Common-GCN TINED adapter must be implemented')

    def graph(self, module):
        edges = torch.tensor([[0, 1, 1, 2, 2, 3, 0, 0], [1, 0, 2, 1, 3, 2, 0, 1]])
        return module.GCNGraph(edges, 6, torch.float64)

    def test_common_gcn_and_transformation_trace(self):
        m = importlib.import_module('gcn_adapter')
        from gblcss.models import GCN
        graph = self.graph(m)
        for classes in (3, 70):
            teacher = m.GCNTeacher([80, 64, classes], .5).double().eval()
            baseline = GCN(80, 64, classes, 2, .5).double().eval()
            baseline.load_state_dict(teacher.backbone.state_dict())
            x = torch.randn(6, 80, dtype=torch.float64)
            torch.testing.assert_close(teacher(graph, x), baseline(x, graph.adjacency), rtol=0, atol=0)
            output, targets = teacher.forward_with_targets(graph, x)
            torch.testing.assert_close(output, teacher(graph, x), rtol=1e-10, atol=1e-10)
            h, manual_targets = x, []
            matrix = graph.normalized.to_dense()
            for index, layer in enumerate(teacher.backbone.layers):
                for operation in ('FT', 'GP') if layer.in_channels > layer.out_channels else ('GP', 'FT'):
                    before = h
                    h = F.linear(h, layer.lin.weight) if operation == 'FT' else matrix @ h
                    ein, eout = m.dirichlet_energy(graph, before), m.dirichlet_energy(graph, h)
                    manual_targets.append(m._log_ratio(ein, eout, 0.))
                h = h + layer.bias
                if index == 0:
                    h = F.relu(h)
            torch.testing.assert_close(targets, torch.stack(manual_targets), rtol=1e-10, atol=1e-10)

    def test_student_injection_and_order(self):
        m = importlib.import_module('gcn_adapter')
        for classes, expected in [(3, ['TeacherMLP', 'TeacherGA', 'TeacherMLP', 'TeacherGA']),
                                  (70, ['TeacherMLP', 'TeacherGA', 'TeacherGA', 'TeacherMLP'])]:
            teacher = m.GCNTeacher([80, 64, classes], .5)
            cfg = m.effective_config('corafull', True)
            student = m.TINEDGCNStudent(teacher, cfg)
            self.assertEqual(student.transformation_types_pseudo, expected)
            self.assertEqual(len(student.layers), 4)
            self.assertEqual(student.dropout.p, 0.)
            for source, target in zip(teacher.backbone.layers, student.learned_MLP_layers):
                torch.testing.assert_close(target.weight, source.lin.weight, rtol=0, atol=0)
                torch.testing.assert_close(target.bias, source.bias, rtol=0, atol=0)
                self.assertNotEqual(target.weight.data_ptr(), source.lin.weight.data_ptr())
            for layer in student.graph_aggregation_layers:
                torch.testing.assert_close(layer.weight, torch.eye(layer.out_features), rtol=0, atol=0)
            logits = student(torch.randn(6, 80))[1]
            self.assertEqual(logits.shape, (6, classes))

    def test_presets_and_validation_ties(self):
        m = importlib.import_module('gcn_adapter')
        for dataset, dropout in [('cora', 0.), ('citeseer', 0.), ('corafull', 0.),
                                 ('coauthorcs', 0.), ('amazonpho', .5)]:
            cfg = m.effective_config(dataset, False)
            self.assertEqual(cfg['teacher'], 'GCN')
            self.assertEqual(cfg['teacher_dropout'], .5)
            self.assertEqual(cfg['dropout'], dropout)
            self.assertEqual(cfg['teacher_epochs'], 200)
            self.assertEqual(cfg['student_epochs'], 200)
            self.assertEqual(cfg['learning_rate'], .01)
            self.assertEqual(cfg['weight_decay'], .0005)
            self.assertEqual(cfg['algorithm_preset_family'], 'SAGE-transductive')
        layer = torch.nn.Linear(2, 2)
        best = {'epoch': 1, 'val': .5}
        state, selected = m.select_checkpoint(layer, best, 2, .5, last_tie=False)
        self.assertIsNone(state)
        self.assertEqual(selected['epoch'], 1)
        state, selected = m.select_checkpoint(layer, best, 2, .5, last_tie=True)
        self.assertIsNotNone(state)
        self.assertEqual(selected['epoch'], 2)

    def test_dirichlet_energy_value_and_detached_norm_gradient(self):
        m = importlib.import_module('gcn_adapter')
        graph = self.graph(m)
        x = torch.randn(6, 7, dtype=torch.float64, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        z = y / y.norm(dim=1, keepdim=True).detach()
        src, dst = graph.edge_index
        expected = (z[src] - z[dst]).square().sum() / len(y)
        actual = m.dirichlet_energy(graph, x)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
