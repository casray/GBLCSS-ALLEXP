"""CPU mathematical tests; optional CUDA smoke is controlled by the parent."""
from __future__ import annotations

import json
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Data


class TINEDTests(unittest.TestCase):
    def adapter(self):
        import tined_adapter
        return tined_adapter

    def graph(self, dtype=torch.float64):
        edges = torch.tensor([[0, 1, 1, 2, 3, 4, 5, 6, 7, 0, 2],
                              [1, 0, 2, 1, 4, 5, 6, 7, 0, 1, 2]])
        return self.adapter().SparseGraph(edges, 8, dtype)

    def test_adapter_exists(self):
        self.assertTrue(Path(__file__).with_name('tined_adapter.py').is_file(),
                        'TINED adapter has not been implemented')

    def test_sparse_propagation_energy_and_gradient(self):
        a, g = self.adapter(), self.graph()
        torch.manual_seed(0)
        x = torch.randn(8, 5, dtype=torch.float64, requires_grad=True)
        src, dst = g.edge_index
        expected = x.clone().index_add(0, dst, x[src])
        expected /= (torch.bincount(dst, minlength=8) + 1).unsqueeze(1)
        torch.testing.assert_close(g.propagate(x), expected)
        norm = x.norm(dim=1, keepdim=True).detach()
        z = x / torch.where(norm == 0, torch.ones_like(norm), norm)
        direct = (z[src] - z[dst]).square().sum() / 8
        energy = a.dirichlet_energy(g, x)
        torch.testing.assert_close(energy, direct)
        actual_grad, = torch.autograd.grad(energy, x, retain_graph=True)
        expected_grad, = torch.autograd.grad(direct, x)
        torch.testing.assert_close(actual_grad, expected_grad)
        self.assertEqual(float(a.dirichlet_energy(g, torch.zeros_like(x))), 0)

    def test_sequence_order_injection_copy_and_de(self):
        a = self.adapter()
        for dims in [(9, 64, 7), (9, 64, 70), (64, 64, 7)]:
            with self.subTest(dims=dims):
                torch.manual_seed(0)
                teacher = a.SAGETeacher(dims, .5)
                cfg = a.resolve_config({'epochs': 2}, 'corafull')
                student = a.TINEDStudent(teacher, cfg)
                self.assertEqual(len(student.layers), 4)
                expected = []
                for din, dout in zip(dims, dims[1:]):
                    expected.extend(['TeacherMLP', 'TeacherGA'] if din > dout
                                    else ['TeacherGA', 'TeacherMLP'])
                self.assertEqual(student.transformation_types_pseudo, expected)
                for source, target in zip(teacher.layers, student.learned_MLP_layers):
                    torch.testing.assert_close(source.fc_neigh.weight, target.weight)
                    torch.testing.assert_close(source.bias, target.bias)
                    self.assertNotEqual(source.fc_neigh.weight.data_ptr(), target.weight.data_ptr())
                for layer in student.graph_aggregation_layers:
                    torch.testing.assert_close(layer.weight, torch.eye(layer.out_features))
                    torch.testing.assert_close(layer.bias, torch.zeros(layer.out_features))
                g, x = self.graph(torch.float32), torch.randn(8, dims[0])
                teacher.eval()
                logits, targets = teacher.forward_with_targets(g, x)
                self.assertEqual(logits.shape, (8, dims[-1]))
                self.assertEqual(targets.shape, (4,))
                student.set_de_targets(targets)
                loss = a.de_loss(torch.stack(student.get_appoximated_DE(x, g)), targets, cfg)
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for layer in student.layers for p in layer.parameters()))
                for source, target in zip(teacher.layers, student.learned_MLP_layers):
                    saved = source.fc_neigh.weight.clone()
                    with torch.no_grad():
                        target.weight.add_(1)
                    torch.testing.assert_close(source.fc_neigh.weight, saved)

    def test_official_ast_energy_matches_port_and_cached_input(self):
        a, g = self.adapter(), self.graph(torch.float32)
        cfg = a.resolve_config({'epochs': 2}, 'corafull')
        torch.manual_seed(0)
        model = a.TINEDStudent(a.SAGETeacher((9, 64, 70), .5), cfg).eval()
        reference = copy.deepcopy(model)
        x = torch.randn(8, 9)

        def edge_energy(graph, feature):
            norm = feature.norm(dim=1, keepdim=True).detach()
            norm = torch.where(norm == 0, torch.ones_like(norm), norm)
            z = feature / norm
            src, dst = graph.edge_index
            return (z[src] - z[dst]).square().sum() / graph.num_nodes

        with patch.dict(a._OfficialSequence.get_appoximated_DE.__globals__,
                        {'compute_dirichlet_energy': edge_energy}):
            expected = torch.stack(a._OfficialSequence.get_appoximated_DE(reference, x, g))
        actual = torch.stack(model.get_appoximated_DE(x, g))
        cached = torch.stack(model.get_appoximated_DE(x, g, a.dirichlet_energy(g, x)))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(cached, actual, rtol=0, atol=0)
        expected.sum().backward()
        actual.sum().backward()
        for left, right in zip(model.layers.parameters(), reference.layers.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=2e-6)

    def test_teacher_trace_matches_direct_edge_math(self):
        a, g = self.adapter(), self.graph(torch.float32)
        teacher = a.SAGETeacher((9, 64, 70), .5).eval()
        x = torch.randn(8, 9)
        actual, targets = teacher.forward_with_targets(g, x)
        src, dst = g.edge_index

        def energy(h):
            norm = h.norm(dim=1, keepdim=True).detach()
            z = h / torch.where(norm == 0, torch.ones_like(norm), norm)
            return (z[src] - z[dst]).square().sum() / 8

        h, ratios = x, []
        for i, layer in enumerate(teacher.layers):
            # For this 9->64->70 architecture both layers are GP then FT.
            incoming = torch.zeros_like(h).index_add(0, dst, h[src])
            gp = (incoming + h) / (torch.bincount(dst, minlength=8) + 1).unsqueeze(1)
            ft = gp @ layer.fc_neigh.weight.T
            ratios.extend([(energy(gp) / energy(h)).log(), (energy(ft) / energy(gp)).log()])
            h = ft + layer.bias
            if i == 0:
                h = h.relu()
        torch.testing.assert_close(actual, h)
        torch.testing.assert_close(targets, torch.stack(ratios), rtol=1e-5, atol=1e-6)

    def test_fixed_presets_and_protocol(self):
        a = self.adapter()
        for name in ['cora', 'citeseer', 'corafull', 'coauthorcs', 'amazonpho']:
            cfg = a.resolve_config({}, name)
            self.assertEqual((cfg['teacher_epochs'], cfg['student_epochs']), (200, 200))
            self.assertEqual((cfg['hidden'], cfg['dropout'], cfg['learning_rate'], cfg['weight_decay']),
                             (64, .5, .01, .0005))
        cfg = a.resolve_config({}, 'corafull')
        self.assertEqual(cfg['author_preset'], 'cora')
        self.assertEqual((cfg['lamb'], cfg['DE_regularization_rate']), (1, 1))
        self.assertTrue(cfg['sqrt_DER'])
        for bad in [{'seed': 1}, {'hidden': 32}, {'dw': True},
                    {'position_encoding': True}, {'epochs': 0}, {'lamb': .2}]:
            with self.subTest(config=bad), self.assertRaises(ValueError):
                a.resolve_config(bad, 'cora')

    def synthetic(self, device='cpu'):
        torch.manual_seed(0)
        y = torch.full((8,), -1, dtype=torch.long)
        indices = [torch.tensor([2, 0, 1]), torch.tensor([5, 3]), torch.tensor([7, 4, 6])]
        y[indices[0]] = torch.tensor([1, 0, 1])
        masks = []
        for idx in indices:
            mask = torch.zeros(8, dtype=torch.bool)
            mask[idx] = True
            masks.append(mask)
        return Data(x=torch.randn(8, 9), edge_index=self.graph(torch.float32).edge_index, y=y,
                    train_index=indices[0], val_index=indices[1], test_index=indices[2],
                    train_mask=masks[0], val_mask=masks[1], test_mask=masks[2],
                    val_y=torch.tensor([1, 0]), num_classes=2, name='synthetic').to(device)

    def test_small_synthetic_phases_no_public_cpu_fallback(self):
        a, data = self.adapter(), self.synthetic()
        a.validate_data(data, require_cuda=False)
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            a.validate_data(data)
        cfg = a.resolve_config({'epochs': 2}, data.name)
        g = a.SparseGraph(data.edge_index, 8, data.x.dtype)
        teacher = a.SAGETeacher((9, 64, 2), .5)
        rows = []
        state, best = a.fit_teacher(teacher, g, data, cfg, rows.append)
        teacher.load_state_dict(state)
        teacher.eval()
        with torch.no_grad():
            logits, targets = teacher.forward_with_targets(g, data.x)
        student = a.TINEDStudent(teacher, cfg)
        student.set_de_targets(targets)
        state, best = a.fit_student(student, g, data, logits.log_softmax(1), cfg, rows.append)
        self.assertEqual([r['phase'] for r in rows], ['teacher', 'teacher', 'student', 'student'])
        for row in rows:
            self.assertTrue(torch.isfinite(torch.tensor(row['loss'])))
            self.assertIn('val', row)
            self.assertNotIn('test', row)
        student.load_state_dict(state)
        student.eval()
        with torch.no_grad():
            probs = student(data.x)[1].softmax(1)
        torch.testing.assert_close(probs.sum(1), torch.ones(8))
        self.assertEqual(best['val'], float((probs[data.val_index].argmax(1) == data.val_y).double().mean()))

    def test_parent_split_allows_unassigned_graph_context(self):
        a, data = self.adapter(), self.synthetic()
        data.test_index = data.test_index[:-1]
        data.test_mask.zero_()
        data.test_mask[data.test_index] = True
        a.validate_data(data, require_cuda=False)
        overlap = data.clone()
        overlap.test_index = torch.cat((overlap.test_index, overlap.train_index[:1]))
        overlap.test_mask[overlap.train_index[0]] = True
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            a.validate_data(overlap, require_cuda=False)

    def test_label_contract_and_last_tie(self):
        a, data = self.adapter(), self.synthetic()
        a.validate_data(data, require_cuda=False)
        bad = data.clone()
        bad.y[bad.test_index[0]] = 0
        with self.assertRaisesRegex(ValueError, 'non-training'):
            a.validate_data(bad, require_cuda=False)
        bad = data.clone()
        bad.val_y = torch.tensor([0])
        with self.assertRaises(ValueError):
            a.validate_data(bad, require_cuda=False)
        best = {'epoch': 0, 'val': -1.0}
        for epoch in [1, 2, 3]:
            _, best = a.select_checkpoint(torch.nn.Linear(2, 2), best, epoch, .5)
        self.assertEqual(best['epoch'], 3)

    @unittest.skipUnless(os.environ.get('TINED_GPU_SMOKE') == '1', 'parent controls GPU smoke')
    def test_cuda_api_artifacts_reload(self):
        a = self.adapter()
        with tempfile.TemporaryDirectory(prefix='tined-synthetic-') as folder:
            data = self.synthetic('cuda')
            result = a.run(data, {'epochs': 2}, folder)
            probs = torch.load(result['probabilities_file'], weights_only=True)
            ckpt = torch.load(Path(folder) / 'checkpoint.pt', weights_only=True)
            self.assertEqual(probs.shape, (8, 2))
            self.assertEqual(len((Path(folder) / 'epochs.jsonl').read_text().splitlines()), 4)
            teacher = a.SAGETeacher(ckpt['teacher_config']['dims'], .5).cuda()
            student = a.TINEDStudent(teacher, ckpt['config']).cuda()
            student.load_state_dict(ckpt['student_state'])
            student.eval()
            with torch.no_grad():
                actual = student(data.x)[1].softmax(1).cpu()
            torch.testing.assert_close(actual, probs, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
