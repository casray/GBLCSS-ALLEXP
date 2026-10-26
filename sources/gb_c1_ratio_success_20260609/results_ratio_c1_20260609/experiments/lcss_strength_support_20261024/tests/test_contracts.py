import importlib
from pathlib import Path
import sys
import unittest

import torch
torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Contracts(unittest.TestCase):
    def test_implementation_present(self):
        self.assertIsNotNone(importlib.util.find_spec('targets'), 'target kernels not implemented')

    def test_target_and_controls(self):
        from targets import evidence, shrink, mass_direction, match_mass, b2_targets
        z = torch.tensor([0, 1, 1, 0, 2, 2, 0, 1])
        a = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
        e = evidence(z, a, 3)
        own = torch.nn.functional.one_hot(z, 3).float()
        self.assertTrue(torch.equal(shrink(e, 0), own))
        self.assertTrue(torch.allclose(shrink(e, 1), e['native'], atol=1e-7))
        self.assertTrue(torch.allclose(e['native'].sum(1), torch.ones(len(z))))
        self.assertEqual(float(e['m'][-1]), 0)
        t = shrink(e, 1, mass_direction(e))
        self.assertTrue(torch.allclose((1-t.gather(1, z[:, None]).squeeze()).double(), e['m'], atol=1e-7))
        for w in [torch.zeros(8), torch.ones(8), torch.linspace(0, 1, 8)]:
            r = match_mass(e['m'], w, torch.flip(w, [0]))
            self.assertLess(abs(float((e['m'].double()*(r.double()-w.double())).sum())), 1e-7)
            self.assertTrue(bool(((r >= 0) & (r <= 1)).all()))
        es = {k: evidence(z, (torch.arange(8)//k), 3) for k in (2, 4, 8)}
        arms, j = b2_targets({8: es[2], 32: es[4], 128: es[8]})
        for name in ['D8', 'M8', 'DM8', 'D128', 'M128', 'DM128']:
            self.assertTrue(torch.equal(arms[name][~j], arms['N32'][~j]))

    def test_selection_contract(self):
        from selection import probe_survivors, balanced, validate_score
        rows = [dict(arm=f'S{k}_L{l}', support=k, strength=l, valid_correct=k % 7,
                     valid_total=20, valid_nll=1., best_epoch=10, epoch=25)
                for k in [4, 8, 16, 32, 64, 128] for l in [.25, .5, 1.]]
        kept = probe_survivors(rows, 6, 25)
        self.assertEqual(len(kept), 6)
        self.assertIn('S32_L1.0', kept)
        with self.assertRaises(ValueError):
            probe_survivors([dict(rows[0], epoch=200)], 1, 25)
        with self.assertRaises(ValueError):
            validate_score(dict(rows[0], test_accuracy=.9))
        self.assertEqual(balanced({0: [.8]*4, .25: [.801]*4, 1: [.802]*4}), 0)

    def test_meta_split_and_gradient(self):
        from protocol import split_validation, check_time
        from meta_gate import gate_features, meta_step
        from targets import evidence
        vi = torch.arange(20, 27)
        y = torch.tensor([0, 0, 0, 1, 1, 2, 3])
        mi, si = split_validation(y)
        self.assertEqual(len(set(mi.tolist()) & set(si.tolist())), 0)
        self.assertEqual(sorted(torch.cat([mi, si]).tolist()), list(range(7)))
        self.assertTrue({5, 6}.issubset(set(si.tolist())))
        with self.assertRaises(RuntimeError):
            check_time('2026-10-23T20:34:59+00:00')
        torch.manual_seed(3)
        head = torch.nn.Linear(4, 3)
        gate = torch.nn.Linear(9, 1)
        torch.nn.init.zeros_(gate.weight)
        torch.nn.init.zeros_(gate.bias)
        x = torch.randn(9, 4)
        z = torch.tensor([0, 1, 2, 0, 1, 0])
        e = evidence(z, torch.zeros(6).long(), 3)
        ft = gate_features(e, head(x[:6]).detach().softmax(1))
        opt = torch.optim.Adam(gate.parameters(), lr=.001)
        value, norm = meta_step(head, gate, opt, x[:6], x[6:], torch.tensor([0, 1, 2]), e, ft)
        self.assertGreater(norm, 0)
        self.assertTrue(torch.isfinite(torch.tensor(value)))

    def test_training_whitelist_and_graph_fold(self):
        from protocol import training_data
        from selection import dataset_fold
        with self.assertRaises(ValueError):
            training_data(dict(x=torch.ones(2, 2), test_y=torch.zeros(2)))
        rows = [dict(dataset=n, candidate=i) for n in ['a', 'b', 'c'] for i in range(5)]
        train, test = dataset_fold(rows, 'b')
        self.assertEqual(len(train), 10)
        self.assertEqual({r['dataset'] for r in test}, {'b'})

    def test_exclusive_lock(self):
        from protocol import acquire_lock, ROOT
        p = ROOT/'tests/test.lock'
        try:
            acquire_lock(p)
            with self.assertRaises(FileExistsError):
                acquire_lock(p)
        finally:
            p.unlink(missing_ok=True)

    def test_fitter_present(self):
        self.assertIsNotNone(importlib.util.find_spec('fit'), 'paired fitter not implemented')

    def test_leave_one_out_and_ties(self):
        from targets import evidence, shrink, match_mass
        z = torch.tensor([0, 1, 2, 1])
        e = evidence(z, torch.zeros(4).long(), 3)
        altered = z.clone()
        altered[0] = 2
        f = evidence(altered, torch.zeros(4).long(), 3)
        self.assertTrue(torch.equal(e['q'][0], f['q'][0]))
        self.assertEqual(float(e['alpha'][0]), float(f['alpha'][0]))
        self.assertTrue(bool(e['tied'][1]))
        self.assertTrue(torch.equal(e['q'][1], e['vote'][1]))
        single = evidence(torch.tensor([0, 1]), torch.tensor([0, 1]), 3)
        self.assertTrue(torch.equal(shrink(single, 1), single['own']))
        self.assertTrue(torch.equal(match_mass(torch.zeros(2), torch.ones(2), torch.ones(2)), torch.zeros(2).double()))

    def test_gate_real_target_detached(self):
        from targets import evidence
        from meta_gate import gate_target, gate_features
        e = evidence(torch.tensor([0, 1, 0, 2]), torch.zeros(4).long(), 3)
        p = torch.randn(4, 3, requires_grad=True).softmax(1)
        ft = gate_features(e, p)
        self.assertFalse(ft.requires_grad)
        w = torch.ones(4, requires_grad=True)*.5
        self.assertTrue(gate_target(e, w).requires_grad)
        self.assertFalse(gate_target(e, w).detach().requires_grad)


if __name__ == '__main__':
    unittest.main()
