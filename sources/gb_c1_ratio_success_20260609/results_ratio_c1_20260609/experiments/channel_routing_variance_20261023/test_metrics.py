import unittest

import numpy as np

from metrics import pair_metrics, route_metrics, stage0_gate, protocol_cell


class MetricTests(unittest.TestCase):
    def test_pair_union_uses_supplied_models(self):
        y = np.array([0, 0, 0, 0])
        a200 = np.array([0, 1, 1, 1])
        a300 = np.array([0, 0, 1, 1])
        c = np.array([1, 1, 0, 1])
        self.assertEqual(pair_metrics(y, a200, c)['union'], .5)
        self.assertEqual(pair_metrics(y, a300, c)['union'], .75)
        self.assertEqual(pair_metrics(y, a300, c)['both_wrong'], .25)

    def test_routing_gain_counts_rescues_minus_harm(self):
        y = np.zeros(5, dtype=int)
        a = np.array([0, 1, 1, 0, 1])
        c = np.array([1, 0, 0, 0, 2])
        r = route_metrics(y, a, c, np.ones(5, dtype=bool))
        self.assertEqual(r['rescued'], 2)
        self.assertEqual(r['harmed'], 1)
        self.assertAlmostEqual(r['gain'], .2)
        self.assertAlmostEqual(r['accuracy'], .6)

    def test_gate_averages_conditions_before_counting_graphs(self):
        rows = [dict(dataset=n, condition=c, gap_pp=g) for n,g in [('cora', 2),('citeseer', 2),('corafull', -4)] for c in ['uniform0.4','uniform0.6','pair0.4']]
        self.assertTrue(stage0_gate(rows)['passed'])
        rows.extend([dict(dataset='corafull',condition='clean',gap_pp=100)])
        self.assertTrue(stage0_gate(rows)['passed'])
        for r in rows:
            if r['dataset']=='citeseer':
                r['gap_pp']=1.99
        self.assertFalse(stage0_gate(rows)['passed'])

    def test_protocol_seed_does_not_follow_student_seed(self):
        self.assertEqual(protocol_cell('cora', 0, 'clean'), ('cora',0,'clean'))
        self.assertEqual(protocol_cell('cora', 1, 'clean'), ('cora',0,'clean'))


if __name__ == '__main__':
    unittest.main()
