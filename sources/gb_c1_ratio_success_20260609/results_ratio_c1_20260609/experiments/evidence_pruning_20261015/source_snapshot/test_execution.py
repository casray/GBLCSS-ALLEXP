import itertools
import unittest
import numpy as np
import common as c
from gblcss.granular_ball_pruning import optimal_pruning


class ExecutionTests(unittest.TestCase):
    def test_real_execution_entry_imports_project_seed_function(self):
        import run
        from gblcss.data import set_seed
        self.assertIs(run.set_seed,set_seed)

    def test_float32_cache_identity_matches_training_tensor(self):
        a=np.array([[.2,.8]],dtype=np.float64);b=a+1e-12
        self.assertEqual(c.target_key(a),c.target_key(b))
        self.assertNotEqual(c.target_key(a),c.target_key(a+.001))

    def test_dynamic_program_matches_exhaustive_cuts(self):
        children={0:(1,2),1:(3,4),2:(5,6)}
        cuts=[[0],[1,2],[3,4,2],[1,5,6],[3,4,5,6]]
        for values in itertools.product([0.,1.,3.],repeat=7):
            costs=dict(enumerate(values));cut,cost=optimal_pruning(children,costs)
            self.assertEqual(cost,min(sum(costs[n] for n in x) for x in cuts))
            self.assertIn(cut,cuts)


if __name__=='__main__':unittest.main()
