import unittest

import numpy as np

import diagnostic_utils as d


class DiagnosticTests(unittest.TestCase):
    def test_interface(self):
        for name in ('geometry', 'crossing_fraction', 'quantile_bins', 'target_metrics',
                     'spearman', 'within_group_rank_correlation'):
            self.assertTrue(callable(getattr(d, name, None)), name)

    def test_geometry_mean_radius_not_outer_radius(self):
        h=np.array([[0.,0.],[2.,0.],[4.,0.],[0.,5.]])
        a=np.array([0,0,0,1])
        dist,radius,ratio,size=d.geometry(h,a)
        np.testing.assert_allclose(dist,[2,0,2,0])
        np.testing.assert_allclose(radius,[4/3,4/3,4/3,0])
        np.testing.assert_allclose(ratio,[1.5,0,1.5,0])
        np.testing.assert_array_equal(size,[3,3,3,1])

    def test_crossing_counts_only_available_votes(self):
        a=np.array([0,0,1])
        neighbors=np.array([[1,2],[0,2],[0,1]])
        support=np.array([1,2,0])
        got=d.crossing_fraction(a,neighbors,support)
        np.testing.assert_allclose(got[:2],[0,.5])
        self.assertTrue(np.isnan(got[2]))

    def test_bins_do_not_artificially_split_ties(self):
        bins,edges=d.quantile_bins(np.ones(8),np.ones(8,dtype=bool))
        self.assertEqual(len(edges),1)
        self.assertEqual(len(np.unique(bins)),1)

    def test_brier_and_softening_are_not_wrong_class(self):
        target=np.array([[.6,.4],[.5,.5],[.3,.7]])
        y=np.array([0,0,0])
        m=d.target_metrics(target,y,y)
        np.testing.assert_allclose(m['brier'],[.32,.5,.98])
        np.testing.assert_array_equal(m['strict_wrong'],[False,False,True])
        np.testing.assert_array_equal(m['truth_tied'],[False,True,False])
        self.assertTrue(m['softened'].all())

    def test_rank_sign_and_constant(self):
        self.assertAlmostEqual(d.spearman(np.arange(6),-np.arange(6)),-1.)
        self.assertIsNone(d.spearman(np.ones(6),np.arange(6)))

    def test_within_group_removes_constant_group_predictor(self):
        self.assertIsNone(d.within_group_rank_correlation(np.array([1,1,2,2]),
                                                        np.array([1,2,3,4]),np.array([0,0,1,1])))


if __name__=='__main__':
    unittest.main(verbosity=2)
