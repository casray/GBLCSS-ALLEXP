import unittest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_curves_reviewed import audit_layout,aggregate_snapshots


class PlotLayoutTests(unittest.TestCase):
    def test_capped_seed_endpoints_remain_one_paired_three_seed_mean(self):
        import pandas as pd
        f=pd.DataFrame(dict(requested_balls=[32,32,32],actual_balls=[20,21,22],test_accuracy=[.3,.4,.5]))
        a=aggregate_snapshots(f,'test_accuracy').iloc[0]
        self.assertAlmostEqual(a.mean_actual,21)
        self.assertAlmostEqual(a['mean'],.4)
        self.assertAlmostEqual(a['std'],.1)
        self.assertEqual(a['count'],3)

    def test_out_of_view_log_ticks_are_not_visible_clipping(self):
        fig,ax=plt.subplots(figsize=(4,3))
        ax.set_position([.2,.2,.6,.6]);ax.set_yscale('log')
        ax.set_yticks([.001,10,100,1000,1e9]);ax.set_ylim(10,1000)
        self.assertEqual(audit_layout(fig),[])
        plt.close(fig)

    def test_real_outside_text_is_still_rejected(self):
        fig,ax=plt.subplots(figsize=(4,3));fig.text(-.2,.5,'ACTUAL_CLIPPING')
        self.assertIn('ACTUAL_CLIPPING',audit_layout(fig))
        plt.close(fig)


if __name__=='__main__':unittest.main()
