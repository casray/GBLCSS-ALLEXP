"""Independent target and partition audit contracts; no experiment data needed."""
import unittest
import numpy as np

from audit_report import audit_target, validate_nested, paired_deltas, verify_trace


class AuditTests(unittest.TestCase):
    def test_singleton_and_one_other_vote_disable_mixing(self):
        result = audit_target(np.array([0, 1, 1]), np.array([0, 0, 1]), 2)
        np.testing.assert_array_equal(result['support'], [0, 1, 1])
        np.testing.assert_array_equal(result['alpha'], [0, 0, 0])
        np.testing.assert_array_equal(result['target'], np.eye(2)[[0, 0, 1]])
        np.testing.assert_allclose(result['r'][0], [.5, .5])

    def test_tie_is_not_arbitrarily_sharpened(self):
        result = audit_target(np.zeros(3, dtype=int), np.array([0, 0, 1]), 2)
        np.testing.assert_array_equal(result['q'][0], [.5, .5])
        self.assertEqual(result['alpha'][0], 0)
        np.testing.assert_array_equal(result['q'][2], [1, 0])
        self.assertEqual(result['alpha'][2], 1)

    def test_leave_one_out_self_label_cannot_change_own_q_or_alpha(self):
        groups = np.array([40, 40, 40, 40, 99, 99])
        observed = np.array([0, 0, 1, 2, 0, 1])
        first = audit_target(groups, observed, 3)
        for node in range(len(groups)):
            changed = observed.copy(); changed[node] = (changed[node]+1) % 3
            second = audit_target(groups, changed, 3)
            for key in ['r', 'q', 'alpha', 'support']:
                np.testing.assert_array_equal(first[key][node], second[key][node])

    def test_valid_nested_partitions_ignore_numeric_ball_ids(self):
        self.assertTrue(validate_nested({1: np.array([7, 7, 7, 7]),
                                         2: np.array([9, 9, 3, 3]),
                                         8: np.array([0, 1, 2, 3])}))

    def test_crossing_partitions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'nested'):
            validate_nested({2: np.array([0, 0, 1, 1]),
                             4: np.array([7, 8, 7, 8])})

    def test_same_count_must_mean_same_regions(self):
        with self.assertRaises(ValueError):
            validate_nested({8: np.array([0, 0, 1, 1]),
                             16: np.array([4, 5, 4, 5])})

    def test_paired_comparisons_do_not_duplicate_count_capped_aliases(self):
        import pandas as pd
        rows = [dict(dataset='toy', seed=0, noise='clean', arm=arm, space=space,
                     kind=kind, requested_balls=req, actual_balls=count,
                     test_accuracy=score) for arm, space, kind, req, count, score in
                [('CE', 'NA', 'CE', 0, 0, .5), ('GRACE_B64', 'GRACE', 'B', 64, 4, .7),
                 ('GRACE_B128', 'GRACE', 'B', 128, 4, .7),
                 ('GREET_B64', 'GREET', 'B', 64, 4, .6),
                 ('GREET_B128', 'GREET', 'B', 128, 4, .6)]]
        result = paired_deltas(pd.DataFrame(rows), 'test_accuracy')
        cross = result[result.comparison == 'matched_actual_count_cross_space']
        self.assertEqual(len(cross), 1)
        self.assertAlmostEqual(cross.iloc[0].delta, -10)

    def test_final_partition_and_trace_reconstruct_every_prefix(self):
        h = np.r_[np.tile([-1., 0.], (32, 1)), np.tile([1., 0.], (32, 1))]
        assignments = {'B1': np.zeros(64, dtype=int), 'B8': np.repeat([0, 1], 32),
                       'B16': np.repeat([0, 1], 32)}
        stats = {key: dict(actual_balls=count, normalized_input_sse=sse,
                           min_train_support=support, mean_other_votes=support-1)
                 for key, count, sse, support in [('B1', 1, 64, 64), ('B8', 2, 0, 32), ('B16', 2, 0, 32)]}
        trace = [dict(accepted=True, parent_id=0, child_ids=[1, 2], child_sizes=[32, 32],
                      child_train_counts=[32, 32], parent_sse=64, child_sse=[0, 0],
                      sse_gain=64, actual_count_after=2)]
        self.assertEqual(verify_trace(assignments, stats, trace, np.arange(64), h), 3)
        trace[0]['child_train_counts'] = [31, 33]
        with self.assertRaisesRegex(ValueError, 'support mismatch'):
            verify_trace(assignments, stats, trace, np.arange(64), h)

    def test_curve_collapse_rejects_conflicting_cap_duplicates(self):
        import pandas as pd
        from plot_curves import collapse_repeated_partitions
        frame = pd.DataFrame([dict(dataset='toy', seed=0, noise='clean', space='GRACE',
                                  kind='B', actual_balls=3, requested_balls=b,
                                  target_brier=value) for b, value in [(8, .2), (16, .2)]])
        self.assertEqual(len(collapse_repeated_partitions(frame, 'target_brier')), 1)
        frame.loc[1, 'target_brier'] = .3
        with self.assertRaisesRegex(ValueError, 'aliases disagree'):
            collapse_repeated_partitions(frame, 'target_brier')


if __name__ == '__main__':
    unittest.main()
