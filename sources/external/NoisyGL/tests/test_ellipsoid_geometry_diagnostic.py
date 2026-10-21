import unittest

import numpy as np

from scripts.diagnose_ellipsoid_geometry import (
    aggregate_rows,
    ellipsoid_cluster_diagnostics,
    h_knn_label_diagnostics,
    root_split_diagnostics,
)


class TestEllipsoidGeometryDiagnostic(unittest.TestCase):
    def test_root_split_diagnostics_reports_positive_gain_for_two_clusters(self):
        features = np.array(
            [
                [-2.0, 0.0],
                [-1.8, 0.1],
                [-2.1, -0.1],
                [2.0, 0.0],
                [1.9, 0.1],
                [2.1, -0.1],
            ],
            dtype=np.float32,
        )

        diag = root_split_diagnostics(features, min_ball_size=2, random_state=0)

        self.assertGreater(diag["root_split_improvement"], 0.8)
        self.assertEqual(
            sorted([diag["root_split_left_size"], diag["root_split_right_size"]]),
            [3, 3],
        )

    def test_h_knn_label_diagnostics_reports_lift_over_random(self):
        features = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [-1.0, 0.0],
                [-0.9, -0.1],
                [-0.8, -0.2],
            ],
            dtype=np.float32,
        )
        labels = np.array([0, 0, 0, 1, 1, 1])

        diag = h_knn_label_diagnostics(features, labels, k=2)

        self.assertAlmostEqual(diag["h_knn10_clean_agree"], 1.0)
        self.assertAlmostEqual(diag["h_knn10_random_agree"], 0.5)
        self.assertAlmostEqual(diag["h_knn10_lift_over_random"], 2.0)

    def test_ellipsoid_cluster_diagnostics_reports_shape_and_weighted_purity_gain(self):
        features = np.array(
            [
                [0.00, 0.00],
                [0.05, 0.00],
                [0.10, 0.00],
                [0.15, 0.00],
                [4.00, 4.00],
            ],
            dtype=np.float32,
        )
        assignment = np.zeros(5, dtype=np.int64)
        noisy = np.array([1, 1, 1, 1, 0], dtype=np.int64)
        clean = np.array([1, 1, 1, 1, 1], dtype=np.int64)
        train_idx = np.arange(5, dtype=np.int64)

        diag = ellipsoid_cluster_diagnostics(
            features=features,
            assignment=assignment,
            noisy_label=noisy,
            clean_label=clean,
            train_idx=train_idx,
            n_classes=2,
            min_train_per_ball=2,
            ridge_scale=0.05,
            weight_floor=0.1,
        )

        self.assertEqual(diag["eligible_balls"], 1)
        self.assertGreater(diag["ellipsoid_anisotropy_mean"], 1.0)
        self.assertLess(diag["ellipsoid_effective_support_ratio_mean"], 1.0)
        self.assertGreater(diag["weighted_noisy_train_purity_delta"], 0.0)
        self.assertAlmostEqual(diag["weighted_true_train_purity_delta"], 0.0)

    def test_aggregate_rows_reports_mean_and_std_by_dataset(self):
        rows = [
            {"dataset": "a", "seed": 0, "root_split_improvement": 0.1, "num_balls": 2},
            {"dataset": "a", "seed": 1, "root_split_improvement": 0.3, "num_balls": 4},
            {"dataset": "b", "seed": 0, "root_split_improvement": 0.5, "num_balls": 1},
        ]

        aggregate = {row["dataset"]: row for row in aggregate_rows(rows)}

        self.assertAlmostEqual(aggregate["a"]["root_split_improvement_mean"], 0.2)
        self.assertAlmostEqual(aggregate["a"]["num_balls_mean"], 3.0)
        self.assertAlmostEqual(aggregate["b"]["root_split_improvement_std"], 0.0)


if __name__ == "__main__":
    unittest.main()
