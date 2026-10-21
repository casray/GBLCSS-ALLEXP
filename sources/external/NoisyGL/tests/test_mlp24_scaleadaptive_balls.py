import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch


DATASET_NODES = {
    "cora": 2708,
    "citeseer": 3327,
    "corafull": 19793,
    "coauthorcs": 18333,
    "amazonpho": 7650,
    "citeseerfull": 4230,
    "coauthorph": 34493,
    "pubmed": 19717,
    "amazoncom": 13752,
    "coraml": 2995,
    "attribwiki": 2405,
}

EXPECTED_SQRT_CAPS = {
    "cora": 64,
    "citeseer": 64,
    "corafull": 141,
    "coauthorcs": 136,
    "amazonpho": 88,
    "citeseerfull": 66,
    "coauthorph": 186,
    "pubmed": 141,
    "amazoncom": 118,
    "coraml": 64,
    "attribwiki": 64,
}


class MLP24ScaleAdaptiveBallTests(unittest.TestCase):
    def test_sqrt_nodes_policy_matches_preregistered_dataset_caps(self):
        from utils.ordinary_gb import resolve_effective_max_balls

        actual = {
            dataset: resolve_effective_max_balls(
                n_nodes=n_nodes,
                configured_max_balls=64,
                policy="sqrt_nodes",
            )
            for dataset, n_nodes in DATASET_NODES.items()
        }
        self.assertEqual(actual, EXPECTED_SQRT_CAPS)

    def test_fixed_policy_preserves_configured_cap(self):
        from utils.ordinary_gb import resolve_effective_max_balls

        self.assertEqual(
            resolve_effective_max_balls(34493, configured_max_balls=64, policy="fixed"),
            64,
        )
        self.assertEqual(
            resolve_effective_max_balls(20, configured_max_balls=64, policy="fixed"),
            20,
        )

    def test_unknown_policy_is_rejected(self):
        from utils.ordinary_gb import resolve_effective_max_balls

        with self.assertRaisesRegex(ValueError, "max_balls_policy"):
            resolve_effective_max_balls(100, configured_max_balls=64, policy="tuned")

    def test_unchanged_small_graph_partition_is_identical(self):
        from utils.ordinary_gb import build_ordinary_granular_balls, resolve_effective_max_balls

        rng = np.random.RandomState(7)
        embeddings = rng.normal(size=(120, 8)).astype(np.float32)
        fixed = build_ordinary_granular_balls(
            embeddings,
            max_balls=64,
            min_ball_size=8,
            split_improvement=0.05,
            random_state=3,
        )
        adaptive = build_ordinary_granular_balls(
            embeddings,
            max_balls=resolve_effective_max_balls(120, 64, "sqrt_nodes"),
            min_ball_size=8,
            split_improvement=0.05,
            random_state=3,
        )
        self.assertTrue(np.array_equal(fixed.assignment, adaptive.assignment))

    def test_predictor_resolves_and_reports_adaptive_capacity(self):
        from predictor.FrozenGraceGB_GCN_Predictor import _FrozenGraceGBGCNBase

        predictor = _FrozenGraceGBGCNBase.__new__(_FrozenGraceGBGCNBase)
        predictor.conf = SimpleNamespace(
            gb={"max_balls": 64, "max_balls_policy": "sqrt_nodes"}
        )
        predictor.noisy_label = np.zeros(34493, dtype=np.int64)

        capacity = predictor._resolve_ball_capacity()

        self.assertEqual(capacity["configured_max_balls"], 64)
        self.assertEqual(capacity["effective_max_balls"], 186)
        self.assertEqual(capacity["max_balls_policy"], "sqrt_nodes")

    def test_ssl_swap_cli_accepts_scale_policy(self):
        from scripts.run_ssl_swap_downstream import build_parser

        args = build_parser().parse_args(
            [
                "--embedding_root",
                "embeddings",
                "--noise_scope",
                "train",
                "--gb_max_balls_policy",
                "sqrt_nodes",
            ]
        )
        self.assertEqual(args.gb_max_balls_policy, "sqrt_nodes")

    def test_embedding_lookup_supports_all_mlp24_dataset_aliases(self):
        from scripts.linear_probe_embeddings import embedding_path_for

        expected_dirs = {
            "corafull": "CoraFull",
            "coraml": "CoraML",
            "citeseerfull": "citeseerfull",
            "attribwiki": "attribwiki",
            "amazoncom": "amazoncom",
        }
        root = Path("virtual_embeddings")
        for dataset, directory in expected_dirs.items():
            with self.subTest(dataset=dataset):
                self.assertEqual(
                    embedding_path_for(root, dataset, "grace"),
                    root / directory / "grace" / "H.pt",
                )

    def test_geometry_gate_requires_exercised_physics_partition(self):
        from scripts.summarize_mlp24_scaleadaptive_balls_20260712 import geometry_gate

        rows = [
            {
                "dataset": "coauthorph",
                "variant": "fixed",
                "status": "ok",
                "num_balls": 64,
                "mean_ball_size": 538.95,
                "mean_train_per_ball": 107.78,
                "alpha_positive_train_count": 6898,
                "ball_build_time_seconds": 20.0,
            },
            {
                "dataset": "coauthorph",
                "variant": "sqrt_nodes",
                "status": "ok",
                "num_balls": 150,
                "mean_ball_size": 229.95,
                "mean_train_per_ball": 45.98,
                "alpha_positive_train_count": 6898,
                "ball_build_time_seconds": 70.0,
            },
        ]
        decision = geometry_gate(rows, require_all=False)
        self.assertEqual(decision["pass"], 1)

        rows[1]["num_balls"] = 64
        rows[1]["mean_ball_size"] = 538.95
        decision = geometry_gate(rows, require_all=False)
        self.assertEqual(decision["pass"], 0)
        self.assertIn("physics_partition_not_exercised", decision["reasons"])

    def test_protocol_audit_rejects_non_train_only_rows(self):
        from scripts.summarize_mlp24_scaleadaptive_balls_20260712 import protocol_row_ok

        valid = {
            "noise_scope": "train",
            "noise_type": "uniform",
            "noise_rate": 0.4,
            "train_percent": 0.2,
            "val_percent": 0.1,
            "test_percent": 0.7,
            "modified_count": 10,
            "modified_train_count": 10,
            "modified_val_count": 0,
            "modified_test_count": 0,
            "status": "ok",
        }
        self.assertTrue(protocol_row_ok(valid, "uniform", 0.4))
        invalid = dict(valid, modified_test_count=1)
        self.assertFalse(protocol_row_ok(invalid, "uniform", 0.4))

    def test_paired_deltas_require_matching_split_and_noise_hashes(self):
        from scripts.summarize_mlp24_scaleadaptive_balls_20260712 import paired_deltas

        rows = [
            {
                "dataset": "coauthorph",
                "seed": 0,
                "variant": "fixed",
                "test": 0.90,
                "split_hash": "split-a",
                "noise_hash": "noise-a",
            },
            {
                "dataset": "coauthorph",
                "seed": 0,
                "variant": "sqrt_nodes",
                "test": 0.92,
                "split_hash": "split-a",
                "noise_hash": "noise-a",
            },
        ]
        deltas = paired_deltas(rows, "sqrt_nodes", "fixed")
        self.assertAlmostEqual(deltas[0]["delta"], 0.02)

        rows[1]["noise_hash"] = "noise-b"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            paired_deltas(rows, "sqrt_nodes", "fixed")

    def test_geometry_helper_preserves_small_graph_targets_and_shuffled_sizes(self):
        from scripts.diagnose_mlp24_scaleadaptive_balls_20260712 import compare_geometry_variants

        rng = np.random.RandomState(11)
        embeddings = torch.tensor(rng.normal(size=(120, 8)), dtype=torch.float32)
        clean = torch.tensor(np.arange(120) % 3, dtype=torch.long)
        noisy = clean.clone()
        noisy[::5] = (noisy[::5] + 1) % 3
        train_mask = np.arange(0, 120, 2)

        rows = compare_geometry_variants(
            embeddings=embeddings,
            noisy_label=noisy,
            clean_label=clean,
            train_mask=train_mask,
            n_classes=3,
            dataset="toy",
            seed=0,
            configured_max_balls=64,
            min_ball_size=8,
            split_improvement=0.05,
        )

        by_variant = {row["variant"]: row for row in rows}
        self.assertEqual(by_variant["sqrt_nodes"]["assignment_equal_fixed"], 1)
        self.assertEqual(by_variant["sqrt_nodes"]["soft_target_equal_fixed"], 1)
        self.assertEqual(by_variant["sqrt_nodes"]["alpha_equal_fixed"], 1)
        self.assertEqual(by_variant["sqrt_nodes_shuffled"]["size_multiset_equal_adaptive"], 1)
        self.assertEqual(by_variant["sqrt_nodes_shuffled"]["size_multiset_equal_source"], 1)

    def test_smoke_and_five_seed_gates_follow_preregistered_thresholds(self):
        from scripts.summarize_mlp24_scaleadaptive_balls_20260712 import (
            AFFECTED_DATASETS,
            five_seed_gate,
            smoke_gate,
        )

        rows = []
        for dataset in AFFECTED_DATASETS:
            for seed in range(5):
                fixed = 0.90
                gain = 0.012 if dataset == "coauthorph" else 0.006
                adaptive = fixed + gain
                common = {
                    "dataset": dataset,
                    "seed": seed,
                    "noise_type": "uniform",
                    "noise_rate": 0.4,
                    "noise_scope": "train",
                    "train_percent": 0.2,
                    "val_percent": 0.1,
                    "test_percent": 0.7,
                    "modified_count": 10,
                    "modified_train_count": 10,
                    "modified_val_count": 0,
                    "modified_test_count": 0,
                    "status": "ok",
                    "split_hash": f"split-{dataset}-{seed}",
                    "noise_hash": f"noise-{dataset}-{seed}",
                }
                rows.extend(
                    [
                        {**common, "variant": "fixed", "test": fixed},
                        {**common, "variant": "sqrt_nodes", "test": adaptive},
                        {**common, "variant": "sqrt_nodes_shuffled", "test": adaptive - 0.03},
                        {**common, "variant": "mlp_jocor", "test": adaptive - 0.011},
                    ]
                )

        self.assertEqual(smoke_gate([row for row in rows if row["seed"] == 0])["pass"], 1)
        self.assertEqual(five_seed_gate(rows)["pass"], 1)

        for row in rows:
            if row["dataset"] == "coauthorph" and row["variant"] == "sqrt_nodes":
                row["test"] = 0.905
        self.assertEqual(five_seed_gate(rows)["pass"], 0)


if __name__ == "__main__":
    unittest.main()
