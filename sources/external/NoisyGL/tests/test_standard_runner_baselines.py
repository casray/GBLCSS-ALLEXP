import unittest

import numpy as np
import torch
from argparse import Namespace

from scripts.run_standard_downstream import (
    METHOD_CAPABILITIES,
    PREDICTOR_MODULES,
    _apply_config_overrides,
    _build_noisy_labels,
    _noise_fingerprint,
    _noise_subset_fingerprint,
    _noise_stats,
    _parameter_count,
    _split_fingerprint,
    build_parser,
    error_row_for,
)
from utils.labelnoise import label_process


class StandardRunnerBaselinesTest(unittest.TestCase):
    def test_nearest_work_methods_have_explicit_x_only_capabilities(self):
        expected_modules = {
            "gsdn_faithful_core": "predictor.GSDNFaithful_Predictor",
            "postel_mlp_noipl_rm": (
                "predictor.NearestWorkSoftTargets_Predictor"
            ),
            "als_mlp_rm": "predictor.NearestWorkSoftTargets_Predictor",
            "sals_mlp_adapted": (
                "predictor.NearestWorkSoftTargets_Predictor"
            ),
        }
        for method, module in expected_modules.items():
            with self.subTest(method=method):
                self.assertEqual(PREDICTOR_MODULES[method], module)
                capability = METHOD_CAPABILITIES[method]
                self.assertFalse(capability["requires_embedding"])
                self.assertTrue(capability["training_uses_graph"])
                self.assertFalse(capability["inference_uses_graph"])
                self.assertEqual(
                    capability["comparison_tier"],
                    "same_resource_x_only",
                )

    def test_gsdn_identity_is_equation_grounded_common_protocol_adaptation(
        self,
    ):
        status = METHOD_CAPABILITIES["gsdn_faithful_core"][
            "implementation_status"
        ]

        self.assertEqual(
            status,
            "equation-grounded common-protocol adaptation",
        )
        self.assertNotIn("equation-faithful", status)

    def test_r2lp_initial_label_state_uses_only_train_nodes(self):
        from predictor.R2LP_Predictor import _build_initial_label_state

        noisy_label = torch.tensor([0, 1, 2, 1, 0, 2])
        train_idx = np.array([1, 4])

        idx_clean, idx_unknown, y_clean, y_unknown = _build_initial_label_state(
            train_mask=train_idx,
            noisy_label=noisy_label,
            n_classes=3,
            device=torch.device("cpu"),
        )

        self.assertEqual(idx_clean.numel(), 0)
        self.assertEqual(idx_unknown.tolist(), train_idx.tolist())
        self.assertEqual(torch.count_nonzero(y_clean).item(), 0)
        self.assertEqual(torch.count_nonzero(y_unknown[[0, 2, 3, 5]]).item(), 0)
        self.assertTrue(torch.equal(
            y_unknown[train_idx],
            torch.nn.functional.one_hot(noisy_label[train_idx], 3).float(),
        ))

    def test_phase14_baselines_are_registered(self):
        for method in [
            "unionnet",
            "tss",
            "cgnn",
            "crgnn",
            "r2lp",
            "nrgnn_xh",
            "tfr",
            "procon",
            "gd2",
            "gcn_xh_ce",
            "frozengrace_gb_gcn_softdist_xh",
            "frozengrace_gb_gcn_softdist_xh_shuffled",
        ]:
            self.assertIn(method, PREDICTOR_MODULES)

    def test_config_overrides_update_only_declared_sections(self):
        conf = Namespace(
            model={"peer_k": 5, "alpha": 0.5},
            training={"lr": 0.01},
            dataset={"sparse": True},
        )

        _apply_config_overrides(conf, {
            "model.peer_k": 20,
            "training.lr": 0.005,
        })

        self.assertEqual(conf.model["peer_k"], 20)
        self.assertEqual(conf.training["lr"], 0.005)
        self.assertEqual(conf.model["alpha"], 0.5)

    def test_config_overrides_reject_unknown_keys(self):
        conf = Namespace(model={"peer_k": 5}, training={}, dataset={})

        with self.assertRaises(KeyError):
            _apply_config_overrides(conf, {"model.missing": 3})

    def test_standard_runner_requires_explicit_noise_scope(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args([])
        args = build_parser().parse_args(["--noise_scope", "train"])
        self.assertEqual(args.noise_scope, "train")
        args = build_parser().parse_args(["--noise_scope", "train_val"])
        self.assertEqual(args.noise_scope, "train_val")

    def test_ssl_swap_runner_requires_explicit_noise_scope(self):
        from scripts.run_ssl_swap_downstream import build_parser as build_ssl_parser

        with self.assertRaises(SystemExit):
            build_ssl_parser().parse_args(["--embedding_root", "embeddings"])
        args = build_ssl_parser().parse_args([
            "--embedding_root",
            "embeddings",
            "--noise_scope",
            "train",
        ])
        self.assertEqual(args.noise_scope, "train")

    def test_parameter_count_excludes_frozen_parameters(self):
        module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
        for parameter in module[1].parameters():
            parameter.requires_grad_(False)
        class_relevance = torch.nn.Parameter(torch.ones(2, 2))
        predictor = Namespace(
            model=module,
            class_relevance=class_relevance,
            duplicate_reference=class_relevance,
        )

        self.assertEqual(_parameter_count(predictor), 3 * 4 + 4 + 2 * 2)

    def test_label_process_only_corrupts_eligible_nodes(self):
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        features = torch.eye(6)
        train_idx = torch.tensor([0, 2, 4])

        noisy, modified = label_process(
            labels=labels,
            features=features,
            n_classes=3,
            noise_type="uniform",
            noise_rate=1.0,
            random_seed=7,
            eligible_mask=train_idx,
            debug=False,
        )

        self.assertTrue(torch.equal(noisy[torch.tensor([1, 3, 5])], labels[torch.tensor([1, 3, 5])]))
        self.assertEqual(set(modified.tolist()), set(train_idx.tolist()))

    def test_noise_stats_reports_zero_validation_and_test_corruption(self):
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        noisy = torch.tensor([1, 1, 0, 0, 2, 2])
        modified = np.array([0, 2, 4])

        stats = _noise_stats(
            labels,
            noisy,
            modified,
            train_mask=np.array([0, 2, 4]),
            val_mask=np.array([1]),
            test_mask=np.array([3, 5]),
        )

        self.assertEqual(stats["modified_train_count"], 3)
        self.assertEqual(stats["modified_val_count"], 0)
        self.assertEqual(stats["modified_test_count"], 0)

    def test_train_val_noise_preserves_train_realization_and_test_labels(self):
        labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        features = torch.eye(9)
        train_idx = torch.tensor([0, 1, 2])
        val_idx = torch.tensor([3, 4, 5])
        test_idx = torch.tensor([6, 7, 8])

        train_noisy, train_modified = _build_noisy_labels(
            labels=labels,
            features=features,
            n_classes=3,
            noise_type="uniform",
            noise_rate=1.0,
            seed=7,
            noise_scope="train",
            train_mask=train_idx,
            val_mask=val_idx,
        )
        train_val_noisy, train_val_modified = _build_noisy_labels(
            labels=labels,
            features=features,
            n_classes=3,
            noise_type="uniform",
            noise_rate=1.0,
            seed=7,
            noise_scope="train_val",
            train_mask=train_idx,
            val_mask=val_idx,
        )

        self.assertTrue(torch.equal(
            train_noisy[train_idx],
            train_val_noisy[train_idx],
        ))
        self.assertEqual(
            _noise_subset_fingerprint(labels, train_noisy, train_idx),
            _noise_subset_fingerprint(labels, train_val_noisy, train_idx),
        )
        self.assertTrue(torch.equal(train_noisy[val_idx], labels[val_idx]))
        self.assertTrue(torch.equal(
            train_val_noisy[test_idx],
            labels[test_idx],
        ))

        train_stats = _noise_stats(
            labels,
            train_noisy,
            train_modified,
            train_idx,
            val_idx,
            test_idx,
        )
        train_val_stats = _noise_stats(
            labels,
            train_val_noisy,
            train_val_modified,
            train_idx,
            val_idx,
            test_idx,
        )
        self.assertEqual(train_stats["modified_val_count"], 0)
        self.assertGreater(train_val_stats["modified_val_count"], 0)
        self.assertEqual(train_val_stats["modified_test_count"], 0)

    def test_split_and_noise_fingerprints_are_stable_and_sensitive(self):
        split_a = _split_fingerprint(np.array([0, 2]), np.array([1]), np.array([3, 4]))
        split_b = _split_fingerprint(np.array([0, 2]), np.array([1]), np.array([3, 4]))
        split_c = _split_fingerprint(np.array([0, 3]), np.array([1]), np.array([2, 4]))
        noise_a = _noise_fingerprint(torch.tensor([0, 1, 2]), torch.tensor([1, 1, 2]))
        noise_b = _noise_fingerprint(torch.tensor([0, 1, 2]), torch.tensor([1, 1, 2]))
        noise_c = _noise_fingerprint(torch.tensor([0, 1, 2]), torch.tensor([2, 1, 2]))

        self.assertEqual(split_a, split_b)
        self.assertNotEqual(split_a, split_c)
        self.assertEqual(noise_a, noise_b)
        self.assertNotEqual(noise_a, noise_c)

    def test_error_row_records_failed_run_without_metrics(self):
        row = error_row_for("cora", "unionnet", 3, "uniform", 0.4, RuntimeError("boom"))

        self.assertEqual(row["dataset"], "cora")
        self.assertEqual(row["method"], "unionnet")
        self.assertEqual(row["seed"], 3)
        self.assertEqual(row["status"], "failed")
        self.assertIn("boom", row["error"])
        self.assertEqual(row["test"], "")


if __name__ == "__main__":
    unittest.main()
