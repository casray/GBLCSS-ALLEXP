import unittest
from unittest.mock import patch

import torch

from scripts.measure_efficiency import (
    SUPPORTED_METHODS,
    count_unique_trainable_params,
    gpu_memory_metrics,
    install_ball_target_timer,
    prepare_gpu_measurement,
)
from scripts.summarize_softdist_mlp_efficiency import aggregate_rows


class FakeOptimizer:
    def __init__(self, param_groups):
        self.param_groups = param_groups


class CountParamsTests(unittest.TestCase):
    def test_counts_unique_trainable_optimizer_params(self):
        shared = torch.nn.Linear(3, 2)
        frozen = torch.nn.Linear(2, 2)
        for param in frozen.parameters():
            param.requires_grad = False
        optimizer = FakeOptimizer(
            [
                {"params": list(shared.parameters())},
                {"params": list(shared.parameters()) + list(frozen.parameters())},
            ]
        )

        self.assertEqual(
            count_unique_trainable_params(optimizer),
            sum(param.numel() for param in shared.parameters()),
        )

    def test_supports_all_preregistered_efficiency_methods(self):
        self.assertEqual(
            set(SUPPORTED_METHODS),
            {
                "mlp_ce",
                "softdist_mlp",
                "gcn",
                "softdist_gcn",
                "gcn_xh_ce",
                "softdist_xh",
                "nrgnn",
            },
        )

    def test_gpu_memory_metrics_safely_degrade_on_cpu(self):
        self.assertEqual(
            gpu_memory_metrics(torch.device("cpu")),
            {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0},
        )

    def test_gpu_measurement_initializes_context_before_reset(self):
        calls = []
        with (
            patch(
                "scripts.measure_efficiency.cuda_synchronize",
                side_effect=lambda device: calls.append("sync"),
            ),
            patch(
                "scripts.measure_efficiency.reset_gpu_peak_memory",
                side_effect=lambda device: calls.append("reset"),
            ),
        ):
            prepare_gpu_measurement(torch.device("cuda:0"))

        self.assertEqual(calls, ["sync", "reset", "sync"])

    def test_ball_target_timer_wraps_real_predictor_method(self):
        class FakePredictor:
            def __init__(self):
                self.calls = 0

            def _make_corrected_labels(self):
                self.calls += 1
                return "done"

        predictor = FakePredictor()
        elapsed = install_ball_target_timer(predictor, torch.device("cpu"))

        self.assertEqual(predictor._make_corrected_labels(), "done")
        self.assertEqual(predictor.calls, 1)
        self.assertGreaterEqual(elapsed(), 0.0)

    def test_efficiency_aggregate_uses_sample_standard_deviation(self):
        rows = [
            {
                "dataset": "cora",
                "method": "mlp_ce",
                "trainable_params": 10,
                "init_time_sec": 1.0,
                "ball_target_time_sec": 0.0,
                "classifier_train_time_sec": value,
                "downstream_total_time_sec": value + 1.0,
                "peak_allocated_bytes": 100,
                "peak_reserved_bytes": 200,
                "test": 0.5,
            }
            for value in (1.0, 2.0, 3.0)
        ]

        summary = aggregate_rows(rows)[0]

        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["classifier_train_time_sec_mean"], 2.0)
        self.assertEqual(summary["classifier_train_time_sec_std"], 1.0)


if __name__ == "__main__":
    unittest.main()
