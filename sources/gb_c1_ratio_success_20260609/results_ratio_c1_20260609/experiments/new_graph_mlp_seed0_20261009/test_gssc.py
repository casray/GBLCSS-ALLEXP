"""Formula/API tests; parent explicitly schedules the opt-in GPU smoke."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F

PATH = Path(__file__).with_name("gssc_adapter.py")

class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = None
        if PATH.is_file():
            spec = importlib.util.spec_from_file_location("gssc_adapter", PATH)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def setUp(self):
        self.assertIsNotNone(self.module, "GSSC adapter is not implemented")

    def test_fixed_parameter_mapping(self):
        mappings = {"Cora": "cora", "CiteSeer": "citeseer", "CoauthorCS": "coauthor-cs", "CoraFull": "cora", "Photo": "cora"}
        for name, mapping in mappings.items():
            cfg = self.module.resolve_config(name, {"epochs": 2})
            self.assertEqual(cfg["official_parameter_dataset"], mapping)
            for key, value in {"epochs": 2, "hidden": 64, "dropout": .5, "lr": .01, "weight_decay": .0005, "ep_epochs": 100, "cl_epochs": 100, "mixup_lr": .0001}.items():
                self.assertEqual(cfg[key], value)
        self.assertEqual(self.module.resolve_config("CiteSeer", {})["temp"], .8)
        self.assertEqual(self.module.resolve_config("CoauthorCS", {})["alpha"], .3)
        for bad in ({"seed": 1}, {"epochs": 0}, {"method": "mlp"}, {"hidden": 128}):
            with self.assertRaises(ValueError):
                self.module.resolve_config("Cora", bad)

    def test_parent_config_smoke_aliases(self):
        cfg = self.module.resolve_config("amazonpho", dict(seed=0, epochs=2, hidden=64,
                 dropout=.5, learning_rate=.01, weight_decay=.0005, noise="uniform0.4",
                 validation_selection="clean validation accuracy; last tie", device="cuda", smoke=True))
        self.assertEqual(cfg["official_parameter_dataset"], "cora")
        self.assertEqual(cfg["phase_budgets"], {"edge_pretrain": 2, "classifier_pretrain": 2, "main": 2})
        self.assertEqual(cfg["supervised_epochs_total"], 4)

    def test_architectures_match_author_source(self):
        cfg = self.module.resolve_config("Cora", {})
        source = PATH.parent / "vendor/gssc/code/model.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                    and node.name in ("FC", "MLP_GNN", "MixupScale")]
        namespace = {"torch": torch, "nn": torch.nn}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
        x = torch.randn(6, 5)
        official = namespace["MLP_GNN"](5, 64, 3, cfg).eval()
        port = self.module.NeighContrast(5, 64, 3, cfg).eval()
        port.load_state_dict(official.state_dict())
        with torch.no_grad():
            for expected, got in zip(official(x), port(x)):
                torch.testing.assert_close(got, expected)
            official_mix = namespace["MixupScale"](5, 64, cfg).eval()
            port_mix = self.module.MixupScale(5, 64, cfg).eval()
            port_mix.load_state_dict(official_mix.state_dict())
            src, dst = torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])
            torch.testing.assert_close(port_mix(x, src, dst), official_mix(x, src, dst))

    def test_validation_uses_ordered_val_y_and_no_other_labels(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x, x
        class ValidationOnly:
            x = torch.tensor([[9., 0.], [9., 0.], [9., 0.], [0., 9.]])
            val_index = torch.tensor([3, 0])
            val_y = torch.tensor([1, 0])
            @property
            def y(self):
                raise AssertionError("validation accessed full labels")
            @property
            def test_index(self):
                raise AssertionError("validation accessed test split")
        self.assertEqual(self.module.validation_accuracy(Model(), ValidationOnly()), 1.)

    def test_chunked_scores_and_gradients(self):
        torch.manual_seed(8)
        h = torch.randn(7, 4, requires_grad=True)
        src = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 3])
        dst = torch.tensor([1, 2, 3, 4, 5, 6, 0, 3, 0])
        expected = (h[src] * h[dst]).sum(1)
        got = self.module.edge_scores(h, src, dst, 2)
        torch.testing.assert_close(got, expected)
        a = torch.autograd.grad(got.square().sum(), h, retain_graph=True)[0]
        b = torch.autograd.grad(expected.square().sum(), h)[0]
        torch.testing.assert_close(a, b)

    def test_straight_through_hard_forward_soft_backward(self):
        soft = torch.tensor([.2, .6, .9], requires_grad=True)
        hard = self.module.straight_through(soft)
        torch.testing.assert_close(hard, torch.tensor([0., 1., 1.]))
        hard.sum().backward()
        torch.testing.assert_close(soft.grad, torch.ones_like(soft))

    def test_sampler_clamp_matches_pyro_clamp_probs_gradient(self):
        from torch.distributions.utils import clamp_probs
        cfg = self.module.resolve_config("Cora", {})
        sampler = self.module.EdgeSampler(3, 64, cfg)
        soft = torch.tensor([0., 1e-18, .2, .8, 1. - 1e-16, 1.], dtype=torch.double, requires_grad=True)
        expected_soft = soft.detach().clone().requires_grad_(True)
        expected = self.module.straight_through(clamp_probs(expected_soft))
        with patch.object(torch.distributions, "RelaxedBernoulli") as distribution:
            distribution.return_value.rsample.return_value = soft
            got = sampler(torch.ones(6, 3), torch.arange(6), torch.arange(6).roll(1))
        torch.testing.assert_close(got, expected)
        a = torch.autograd.grad(got.sum(), soft)[0]
        b = torch.autograd.grad(expected.sum(), expected_soft)[0]
        torch.testing.assert_close(a, b)

    def test_sampler_distribution_and_parameter_gradient(self):
        from torch.distributions.utils import clamp_probs
        cfg = self.module.resolve_config("Cora", {})
        sampler = self.module.EdgeSampler(3, 64, cfg).double()
        x = torch.randn(6, 3, dtype=torch.double)
        src, dst = torch.arange(6), torch.arange(6).roll(1)
        rng = torch.get_rng_state()
        got = sampler(x, src, dst)
        a = torch.autograd.grad(got.sum(), tuple(sampler.parameters()))
        torch.set_rng_state(rng)
        h = sampler.linear2(sampler.linear1(x))
        probs = (h[src] * h[dst]).sum(1).sigmoid() * (1 - cfg["alpha"]) + cfg["alpha"]
        soft = clamp_probs(torch.distributions.RelaxedBernoulli(x.new_tensor(cfg["temp"]), probs=probs).rsample())
        expected = self.module.straight_through(soft)
        b = torch.autograd.grad(expected.sum(), tuple(sampler.parameters()))
        torch.testing.assert_close(got, expected)
        for actual, reference in zip(a, b):
            torch.testing.assert_close(actual, reference)

    def test_homophily_sampler_gradient(self):
        soft = torch.tensor([.7, .8, .9], requires_grad=True)
        mean, total = self.module.homophily_loss(torch.tensor([0, 0, 1]), torch.tensor([0, 1, 0]), torch.tensor([1, 2, 2]), self.module.straight_through(soft))
        self.assertAlmostEqual(mean.item(), 1.)
        self.assertAlmostEqual(total.item(), 2.)
        mean.backward()
        self.assertLess(soft.grad[0].item(), 0)
        self.assertGreater(soft.grad[1].item(), 0)
        self.assertGreater(soft.grad[2].item(), 0)

    def test_self_contrast_official_formula(self):
        torch.manual_seed(9)
        y1, y2, z3, z4, z = [torch.randn(4, 3) for _ in range(5)]
        expected = F.mse_loss(y1, z3) + F.mse_loss(y2, z4) - F.mse_loss(y1.softmax(-1), z.softmax(-1)) - F.mse_loss(y2.softmax(-1), z.softmax(-1))
        torch.testing.assert_close(self.module.self_contrast(y1, y2, z3, z4, z), expected)

    def test_endpoint_weights_noisy_train_only(self):
        y = torch.tensor([0, -1, 1, -1, 0])
        src = torch.tensor([0, 1, 2, 3, 4])
        dst = torch.tensor([1, 2, 3, 4, 0])
        weight, missing = self.module.endpoint_weights(y, y >= 0, src, dst, 3)
        raw = torch.tensor([2 / 4, 1 / 2, 0.])
        torch.testing.assert_close(weight, 3 * F.normalize(raw, p=1, dim=0))
        self.assertEqual(missing, [])
        self.assertTrue(torch.isfinite(weight).all())

    def test_endpoint_weights_degenerate_classes(self):
        y = torch.tensor([0, 1, -1, -1])
        src, dst = torch.tensor([2, 3]), torch.tensor([3, 2])
        weight, missing = self.module.endpoint_weights(y, y >= 0, src, dst, 3)
        self.assertEqual(missing, [0, 1])
        torch.testing.assert_close(weight, torch.zeros(3))

    def test_run_rejects_cpu_before_writing(self):
        from torch_geometric.data import Data
        with patch.object(Path, "mkdir", side_effect=AssertionError("unexpected write")):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                self.module.run(Data(x=torch.ones(4, 3)), {"epochs": 1}, "must_not_be_created")

    @unittest.skipUnless(os.environ.get("GSSC_RUN_GPU_TESTS") == "1", "parent schedules GPU smoke")
    def test_tiny_cuda_api(self):
        from torch_geometric.data import Data
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        device = torch.device("cuda")
        n = 12
        train = torch.tensor([4, 0, 8, 2], device=device)
        val = torch.tensor([5, 1, 9], device=device)
        test = torch.tensor([11, 3, 7], device=device)
        y = torch.full((n,), -1, device=device, dtype=torch.long)
        y[train] = torch.tensor([0, 1, 0, 1], device=device)
        data = Data(x=torch.randn(n, 5, device=device), edge_index=torch.stack([torch.arange(n, device=device), torch.arange(n, device=device).roll(1)]), y=y, train_index=train, val_index=val, test_index=test, val_y=torch.tensor([1, 0, 1], device=device), num_classes=2, name="Cora")
        for split in ("train", "val", "test"):
            mask = torch.zeros(n, dtype=torch.bool, device=device)
            mask[getattr(data, split + "_index")] = True
            setattr(data, split + "_mask", mask)
        with tempfile.TemporaryDirectory(prefix="gssc_api_") as directory:
            with patch.object(self.module, "validation_accuracy", return_value=.5):
                result = self.module.run(data, {"epochs": 2, "ep_epochs": 1, "cl_epochs": 1, "edge_chunk_size": 3}, directory)
            probs = torch.load(result["probabilities_file"], weights_only=True)
            self.assertEqual(probs.device.type, "cpu")
            self.assertEqual(tuple(probs.shape), (n, 2))
            torch.testing.assert_close(probs.sum(1), torch.ones(n))
            self.assertEqual(result["best_epoch"], 3)
            saved = torch.load(Path(directory) / "checkpoint.pt", weights_only=True)
            self.assertEqual(saved["config"], result["config"])
            self.assertEqual(set(saved["state"]), {"model", "mixup", "sampler"})
            rows = [json.loads(line) for line in (Path(directory) / "epochs.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual([r["phase"] for r in rows], ["edge_pretrain", "classifier_pretrain", "main", "main"])
            self.assertTrue(all("loss" in r and "val" in r for r in rows))
            self.assertTrue(torch.equal(data.y, y))
            self.assertFalse(result["diagnostics"]["test_labels_read"])
            model = self.module.NeighContrast(5, 64, 2, result["config"]).to(device)
            model.load_state_dict(saved["state"]["model"])
            model.eval()
            with torch.no_grad():
                torch.testing.assert_close(probs, model(data.x)[0].softmax(-1).cpu())
            with self.assertRaises(FileExistsError):
                self.module.run(data, {"epochs": 1}, directory)
        # Reject an actual leakage attempt before writing any artifacts.
        data.y = y.clone()
        data.y[val[0]] = 0
        with patch.object(Path, "mkdir", side_effect=AssertionError("unexpected write")):
            with self.assertRaisesRegex(ValueError, "label isolation"):
                self.module.run(data, {"epochs": 1}, "must_not_be_created")

if __name__ == "__main__":
    unittest.main()
