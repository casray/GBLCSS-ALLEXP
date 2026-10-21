from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "GB-LCSS_code" / "src"))
sys.path.insert(0, str(HERE))
from gblcss.data import GraphData
from gblcss.training import train_x_only_mlp

try:
    import metric_training as metrics
except ModuleNotFoundError:
    metrics = None


def small_graph(classes=3):
    count = 6 * classes
    generator = torch.Generator().manual_seed(81)
    labels = torch.arange(count) % classes
    masks = [torch.zeros(count, dtype=torch.bool) for _ in range(3)]
    for index, mask in enumerate(masks):
        mask[index * 2 * classes:(index + 1) * 2 * classes] = True
    noisy = labels.clone()
    noisy[masks[0]] = (noisy[masks[0]] + 1) % classes
    # Deliberately differ on validation to lock the original noisy-label semantics.
    noisy[masks[1]] = (noisy[masks[1]] + 1) % classes
    return GraphData(
        "small", torch.randn(count, 5, generator=generator),
        torch.empty((2, 0), dtype=torch.long), labels, *masks, noisy_y=noisy,
        train_index=torch.where(masks[0])[0].flip(0),
        val_index=torch.where(masks[1])[0].flip(0),
    )


class MetricTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.assertIsNotNone(metrics, "metric_training module must implement the experiment trainer")

    def test_binary_metrics_use_positive_softmax_probability(self):
        # Extreme logits saturate softmax: logit differences would give a different AUC.
        logits = torch.tensor([[0., 100.], [0., 90.], [0., -100.], [0., -90.]])
        labels = torch.tensor([1, 0, 0, 1])
        result = metrics.evaluate_logits(logits, labels, binary=True)
        probability = logits.softmax(dim=1)[:, 1].numpy()
        self.assertEqual(set(result), {"accuracy", "macro_f1", "roc_auc", "average_precision"})
        self.assertEqual(result["roc_auc"], roc_auc_score(labels, probability))
        self.assertEqual(result["average_precision"], average_precision_score(labels, probability))
        self.assertNotEqual(result["roc_auc"], roc_auc_score(labels, (logits[:, 1] - logits[:, 0]).numpy()))

    def test_multiclass_metrics_match_sklearn(self):
        logits = torch.tensor([[3., 1., 0.], [1., 3., 0.], [3., 0., 1.], [1., 0., 3.]])
        labels = torch.tensor([0, 1, 2, 2])
        result = metrics.evaluate_logits(logits, labels)
        self.assertEqual(set(result), {"accuracy", "macro_f1"})
        self.assertAlmostEqual(result["accuracy"], accuracy_score(labels, logits.argmax(1)))
        self.assertAlmostEqual(result["macro_f1"], f1_score(labels, logits.argmax(1), average="macro"))

    def test_auc_and_accuracy_can_choose_different_epochs_and_keep_first_ties(self):
        labels = torch.tensor([0, 0, 1, 1])
        logits = [torch.tensor([[0., -1.], [0., -2.], [0., -.5], [0., -.4]]),
                  torch.tensor([[0., -1.], [0., 2.], [0., 1.], [0., 3.]])]
        results = [metrics.evaluate_logits(value, labels, binary=True) for value in logits]
        self.assertEqual(metrics.select_best_epoch([value["roc_auc"] for value in results]), 1)
        self.assertEqual(metrics.select_best_epoch([value["accuracy"] for value in results]), 2)
        self.assertEqual(metrics.select_best_epoch([.5, .7, .7, .6]), 2)
        self.assertEqual(metrics.select_best_epoch([0., 0.]), 1)

    def test_accuracy_training_exactly_matches_original_logits(self):
        graph = small_graph()
        targets = .85 * F.one_hot(graph.noisy_y, graph.num_classes).float() + .15 / graph.num_classes
        torch.manual_seed(321)
        original = train_x_only_mlp(graph, targets, hidden=7, epochs=6)
        torch.manual_seed(321)
        actual = metrics.train_metric_mlp(graph, targets, hidden=7, epochs=6)
        self.assertTrue(torch.equal(actual.logits, original.logits))
        self.assertEqual(actual.best_epoch, original.best_epoch)
        self.assertEqual(actual.logits.device.type, "cpu")
        self.assertEqual(actual.train_metrics["accuracy"], original.train_accuracy)
        self.assertEqual(actual.val_metrics["accuracy"], original.validation_accuracy)
        self.assertEqual(actual.test_metrics["accuracy"], original.test_accuracy)

    def test_auc_two_epoch_run_recomputes_all_final_metrics(self):
        graph = small_graph(2)
        targets = F.one_hot(graph.noisy_y, 2).float()
        torch.manual_seed(29)
        result = metrics.train_metric_mlp(graph, targets, primary_metric="roc_auc", hidden=7, epochs=2)
        self.assertEqual(len(result.history), 2)
        self.assertEqual(result.best_epoch, metrics.select_best_epoch([row["validation_score"] for row in result.history]))
        for split, labels, values in [
            (graph.train_mask, graph.noisy_y, result.train_metrics),
            (graph.val_mask, graph.noisy_y, result.val_metrics),
            (graph.test_mask, graph.y, result.test_metrics),
        ]:
            self.assertEqual(values, metrics.evaluate_logits(result.logits[split], labels[split], binary=True))

    def test_test_metrics_evaluated_only_after_checkpoint_selection(self):
        graph = small_graph(2)
        targets = F.one_hot(graph.noisy_y, 2).float()
        original = metrics.evaluate_logits
        calls = []

        def capture(logits, labels, **kwargs):
            calls.append(labels.clone())
            return original(logits, labels, **kwargs)

        with patch.object(metrics, "evaluate_logits", side_effect=capture):
            metrics.train_metric_mlp(graph, targets, primary_metric="roc_auc", epochs=2, hidden=7)
        # Two validation calls, followed by train, validation, test only once.
        self.assertEqual(len(calls), 5)
        self.assertTrue(torch.equal(calls[-1], graph.y[graph.test_mask]))

    def test_rejects_single_class_and_nonbinary_auc_protocol(self):
        with self.assertRaisesRegex(ValueError, "both classes"):
            metrics.evaluate_logits(torch.zeros(4, 2), torch.zeros(4, dtype=torch.long), binary=True)
        graph = small_graph(2)
        graph.noisy_y[graph.val_mask] = 0
        with self.assertRaisesRegex(ValueError, "both classes"):
            metrics.train_metric_mlp(graph, F.one_hot(graph.noisy_y, 2).float(), primary_metric="roc_auc", epochs=2)
        graph = small_graph(3)
        with self.assertRaisesRegex(ValueError, "two classes|2 classes"):
            metrics.train_metric_mlp(graph, F.one_hot(graph.noisy_y, 3).float(), primary_metric="roc_auc", epochs=2)

    def test_invalid_training_or_scores_rejected(self):
        graph = small_graph()
        targets = F.one_hot(graph.noisy_y, 3).float()
        with self.assertRaisesRegex(ValueError, "epochs"):
            metrics.train_metric_mlp(graph, targets, epochs=0)
        with self.assertRaisesRegex(ValueError, "primary_metric"):
            metrics.train_metric_mlp(graph, targets, primary_metric="test_accuracy")
        for scores in [[], [np.nan], [float("inf")]]:
            with self.assertRaises(ValueError):
                metrics.select_best_epoch(scores)


if __name__ == "__main__":
    unittest.main()
