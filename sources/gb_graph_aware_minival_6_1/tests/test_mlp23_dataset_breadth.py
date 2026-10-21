import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from noisy_label_gb_grace.data import load_dataset
from pretrain_ssl_candidates import ssl_defaults


class MLP23DatasetBreadthTests(unittest.TestCase):
    def test_coraml_uses_fixed_cora_grace_defaults(self):
        cfg = ssl_defaults("grace", "CoraML")

        self.assertEqual(cfg["config_source_dataset"], "Cora")
        self.assertIn("fallback uses Cora", cfg["fallback_note"])

    def test_coraml_routes_to_citationfull_cora_ml(self):
        dataset = MagicMock()
        dataset.__getitem__.return_value = MagicMock()
        with patch("noisy_label_gb_grace.data.CitationFull", return_value=dataset) as loader:
            result, _ = load_dataset("CoraML", data_root=Path("test-data"))

        self.assertIs(result, dataset)
        loader.assert_called_once()
        self.assertEqual(loader.call_args.args[0], "test-data")
        self.assertEqual(loader.call_args.args[1], "cora_ml")

    def test_attribwiki_uses_fixed_cora_grace_defaults(self):
        cfg = ssl_defaults("grace", "AttributedWiki")

        self.assertEqual(cfg["config_source_dataset"], "Cora")
        self.assertIn("fallback uses Cora", cfg["fallback_note"])

    def test_attribwiki_routes_to_attributed_graph_wiki(self):
        dataset = MagicMock()
        dataset.__getitem__.return_value = MagicMock()
        with patch("noisy_label_gb_grace.data.AttributedGraphDataset", return_value=dataset) as loader:
            result, _ = load_dataset("AttributedWiki", data_root=Path("test-data"))

        self.assertIs(result, dataset)
        loader.assert_called_once()
        self.assertEqual(loader.call_args.args[0], "test-data")
        self.assertEqual(loader.call_args.args[1], "Wiki")

    def test_citeseerfull_uses_fixed_citeseer_grace_defaults(self):
        cfg = ssl_defaults("grace", "CiteSeerFull")

        self.assertEqual(cfg["config_source_dataset"], "CiteSeer")
        self.assertIn("fallback uses CiteSeer", cfg["fallback_note"])

    def test_citeseerfull_routes_to_citationfull_citeseer(self):
        dataset = MagicMock()
        dataset.__getitem__.return_value = MagicMock()
        with patch("noisy_label_gb_grace.data.CitationFull", return_value=dataset) as loader:
            result, _ = load_dataset("CiteSeerFull", data_root=Path("test-data"))

        self.assertIs(result, dataset)
        loader.assert_called_once()
        self.assertEqual(loader.call_args.args[0], "test-data")
        self.assertEqual(loader.call_args.args[1], "citeseer")


if __name__ == "__main__":
    unittest.main()
