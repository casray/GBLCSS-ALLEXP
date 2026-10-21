import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from noisy_label_gb_grace.data import load_dataset
from pretrain_ssl_candidates import ssl_defaults


class LastFMAsiaSupportTest(unittest.TestCase):
    def test_ssl_defaults_use_preregistered_amazon_photo_tier_fallback(self):
        cfg = ssl_defaults("grace", "LastFMAsia")

        self.assertEqual(cfg["epochs"], 1500)
        self.assertEqual(cfg["hidden_dim"], 256)
        self.assertAlmostEqual(cfg["tau"], 0.7)
        self.assertEqual(cfg["config_source_dataset"], "PubMed")
        self.assertIn("fallback uses PubMed", cfg["fallback_note"])

    def test_upstream_loader_routes_lastfmasia_to_native_dataset(self):
        dataset = unittest.mock.MagicMock()
        dataset.__getitem__.return_value = unittest.mock.MagicMock()
        with patch("noisy_label_gb_grace.data.LastFMAsia", return_value=dataset) as loader:
            result, _ = load_dataset("LastFMAsia", data_root=Path("test-data"))

        self.assertIs(result, dataset)
        loader.assert_called_once()
        self.assertEqual(loader.call_args.args[0], "test-data")


if __name__ == "__main__":
    unittest.main()
