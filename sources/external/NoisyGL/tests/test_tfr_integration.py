import unittest

import torch


class TFRIntegrationTest(unittest.TestCase):
    def test_decoder_returns_reconstructed_features_and_confidence(self):
        from predictor.module.TFR import TFRDecoder

        decoder = TFRDecoder(in_channels=3, hidden_channels=4, out_channels=5)
        x = torch.softmax(torch.randn(6, 3), dim=1)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 0],
        ])

        recon, confidence = decoder(x, edge_index)

        self.assertEqual(tuple(recon.shape), (6, 5))
        self.assertEqual(confidence.shape[0], 6)

    def test_tfr_predictor_aliases_exist(self):
        from predictor.TFR_Predictor import tfr_Predictor, tfr_gcn_Predictor

        self.assertTrue(issubclass(tfr_gcn_Predictor, tfr_Predictor))


if __name__ == "__main__":
    unittest.main()
