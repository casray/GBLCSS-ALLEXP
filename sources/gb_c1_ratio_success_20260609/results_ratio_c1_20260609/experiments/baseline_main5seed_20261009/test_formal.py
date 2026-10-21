from pathlib import Path
import importlib.util
import unittest


class FormalTests(unittest.TestCase):
    def test_parameterized_protocol_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('protocol'),
                             'A seed/rate-aware protocol module is required')

    def test_split_noise_seed_and_scope(self):
        from protocol import load_data
        hashes = []
        for seed in range(5):
            _, record = load_data('cora', seed, .4, device='cpu')
            self.assertEqual(record['modified_val_count'], 0)
            self.assertEqual(record['modified_test_count'], 0)
            self.assertTrue(record['historical_hashes_verified'])
            hashes.append((record['split_hash'], record['noise_hash']))
        self.assertEqual(len(set(hashes)), 5)
        _, high = load_data('cora', 1, .6, device='cpu')
        self.assertEqual(high['split_hash'], hashes[1][0])
        self.assertNotEqual(high['noise_hash'], hashes[1][1])

    def test_tined_seed_and_order(self):
        from tined_formal import effective_config, ordered_training_indices
        import torch
        for seed in range(5):
            self.assertEqual(effective_config('cora', False, seed)['seed'], seed)
        torch.testing.assert_close(ordered_training_indices(torch.tensor([8, 1, 5])),
                                   torch.tensor([1, 5, 8]))

    def test_pretrained_encoder_fixed(self):
        from sim_probe import build_probe
        from protocol import load_data
        data, _ = load_data('cora', 0, .4, device='cpu')
        first, head0, cfg = build_probe(data, 0)
        second, head1, _ = build_probe(data, 1)
        import torch
        for key, value in first.encoder.state_dict().items():
            torch.testing.assert_close(value, second.encoder.state_dict()[key], rtol=0, atol=0)
        self.assertFalse(torch.equal(head0.weight, head1.weight))
        self.assertFalse(first.encoder.training)
        self.assertTrue(all(not p.requires_grad for p in first.encoder.parameters()))
        self.assertEqual(cfg['pretrain_seed'], 0)


if __name__ == '__main__':
    unittest.main()
