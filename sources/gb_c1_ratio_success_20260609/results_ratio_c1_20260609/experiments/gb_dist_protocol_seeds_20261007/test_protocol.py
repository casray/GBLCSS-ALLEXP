import unittest
import torch
import run_protocol as run


class ProtocolTests(unittest.TestCase):
    def test_all_historical_hashes_and_disjoint_splits(self):
        seen = {}
        for dataset, rate, seed in run.cells():
            data, meta, assignment = run.load_data(dataset, rate, seed)
            counts = data.train_mask.int()+data.val_mask.int()+data.test_mask.int()
            self.assertLessEqual(int(counts.max()), 1)
            self.assertEqual(len(assignment), data.num_nodes)
            self.assertTrue(torch.equal(data.y[~data.train_mask], data.noisy_y[~data.train_mask]))
            seen[(dataset,rate,seed)] = meta
        for dataset in run.DATASETS:
            for rate in (.4,.6):
                self.assertNotEqual(seen[dataset,rate,1]['split_hash'], seen[dataset,rate,2]['split_hash'])
                self.assertNotEqual(seen[dataset,rate,1]['noise_hash'], seen[dataset,rate,2]['noise_hash'])

    def test_seed_zero_matches_previous_protocol(self):
        a, meta, _ = run.load_data('cora', .4, 0)
        b, prior = run.previous.load_data('cora', .4)
        self.assertEqual(meta['split_hash'], prior['split_hash'])
        self.assertEqual(meta['noise_hash'], prior['noise_hash'])
        self.assertTrue(torch.equal(a.x,b.x))

    def test_reproducible_protocol(self):
        a, am, aa = run.load_data('citeseer', .6, 2)
        b, bm, ba = run.load_data('citeseer', .6, 2)
        self.assertEqual(am,bm)
        self.assertTrue(torch.equal(a.noisy_y,b.noisy_y))
        self.assertTrue(torch.equal(aa,ba))


if __name__ == '__main__':
    unittest.main()
