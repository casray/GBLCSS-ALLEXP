import unittest

import torch


class ProConReimplementationTest(unittest.TestCase):
    def test_exact_peers_apply_two_hop_structure_weighting(self):
        from predictor.module.ProCon import build_peer_index_exact

        features = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.6],
            [1.0, 0.0],
        ])
        edge_index = torch.tensor([
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ])

        peers = build_peer_index_exact(features, edge_index, k=1, alpha=0.2)

        self.assertEqual(int(peers[0, 0]), 2)

    def test_bounded_peers_match_exact_when_candidate_limit_covers_graph(self):
        from predictor.module.ProCon import build_peer_index_bounded, build_peer_index_exact

        generator = torch.Generator().manual_seed(3)
        features = torch.randn(24, 8, generator=generator)
        src = torch.arange(23)
        edge_index = torch.stack([torch.cat([src, src + 1]), torch.cat([src + 1, src])])

        exact = build_peer_index_exact(features, edge_index, k=5, alpha=0.5)
        bounded = build_peer_index_bounded(
            features,
            edge_index,
            k=5,
            alpha=0.5,
            candidate_limit=24,
            random_seed=3,
        )
        overlap = [
            len(set(exact[i].tolist()) & set(bounded[i].tolist())) / 5
            for i in range(features.shape[0])
        ]

        self.assertGreaterEqual(sum(overlap) / len(overlap), 0.95)

    def test_gmm_selects_high_consistency_component(self):
        from predictor.module.ProCon import gmm_clean_probability

        scores = torch.tensor([0.02, 0.05, 0.1, 0.88, 0.93, 0.98])
        probability = gmm_clean_probability(scores, random_seed=5)

        self.assertLess(float(probability[:3].mean()), 0.2)
        self.assertGreater(float(probability[3:].mean()), 0.8)

    def test_prototype_ema_retains_missing_classes(self):
        from predictor.module.ProCon import update_prototypes_ema

        prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        embeddings = torch.tensor([[0.0, 2.0], [0.0, 4.0]])
        labels = torch.tensor([0, 0])
        weights = torch.tensor([1.0, 1.0])

        updated = update_prototypes_ema(
            prototypes,
            embeddings,
            labels,
            weights,
            momentum=0.5,
            n_classes=2,
        )

        self.assertTrue(torch.allclose(updated[0], torch.tensor([0.5, 1.5])))
        self.assertTrue(torch.equal(updated[1], prototypes[1]))

    def test_ema_update_moves_target_toward_online(self):
        from predictor.module.ProCon import ema_update

        online = torch.nn.Linear(2, 2, bias=False)
        target = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            online.weight.fill_(2.0)
            target.weight.zero_()

        ema_update(target, online, momentum=0.75)

        self.assertTrue(torch.allclose(target.weight, torch.full_like(target.weight, 0.5)))

    def test_predictor_alias_exists(self):
        from predictor.ProCon_Predictor import procon_Predictor

        self.assertIsNotNone(procon_Predictor)


if __name__ == "__main__":
    unittest.main()
