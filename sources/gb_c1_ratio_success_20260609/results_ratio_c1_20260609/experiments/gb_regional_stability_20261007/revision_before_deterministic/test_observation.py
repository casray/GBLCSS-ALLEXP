import unittest
import torch
from torch.nn import functional as F
from run_stability import RegionalKL, gradient_observation


class ObservationTests(unittest.TestCase):
    def test_observation_preserves_rng_gradients_and_update(self):
        teacher = torch.tensor([[.8, .2], [.3, .7], [.6, .4], [.1, .9]])
        groups = torch.tensor([0, 0, 1, 1])
        region = RegionalKL(teacher, groups)
        outcomes = []
        for observe in (False, True):
            torch.manual_seed(31)
            model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.BatchNorm1d(5),
                                        torch.nn.ReLU(), torch.nn.Dropout(.5), torch.nn.Linear(5, 2))
            optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.0005)
            features = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 12
            for _ in range(3):
                optimizer.zero_grad()
                logits = model(features)
                node_loss = -(teacher * F.log_softmax(logits, 1)).sum(1).mean()
                region_loss = region(logits.softmax(1))
                if observe:
                    rng = torch.get_rng_state().clone()
                    result = gradient_observation(node_loss, region_loss, model.parameters())
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    self.assertTrue(all(p.grad is None for p in model.parameters()))
                    self.assertGreater(result['node_gradient_norm'], 0)
                    self.assertGreaterEqual(result['region_gradient_norm'], 0)
                    self.assertLessEqual(abs(result['gradient_cosine']), 1+1e-12)
                (node_loss+region_loss).backward()
                optimizer.step()
            outcomes.append({k: v.clone() for k,v in model.state_dict().items()})
        for key in outcomes[0]:
            self.assertTrue(torch.equal(outcomes[0][key], outcomes[1][key]), key)


if __name__ == '__main__':
    unittest.main()
