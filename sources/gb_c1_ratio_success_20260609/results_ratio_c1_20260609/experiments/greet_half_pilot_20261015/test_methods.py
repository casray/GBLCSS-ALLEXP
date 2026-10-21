import unittest
import numpy as np
import torch
from methods import weighted_filters, propagate, return_features, rank_gate, match_budget, GreetPortable, contrastive_loss


class NumericalTests(unittest.TestCase):
    def test_sparse_dense_filters_and_gradients(self):
        edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        weights = torch.tensor([.2, .8, .4, .6], dtype=torch.float64, requires_grad=True)
        x = torch.arange(9, dtype=torch.float64).reshape(3, 3)/10
        indices, low, high = weighted_filters(edges, weights, 3, .1)
        adj = torch.zeros(3, 3, dtype=torch.float64)
        adj[edges[0], edges[1]] = weights
        adj = adj + torch.eye(3, dtype=torch.float64)
        inv = adj.sum(1).rsqrt()
        dense_low = adj*inv[:, None]*inv[None, :]
        opposite = torch.zeros_like(adj)
        opposite[edges[0], edges[1]] = 1-weights
        opposite += torch.eye(3, dtype=torch.float64)
        inv_h = opposite.sum(1).rsqrt()
        mask = torch.zeros_like(adj); mask[edges[0], edges[1]] = 1
        dense_high = torch.eye(3, dtype=torch.float64)-.1*opposite*inv_h[:, None]*inv_h[None, :]*mask
        actual = propagate(x, indices, low)+propagate(x, indices, high)
        expected = dense_low@x+dense_high@x
        torch.testing.assert_close(actual, expected)
        first = torch.autograd.grad(actual.square().sum(), weights, retain_graph=True)[0]
        second = torch.autograd.grad(expected.square().sum(), weights)[0]
        torch.testing.assert_close(first, second)

    def test_walk_cycle_and_isolate(self):
        edges = np.array([[0, 1], [1, 0]])
        se = return_features(edges, 3, hops=4, walks=32, seed=2, block=8)
        np.testing.assert_array_equal(se[:2], [[0, 1, 0, 1], [0, 1, 0, 1]])
        np.testing.assert_array_equal(se[2], np.zeros(4))

    def test_rank_ties_and_budget(self):
        np.testing.assert_allclose(rank_gate(np.ones(4)), .5)
        gate = rank_gate(np.array([2., 1., 1., 3.]))
        np.testing.assert_allclose(gate, [.625, .25, .25, .875])
        alpha = np.array([0., .8, .3, 1.])
        desired = float((alpha*gate).sum())
        matched = match_budget(alpha, gate[::-1], desired)
        self.assertAlmostEqual(float(matched.sum()), desired, places=10)
        self.assertTrue(np.all(matched <= alpha+1e-12))
        np.testing.assert_array_equal(match_budget(np.zeros(4), gate, 0.), np.zeros(4))

    def test_export_is_deterministic_from_fixed_seed(self):
        torch.manual_seed(8)
        model = GreetPortable(4, half_dim=3, projection_dim=3, layers=2)
        x = torch.rand(4, 4)
        se = torch.rand(4, 2)
        edges = torch.tensor([[0,1,1,2,2,3,3,0],[1,0,2,1,3,2,0,3]])
        model = GreetPortable(4, half_dim=3, projection_dim=3, layers=2, se_dim=2)
        model.eval()
        first = model.export(x, se, edges, seed=42)
        second = model.export(x, se, edges, seed=42)
        for a,b in zip(first, second): torch.testing.assert_close(a,b,rtol=0,atol=0)
        indices, low, high = weighted_filters(edges, first[1], len(x), model.highpass_alpha)
        reproduced = model.encode(x, indices, low, high)
        torch.testing.assert_close(reproduced, first[0])

    def test_contrastive_loss_matches_author_negative_denominator(self):
        z1=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
        z2=torch.tensor([[.8,.2],[.1,.9],[.7,.7]])
        ids=torch.arange(3); knn=torch.tensor([[1],[0],[1]])
        positive=(knn[:,:,None]==ids[None,None,:]).any(1)|torch.eye(3,dtype=torch.bool)
        sim=torch.nn.functional.normalize(z1)@torch.nn.functional.normalize(z2).T/.2
        ref=-((sim-torch.log((torch.exp(sim)*(~positive)).sum(1,keepdim=True)))*positive).sum(1).div(positive.sum(1)).mean()
        torch.testing.assert_close(contrastive_loss(z1,z2,ids,knn),ref)

    def test_training_projects_each_channel_separately(self):
        model=GreetPortable(2,half_dim=3,projection_dim=4,layers=2,se_dim=1)
        x=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
        edges=torch.tensor([[0,1,1,2],[1,0,2,1]])
        indices,low,high=weighted_filters(edges,torch.full((4,),.5),3)
        a=model.project_channel(x,indices,low,'low')
        b=model.project_channel(x,indices,high,'high')
        self.assertEqual(tuple(a.shape),(3,4))
        reference=torch.relu(model.low_encoder(x))
        for _ in range(2): reference=propagate(reference,indices,low)
        torch.testing.assert_close(a,model.low_projector(reference))


if __name__=='__main__': unittest.main()
