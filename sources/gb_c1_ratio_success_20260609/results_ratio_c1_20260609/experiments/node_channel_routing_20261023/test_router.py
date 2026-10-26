import unittest
import numpy as np
from router import fit_model, predict_model, fit_router, apply_route


class RouterTests(unittest.TestCase):
    def example(self):
        rng = np.random.default_rng(11)
        a = rng.dirichlet([2, 2, 2], 80)
        c = rng.dirichlet([2, 2, 2], 80)
        y = np.where(a.max(1) > c.max(1), a.argmax(1), c.argmax(1))
        return a, c, y

    def test_validation_labels_only(self):
        a, c, y = self.example()
        one = fit_router(a, c, np.arange(60), y[:60])
        changed = y.copy(); changed[60:] = (changed[60:] + 1) % 3
        two = fit_router(a, c, np.arange(60), changed[:60])
        np.testing.assert_array_equal(one['score'], two['score'])
        self.assertEqual(one['threshold'], two['threshold'])

    def test_model_serialization(self):
        x = np.arange(40).reshape(20, 2) / 20
        m = fit_model(x, np.arange(20) > 9)
        p = predict_model(m, x)
        self.assertTrue(np.all(np.diff(p) > 0))
        self.assertTrue(np.isfinite(p).all())

    def test_constant_and_empty(self):
        x = np.zeros((4, 2))
        np.testing.assert_array_equal(predict_model(fit_model(x, np.ones(4)), x), 1)
        np.testing.assert_array_equal(predict_model(fit_model(x[:0], np.zeros(0)), x), 0)

    def test_route_keeps_agreements(self):
        a, c = np.array([0, 1, 2]), np.array([0, 2, 1])
        np.testing.assert_array_equal(apply_route(a, c, np.ones(3), None), a)
        np.testing.assert_array_equal(apply_route(a, c, np.ones(3), .5), c)

    def test_all_nodes_receive_outer_prediction(self):
        a, c, y = self.example()
        r = fit_router(a, c, np.arange(60), y[:60])
        self.assertEqual(len(r['nested_pred']), 60)
        self.assertEqual(set(r['fold_ids']), set(range(5)))
        self.assertEqual(len(r['fold_models']), 5)


if __name__ == '__main__':
    unittest.main()
