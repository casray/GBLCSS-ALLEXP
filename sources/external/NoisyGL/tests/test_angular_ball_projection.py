import sys
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class AngularBallProjectionTest(unittest.TestCase):
    def test_projection_identity_init_preserves_shape_and_normalizes(self):
        from scripts.train_angular_ball_projection import AngularProjection

        embeddings = torch.tensor([[3.0, 4.0], [1.0, 0.0], [0.0, -2.0]])
        model = AngularProjection(dim=2)

        projected = model(embeddings)

        self.assertEqual(tuple(projected.shape), tuple(embeddings.shape))
        self.assertTrue(torch.allclose(projected.norm(p=2, dim=1), torch.ones(3), atol=1e-6))
        self.assertTrue(torch.allclose(projected, F.normalize(embeddings, p=2, dim=1), atol=1e-6))

    def test_projection_loss_is_finite_and_backpropagates(self):
        from scripts.train_angular_ball_projection import (
            AngularProjection,
            angular_projection_loss,
            build_knn_targets,
        )

        embeddings = F.normalize(torch.randn(8, 4), p=2, dim=1)
        model = AngularProjection(dim=4)
        knn_indices, knn_weights = build_knn_targets(embeddings, k=3)

        loss, metrics = angular_projection_loss(
            model,
            embeddings,
            knn_indices,
            knn_weights,
            tau=0.5,
            drop1=0.1,
            drop2=0.2,
            knn_weight=0.05,
            anchor_weight=1.0,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("contrastive_loss", metrics)
        self.assertIn("knn_loss", metrics)
        self.assertIn("anchor_loss", metrics)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_runner_resolves_feature_and_ball_embedding_paths_separately(self):
        from scripts.run_ssl_swap_downstream import resolve_embedding_paths

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Cora" / "grace").mkdir(parents=True)
            (root / "Cora" / "grace_angproj_knn10_anchor1").mkdir(parents=True)
            feature_path = root / "Cora" / "grace" / "H.pt"
            ball_path = root / "Cora" / "grace_angproj_knn10_anchor1" / "H.pt"
            torch.save(torch.ones(2, 2), feature_path)
            torch.save(torch.ones(2, 2), ball_path)
            args = Namespace(
                embedding_root=str(root),
                ball_embedding_root=None,
                feature_ssl_method="grace",
            )

            resolved_feature, resolved_ball = resolve_embedding_paths(
                args, "cora", "grace_angproj_knn10_anchor1"
            )

            self.assertEqual(resolved_feature, feature_path)
            self.assertEqual(resolved_ball, ball_path)


if __name__ == "__main__":
    unittest.main()
