"""Small checks for the frozen stage-one protocol."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor" / "mgbcc_main"))
from granular.tools import relation_of_views_gblists_tensor  # noqa: E402
from diagnose_r0 import noise_permutation  # noqa: E402


class RuleAuditTest(unittest.TestCase):
    def test_original_rule_matches_contingency_formula(self) -> None:
        a = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        b = np.array([0, 0, 1, 0, 1, 1, 2, 2, 2, 2, 1, 2])
        k = 3
        table = np.zeros((k, k), dtype=np.int32)
        np.add.at(table, (a, b), 1)
        na, nb = np.bincount(a, minlength=k), np.bincount(b, minlength=k)
        expected = table / np.minimum(na[:, None], nb[None, :]) > .1

        class FakeBallList:
            def __init__(self, parts: np.ndarray):
                self.y_parts = torch.as_tensor(parts)

            def __len__(self) -> int:
                return k

        actual = relation_of_views_gblists_tensor(FakeBallList(a), FakeBallList(b))
        np.testing.assert_array_equal(actual.numpy().astype(bool), expected)

    def test_noise_permutation_preserves_bijection(self) -> None:
        n = 1000
        perm = noise_permutation(n, .4, 0)
        self.assertEqual(np.unique(perm).size, n)
        self.assertLessEqual(np.count_nonzero(perm != np.arange(n)), 400)
        self.assertGreaterEqual(np.count_nonzero(perm != np.arange(n)), 390)


if __name__ == "__main__":
    unittest.main()
