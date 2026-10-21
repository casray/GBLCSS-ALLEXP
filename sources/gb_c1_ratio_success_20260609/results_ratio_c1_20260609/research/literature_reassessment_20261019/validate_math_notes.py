"""Finite enumerations checking algebra in the design note, not ML experiments."""
import itertools
import json
from pathlib import Path

import numpy as np


def gini_sum(labels, partition, classes):
    total = 0.0
    for group in partition:
        if group:
            q = np.bincount(labels[group], minlength=classes) / len(group)
            total += len(group) * (1.0 - q @ q)
    return total


def main():
    checks = []
    for classes, clean in ((2, [0, 0, 1, 1]), (3, [0, 0, 1, 1, 2])):
        clean = np.asarray(clean)
        n = len(clean)
        # Empty group means an existing ball with zero training labels.
        partitions = [[list(range(n))], [[0, 1], list(range(2, n))],
                      [[0], [1], list(range(2, n)), []]]
        for eta in (0.0, 0.2, (classes - 1) / classes, 0.8):
            gamma = 1 - classes * eta / (classes - 1)
            variance_trace = 1 - ((1 - eta) ** 2 + eta ** 2 / (classes - 1))
            for partition in partitions:
                expectation = 0.0
                probability_sum = 0.0
                for labels in itertools.product(range(classes), repeat=n):
                    labels = np.asarray(labels)
                    probability = np.prod(np.where(labels == clean, 1 - eta, eta / (classes - 1)))
                    expectation += probability * gini_sum(labels, partition, classes)
                    probability_sum += probability
                labeled_balls = sum(bool(group) for group in partition)
                predicted = (gamma ** 2 * gini_sum(clean, partition, classes)
                             + (1 - gamma ** 2) * (1 - 1 / classes) * n
                             - variance_trace * labeled_balls)
                assert abs(probability_sum - 1) < 1e-12
                assert abs(expectation - predicted) < 1e-12
                checks.append({"classes": classes, "eta": eta, "balls": len(partition),
                               "labeled_balls": labeled_balls,
                               "expected_gini_enumerated": float(expectation),
                               "formula_value": float(predicted),
                               "absolute_error": float(abs(expectation - predicted))})

    rng = np.random.default_rng(20261019)
    n, m = 7, 3
    h = np.eye(m)[[0, 1, 1, 2, 2, 2, 2]]
    r = np.linalg.solve(h.T @ h, h.T)
    projector = h @ r
    a = rng.uniform(0, 1, (n, n))
    a = (a + a.T) / 2 + np.eye(n)
    scale = 1 / np.sqrt(a.sum(axis=1))
    p = scale[:, None] * a * scale[None, :]
    pc = r @ p @ h
    u = rng.normal(size=(n, 4))
    left = np.linalg.norm(p @ u - h @ pc @ r @ u, "fro")
    right = (np.linalg.norm((np.eye(n) - projector) @ p @ u, "fro")
             + np.linalg.norm(p, 2) * np.linalg.norm((np.eye(n) - projector) @ u, "fro"))
    assert left <= right + 1e-12
    assert np.allclose(projector, projector.T)
    assert np.allclose(projector @ projector, projector)

    ac = h.T @ a @ h
    coarse_scale = 1 / np.sqrt(ac.sum(axis=1))
    alternate = coarse_scale[:, None] * ac * coarse_scale[None, :]
    operator_difference = np.linalg.norm(pc - alternate, "fro")
    assert operator_difference > 1e-8

    train = np.array([0, 1, 2, 3, 5])
    labels = np.array([0, 1, 2, 0, 2])
    logits = rng.normal(size=(m, 3))
    logits -= logits.max(axis=1, keepdims=True)
    log_prob = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    membership = h.argmax(axis=1)
    counts = np.zeros((m, 3))
    np.add.at(counts, (membership[train], labels), 1)
    node_loss = -log_prob[membership[train], labels].sum()
    coarse_loss = -(counts * log_prob).sum()
    assert abs(node_loss - coarse_loss) < 1e-12

    result = {"status": "passed", "kind": "finite algebra checks; no training or performance evidence",
              "gini_cases": checks, "max_gini_error": max(x["absolute_error"] for x in checks),
              "propagation_error": float(left), "propagation_upper_bound": float(right),
              "normalized_adjacency_vs_rph_difference": float(operator_difference),
              "node_ce": float(node_loss), "coarse_ce": float(coarse_loss)}
    path = Path(__file__).resolve().parent / "math_sanity_checks.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "gini_cases"}, indent=2))


if __name__ == "__main__":
    main()
