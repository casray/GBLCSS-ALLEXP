"""Exact finite-population checks; no data downloads or model training."""
from collections import Counter
from fractions import Fraction
from itertools import combinations, permutations
from math import comb, factorial
from pathlib import Path
import json


def check():
    reports = []
    for n in range(2, 8):
        universe = tuple(range(n))
        # Representatives include unequal groups, nested and disjoint groups.
        pairs = [
            (set(range(max(1, n // 2))), set(range(1, n))),
            ({0}, {0}),
            ({0}, {n - 1}),
            (set(universe), {0}),
        ]
        for b, d in pairs:
            chance = Fraction(len(b) * len(d), n)
            truth = len(b & d)
            for k in range(n + 1):
                total_overlap = 0
                total_moved = 0
                total = 0
                for selected in combinations(universe, k):
                    s = set(selected)
                    distribution = Counter()
                    outside_overlap = len((b & d) - s)
                    for destination in permutations(selected):
                        mapping = dict(zip(selected, destination))
                        c = sum(mapping.get(i, i) in d for i in b)
                        distribution[c - outside_overlap] += 1
                        total_overlap += c
                        total_moved += sum(i != j for i, j in mapping.items())
                        total += 1
                    if k:
                        bs, ds = len(b & s), len(d & s)
                        for x in range(bs + 1):
                            numer = (
                                comb(ds, x) * comb(k - ds, bs - x)
                                if x <= ds and 0 <= bs - x <= k - ds else 0
                            )
                            assert Fraction(distribution[x], factorial(k)) == Fraction(
                                numer, comb(k, bs)
                            )
                actual = Fraction(total_overlap, total)
                expected = (
                    Fraction(truth) if k == 0
                    else chance + Fraction(n - k, n - 1) * (truth - chance)
                )
                moved = Fraction(total_moved, total * n)
                expected_moved = Fraction(max(0, k - 1), n)
                assert actual == expected, (n, k, b, d, actual, expected)
                assert moved == expected_moved
                reports.append({
                    "N": n, "k": k, "B": sorted(b), "D": sorted(d),
                    "expected_overlap": str(expected),
                    "enumerated_overlap": str(actual),
                    "expected_moved_fraction": str(moved),
                    "conditional_hypergeometric": "exact match",
                    "enumerated_outcomes": total,
                })
    result = {
        "scope": "Fixed partitions; uniformly chosen size-k subset; uniform permutation within it, fixed points allowed.",
        "case_count": len(reports),
        "all_exact_checks_passed": True,
        "result_does_not_establish": [
            "semantic correctness of significant overlap",
            "monotone optimal group size",
            "neural training performance",
            "validity after correspondence-dependent adaptive grouping",
        ],
        "cases": reports,
    }
    output = Path(__file__).with_name("theory_check_results.json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "case_count": len(reports),
        "all_exact_checks_passed": True,
        "output": str(output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    check()
