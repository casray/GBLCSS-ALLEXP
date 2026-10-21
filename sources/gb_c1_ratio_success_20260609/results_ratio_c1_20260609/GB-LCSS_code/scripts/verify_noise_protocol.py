from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gblcss.data import inject_train_label_noise, load_public_dataset


DATASET_NAMES = {
    "amazonpho": "AmazonPhoto",
    "citeseer": "CiteSeer",
    "coauthorcs": "CoauthorCS",
    "cora": "Cora",
    "corafull": "CoraFull",
}

FORMAL_SIGNATURES = {
    ("cora", 0, "uniform", 0.4): {
        "split": "fe241a5bd04c1aba7c5312aa410fc4c653a2c14dd68e58e9015b5271eb4c67f9",
        "noise": "eb786f64ad9bf24efea0826e657708af4a1f081023c1eca47b0106fc2765d3a1",
    },
    ("corafull", 0, "uniform", 0.6): {
        "split": "947dc6d3f86b92f3cfafeec28078957e531addaa4b0ff2ed764373e78273da12",
        "noise": "d8826084bac32eba5dff58b47e85a9a567f827b36bd1fa44c8f38405e71f0715",
    },
}


def _fingerprint(*arrays: object) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.asarray(array, dtype=np.int64).reshape(-1)
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the fixed-seed split and label-noise protocol.")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()

    archived = pd.read_csv(ROOT / "results" / "figure_data" / "fig4_target_quality_per_seed.csv")
    mismatches: list[tuple[object, ...]] = []
    signature_checks = 0

    for (dataset, noise_type, noise_rate, seed), rows in archived.groupby(
        ["dataset", "noise_type", "noise_rate", "seed"], sort=True
    ):
        graph = load_public_dataset(DATASET_NAMES[dataset], args.data_root, int(seed))
        inject_train_label_noise(graph, str(noise_type), float(noise_rate), int(seed))
        if graph.noisy_y is None or graph.train_index is None:
            raise AssertionError("noise injection did not preserve the ordered training indices")

        changed = (graph.noisy_y[graph.train_index] != graph.y[graph.train_index]).float().mean().item()
        actual_brier = 2.0 * changed
        expected_brier = float(rows.iloc[0].noisy_onehot_brier)
        if abs(actual_brier - expected_brier) > 5e-7:
            mismatches.append((dataset, noise_type, noise_rate, seed, expected_brier, actual_brier))

        signature = FORMAL_SIGNATURES.get((dataset, int(seed), str(noise_type), float(noise_rate)))
        if signature is not None:
            split_hash = _fingerprint(graph.train_index, graph.val_index, graph.test_index)
            noise_hash = _fingerprint(graph.y, graph.noisy_y)
            if split_hash != signature["split"] or noise_hash != signature["noise"]:
                raise AssertionError(
                    f"formal signature mismatch for {(dataset, seed, noise_type, noise_rate)}"
                )
            signature_checks += 1

    if mismatches:
        raise AssertionError(f"Fig. 4 noisy-label Brier mismatches: {mismatches[:5]}")
    print(f"noise protocol verification: {len(archived)}/125 rows matched; signatures={signature_checks}")


if __name__ == "__main__":
    main()
