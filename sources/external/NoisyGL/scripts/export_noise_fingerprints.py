#!/usr/bin/env python3
"""Export split and noise fingerprints without training a predictor."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_standard_downstream import (
    _build_noisy_labels,
    _load_dataset,
    _noise_fingerprint,
    _noise_stats,
    _noise_subset_fingerprint,
    _parse_int_list,
    _parse_str_list,
    _split_fingerprint,
)
from utils.tools import setup_seed


def build_rows(
        *,
        datasets: Sequence[str],
        seeds: Sequence[int],
        device: str,
        noise_type: str,
        noise_rate: float,
        split_type: str,
        train_percent: float,
        val_percent: float,
        test_percent: float) -> list[dict]:
    rows = []
    for dataset_name in datasets:
        for seed in seeds:
            setup_seed(int(seed))
            data = _load_dataset(
                dataset_name,
                device=device,
                split_type=split_type,
                train_percent=train_percent,
                val_percent=val_percent,
                test_percent=test_percent,
            )
            noisy_label, modified_mask = _build_noisy_labels(
                labels=data.labels,
                features=data.feats,
                n_classes=data.n_classes,
                noise_type=noise_type,
                noise_rate=noise_rate,
                seed=int(seed),
                noise_scope="train",
                train_mask=data.train_masks,
                val_mask=data.val_masks,
                debug=False,
            )
            rows.append({
                "dataset": str(dataset_name).lower(),
                "seed": int(seed),
                "noise_type": noise_type,
                "noise_rate": float(noise_rate),
                "noise_scope": "train",
                "split_type": split_type,
                "train_percent": float(train_percent),
                "val_percent": float(val_percent),
                "test_percent": float(test_percent),
                "split_hash": _split_fingerprint(
                    data.train_masks,
                    data.val_masks,
                    data.test_masks,
                ),
                "noise_hash": _noise_fingerprint(data.labels, noisy_label),
                "train_noise_hash": _noise_subset_fingerprint(
                    data.labels,
                    noisy_label,
                    data.train_masks,
                ),
                "val_noise_hash": _noise_subset_fingerprint(
                    data.labels,
                    noisy_label,
                    data.val_masks,
                ),
                **_noise_stats(
                    data.labels,
                    noisy_label,
                    modified_mask,
                    data.train_masks,
                    data.val_masks,
                    data.test_masks,
                ),
            })
    return rows


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot write an empty fingerprint manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--noise_type", default="uniform")
    parser.add_argument("--noise_rate", type=float, default=0.4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent")
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = build_rows(
        datasets=_parse_str_list(args.datasets),
        seeds=_parse_int_list(args.seeds),
        device=args.device,
        noise_type=args.noise_type,
        noise_rate=args.noise_rate,
        split_type=args.split_type,
        train_percent=args.train_percent,
        val_percent=args.val_percent,
        test_percent=args.test_percent,
    )
    write_csv(args.output, rows)
    print(f"WROTE {len(rows)} protocol fingerprint rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
