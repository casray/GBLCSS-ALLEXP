from __future__ import annotations

import argparse
import csv
import sys
from argparse import Namespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_standard_downstream import error_row_for, run_one, write_csv


METHOD = "frozengrace_gb_mlp_softdist_beta05_small_loss"


def _parse_list(value, cast=str):
    return tuple(cast(item.strip()) for item in str(value).split(",") if item.strip())


def grid_cells(datasets, noise_rates, seeds, forget_rates):
    return [
        (str(dataset), float(noise_rate), int(seed), float(forget_rate))
        for dataset in datasets
        for noise_rate in noise_rates
        for seed in seeds
        for forget_rate in forget_rates
    ]


def _key(row):
    return (
        str(row["dataset"]),
        float(row["noise_rate"]),
        int(row["seed"]),
        float(row["requested_forget_rate"]),
    )


def _read_existing(path: Path):
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keyed = {}
    for row in rows:
        key = _key(row)
        if key in keyed:
            raise ValueError(f"duplicate existing grid cell: {key}")
        keyed[key] = row
    return keyed


def _args_for(base_args, noise_rate, forget_rate):
    return Namespace(
        device=base_args.device,
        split_type=base_args.split_type,
        train_percent=base_args.train_percent,
        val_percent=base_args.val_percent,
        test_percent=base_args.test_percent,
        noise_type="uniform",
        noise_rate=float(noise_rate),
        noise_scope="train",
        config_overrides={"model.forget_rate": float(forget_rate)},
        debug=False,
    )


def _annotate(row, base_args, forget_rate):
    row = dict(row)
    row.update(
        {
            "split_type": base_args.split_type,
            "train_percent": float(base_args.train_percent),
            "val_percent": float(base_args.val_percent),
            "test_percent": float(base_args.test_percent),
            "requested_forget_rate": float(forget_rate),
            "selection_role": (
                "fixed_0p5" if abs(float(forget_rate) - 0.5) < 1e-12 else "candidate"
            ),
        }
    )
    return row


def build_parser():
    parser = argparse.ArgumentParser(description="Run the resumable MLP-35 clean-val grid")
    parser.add_argument("--datasets", default="cora,citeseer,corafull,amazonpho,pubmed")
    parser.add_argument("--noise_rates", default="0.2,0.4,0.6")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--forget_rates", default="0.0,0.2,0.4,0.5,0.6")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split_type", default="percent")
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stop_on_error", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    datasets = _parse_list(args.datasets)
    noise_rates = _parse_list(args.noise_rates, float)
    seeds = _parse_list(args.seeds, int)
    forget_rates = _parse_list(args.forget_rates, float)
    output = Path(args.output)
    keyed = _read_existing(output)

    for dataset, noise_rate, seed, forget_rate in grid_cells(
        datasets, noise_rates, seeds, forget_rates
    ):
        key = (dataset, noise_rate, seed, forget_rate)
        if key in keyed and keyed[key].get("status") == "ok":
            print(f"MLP35_SKIP dataset={dataset} noise={noise_rate} seed={seed} forget={forget_rate}")
            continue
        print(f"MLP35_RUN dataset={dataset} noise={noise_rate} seed={seed} forget={forget_rate}")
        run_args = _args_for(args, noise_rate, forget_rate)
        try:
            row = run_one(run_args, dataset, METHOD, seed)
        except Exception as exc:
            if args.stop_on_error:
                raise
            row = error_row_for(
                dataset,
                METHOD,
                seed,
                "uniform",
                noise_rate,
                exc,
                "train",
            )
        row = _annotate(row, args, forget_rate)
        keyed[key] = row
        ordered = [keyed[cell] for cell in grid_cells(datasets, noise_rates, seeds, forget_rates) if cell in keyed]
        write_csv(output, ordered)
        print(
            f"MLP35_RESULT status={row['status']} valid={row.get('valid', 'NA')} "
            f"test={row.get('test', 'NA')}"
        )
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
