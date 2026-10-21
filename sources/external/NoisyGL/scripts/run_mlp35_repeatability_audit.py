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


SETTING_SPECS = {
    "cora_pair0p4": {
        "dataset": "cora",
        "noise_type": "pair",
        "noise_rate": 0.4,
        "methods": (
            "mlp_small_loss",
            "frozengrace_gb_mlp_softdist_beta05_small_loss",
        ),
    },
    "amazonpho_pair0p4": {
        "dataset": "amazonpho",
        "noise_type": "pair",
        "noise_rate": 0.4,
        "methods": (
            "mlp_small_loss",
            "frozengrace_gb_mlp_softdist_beta05_small_loss",
        ),
    },
    "pubmed_pair0p4": {
        "dataset": "pubmed",
        "noise_type": "pair",
        "noise_rate": 0.4,
        "methods": (
            "mlp_jocor",
            "frozengrace_gb_mlp_softdist_beta05_small_loss",
        ),
    },
}


def _parse_ints(value):
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def audit_cells(setting_specs, replicates):
    return [
        (setting_id, int(replicate), method)
        for setting_id, spec in setting_specs.items()
        for replicate in replicates
        for method in spec["methods"]
    ]


def _key(row):
    return (str(row["setting_id"]), int(row["replicate"]), str(row["method"]))


def _read_existing(path: Path):
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keyed = {}
    for row in rows:
        key = _key(row)
        if key in keyed:
            raise ValueError(f"duplicate existing repeatability cell: {key}")
        keyed[key] = row
    return keyed


def _run_args(base_args, spec):
    return Namespace(
        device=base_args.device,
        split_type="percent",
        train_percent=0.2,
        val_percent=0.1,
        test_percent=0.7,
        noise_type=spec["noise_type"],
        noise_rate=float(spec["noise_rate"]),
        noise_scope="train",
        config_overrides={},
        debug=False,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Run MLP-35 same-seed repeatability audit")
    parser.add_argument("--replicates", default="0,1,2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stop_on_error", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    replicates = _parse_ints(args.replicates)
    output = Path(args.output)
    keyed = _read_existing(output)
    cells = audit_cells(SETTING_SPECS, replicates)
    for setting_id, replicate, method in cells:
        key = (setting_id, replicate, method)
        if key in keyed and keyed[key].get("status") == "ok":
            print(f"MLP35_REPEAT_SKIP setting={setting_id} replicate={replicate} method={method}")
            continue
        spec = SETTING_SPECS[setting_id]
        run_args = _run_args(args, spec)
        print(f"MLP35_REPEAT_RUN setting={setting_id} replicate={replicate} method={method}")
        try:
            row = run_one(run_args, spec["dataset"], method, seed=0)
        except Exception as exc:
            if args.stop_on_error:
                raise
            row = error_row_for(
                spec["dataset"],
                method,
                0,
                spec["noise_type"],
                spec["noise_rate"],
                exc,
                "train",
            )
        row = dict(row)
        row.update(
            {
                "setting_id": setting_id,
                "replicate": int(replicate),
                "split_type": "percent",
                "train_percent": 0.2,
                "val_percent": 0.1,
                "test_percent": 0.7,
            }
        )
        keyed[key] = row
        ordered = [keyed[cell] for cell in cells if cell in keyed]
        write_csv(output, ordered)
        print(
            f"MLP35_REPEAT_RESULT status={row['status']} valid={row.get('valid', 'NA')} "
            f"test={row.get('test', 'NA')}"
        )
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
