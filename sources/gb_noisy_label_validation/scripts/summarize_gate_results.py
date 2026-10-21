from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noisy_label_gb_grace.evaluation.summary import summarize_gate_results
from noisy_label_gb_grace.utils import PROJECT_ROOT, save_csv


def run(args: argparse.Namespace) -> None:
    summary_rows, paired_rows = summarize_gate_results(
        args.results_root,
        formal_only=not args.include_all,
    )
    save_csv(args.output_dir / "gate_summary.csv", summary_rows)
    save_csv(args.output_dir / "gate_paired_deltas.csv", paired_rows)
    print(f"(S) | Wrote {len(summary_rows)} summary rows and {len(paired_rows)} paired-delta rows to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include old smoke/debug rows instead of filtering to the formal frozen-gate protocol.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
