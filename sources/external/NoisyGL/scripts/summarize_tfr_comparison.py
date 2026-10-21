from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_phase24_fairness import (
    METHOD_LABELS,
    aggregate_rows,
    load_raw_rows,
    paired_deltas,
    write_csv,
)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _pp(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.2f}pp"


def write_report(path: Path, aggregate: list[dict], deltas: list[dict], args) -> None:
    lines = [
        "# Phase 25: TFR Same-Protocol Comparison",
        "",
        "## Protocol",
        "",
        "- External repository: `https://github.com/eaglelab-zju/TFR.git`",
        "- Cloned commit: `d7e9ff8a58d457b519f068c3d170cd37579c8bd8`",
        "- Method integrated: `tfr` / TFR-GCN.",
        "- Important caveat: the official TFR repo does not provide Cora/CiteSeer configs; "
        "this run transfers the official `tfr_gcn_dblp.yaml` hyperparameter template to Cora/CiteSeer without tuning.",
        "- Split: `percent`, train/val/test = `0.2/0.1/0.7`.",
        "- Noise: `uniform`, noise rate = `0.4`; seeds `0,1,2,3,4`.",
        "",
        "## Mean ± Std",
        "",
        "| Dataset | Method | Test acc |",
        "|---|---|---:|",
    ]
    keep_methods = ["nrgnn", "nrgnn_xh", "frozengrace_gb_gcn_softdist_xh", "tfr"]
    for dataset in sorted({row["dataset"] for row in aggregate}):
        for method in keep_methods:
            row = next((r for r in aggregate if r["dataset"] == dataset and r["method"] == method), None)
            if row is None:
                continue
            lines.append(f"| {dataset} | {METHOD_LABELS[method]} | {_fmt(row['mean'])} ± {_fmt(row['std'])} |")

    lines.extend([
        "",
        "## Paired Deltas",
        "",
        "| Dataset | Comparison | Mean delta | Positive seeds | Per-seed deltas |",
        "|---|---|---:|---:|---|",
    ])
    for row in deltas:
        lines.append(
            f"| {row['dataset']} | {row['left_label']} - {row['right_label']} | "
            f"{_pp(row['mean_delta'])} | {row['positive_seed_count']}/{row['n']} | {row['deltas']} |"
        )

    lines.extend([
        "",
        "## Decision",
        "",
        "- TFR is runnable under the local NoisyGL percent-split protocol after a minimal GCN-version integration.",
        "- Under the transferred official DBLP template, TFR does not beat SoftDist [X||H] on Cora and is slightly above SoftDist [X||H] on CiteSeer, but remains below NRGNN [X||H].",
        "- Because the official repo lacks Cora/CiteSeer tuned configs, these numbers should be reported as a same-protocol transferred-config comparison, not as definitive tuned TFR performance.",
        "",
        "## Artifacts",
        "",
        f"- Raw TFR runs: `{args.tfr_raw}`",
        f"- Aggregate: `{path.parent / 'tfr_comparison_aggregate.csv'}`",
        f"- Paired deltas: `{path.parent / 'tfr_comparison_paired_deltas.csv'}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TFR same-protocol comparison.")
    parser.add_argument("--base_raw", default="runs/softdist_5seed_rocm_20260610/softdist_5seed_rocm_raw_summary_20260610.csv")
    parser.add_argument("--xh_raw", default="runs/xh_5seed_rocm_20260611/xh_5seed_raw_summary_20260611.csv")
    parser.add_argument("--nrgnn_xh_raw", default="runs/phase24_nrgnn_xh_20260616/nrgnn_xh_5seed.csv")
    parser.add_argument("--tfr_raw", default="runs/tfr_comparison_20260616/tfr_5seed.csv")
    parser.add_argument("--output_dir", default="runs/tfr_comparison_20260616")
    args = parser.parse_args()

    rows = load_raw_rows([args.base_raw, args.xh_raw, args.nrgnn_xh_raw, args.tfr_raw])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_rows(rows)
    deltas = paired_deltas(rows, [
        ("tfr", "frozengrace_gb_gcn_softdist_xh"),
        ("tfr", "nrgnn_xh"),
        ("tfr", "nrgnn"),
        ("frozengrace_gb_gcn_softdist_xh", "tfr"),
    ])
    write_csv(output_dir / "tfr_comparison_combined_raw.csv", rows)
    write_csv(output_dir / "tfr_comparison_aggregate.csv", aggregate)
    write_csv(output_dir / "tfr_comparison_paired_deltas.csv", deltas)
    write_report(output_dir / "tfr_comparison_report.md", aggregate, deltas, args)
    print(f"WROTE {output_dir / 'tfr_comparison_report.md'}")


if __name__ == "__main__":
    main()
