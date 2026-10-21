from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.extract_deployable_proxies import _pearson, _safe_float, _spearman


FROZEN_PROXIES = [
    "mean_vote_margin_weighted",
    "mean_vote_entropy_weighted",
    "ball_gcn_disagree_confident_train",
]

TARGETS_BY_CONFIG = {
    "frozengrace_gb_gcn_softdist_xh": [
        "softdist_vs_gcn_xh_mean",
        "softdist_vs_shuffled_mean",
    ],
    "gbselect": [
        "gbselect_vs_gcn_only_mean",
        "gbselect_vs_shuffled_mean",
    ],
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _target_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("dataset", "")).strip().lower(): row for row in rows if row.get("dataset")}


def _proxy_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (str(row.get("dataset", "")).strip().lower(), str(row.get("proxy_config", ""))): row
        for row in rows
        if row.get("dataset") and row.get("proxy_config") and row.get("status") == "ok"
    }


def build_scatter_rows(proxy_rows: list[dict[str, str]], target_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    proxies = _proxy_lookup(proxy_rows)
    targets = _target_lookup(target_rows)
    rows: list[dict[str, object]] = []
    for (dataset, config), proxy_row in sorted(proxies.items()):
        if config not in TARGETS_BY_CONFIG:
            continue
        target_row = targets.get(dataset, {})
        for target in TARGETS_BY_CONFIG[config]:
            y = _safe_float(target_row.get(target))
            if y is None:
                continue
            for proxy in FROZEN_PROXIES:
                x = _safe_float(proxy_row.get(f"{proxy}_mean"))
                if x is None:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "proxy_config": config,
                        "target": target,
                        "proxy": proxy,
                        "proxy_value": x,
                        "target_delta": y,
                    }
                )
    return rows


def build_correlation_rows(scatter_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    keys = sorted({(row["proxy_config"], row["target"], row["proxy"]) for row in scatter_rows})
    for config, target, proxy in keys:
        group = [
            row
            for row in scatter_rows
            if row["proxy_config"] == config and row["target"] == target and row["proxy"] == proxy
        ]
        x = [float(row["proxy_value"]) for row in group]
        y = [float(row["target_delta"]) for row in group]
        datasets = [str(row["dataset"]) for row in group]
        rows.append(
            {
                "proxy_config": config,
                "target": target,
                "proxy": proxy,
                "n": len(group),
                "spearman": _spearman(x, y),
                "pearson": _pearson(x, y) if len(group) >= 2 else math.nan,
                "datasets": " ".join(datasets),
                "power_note": "decision_eligible" if len(group) >= 10 else ("underpowered" if len(group) < 7 else "informative_but_below_adaptive_bar"),
            }
        )
    return rows


def _fmt(value, digits: int = 4) -> str:
    value = _safe_float(value)
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def write_svg_scatter(path: Path, rows: list[dict[str, object]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"640\" height=\"420\"></svg>\n", encoding="utf-8")
        return
    xs = np.asarray([float(row["proxy_value"]) for row in rows], dtype=np.float64)
    ys = np.asarray([float(row["target_delta"]) for row in rows], dtype=np.float64)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    if abs(x_max - x_min) < 1e-12:
        x_min -= 0.5
        x_max += 0.5
    if abs(y_max - y_min) < 1e-12:
        y_min -= 0.01
        y_max += 0.01
    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.12
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad
    width, height = 760, 500
    left, right, top, bottom = 80, 30, 50, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-family="Arial" font-size="18" font-weight="600">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{sy(0.0)}" x2="{left + plot_w}" y2="{sy(0.0)}" stroke="#999" stroke-width="1" stroke-dasharray="4 4"/>',
        f'<text x="{left + plot_w / 2 - 40}" y="{height - 20}" font-family="Arial" font-size="13">deployable proxy</text>',
        f'<text x="16" y="{top + plot_h / 2}" font-family="Arial" font-size="13" transform="rotate(-90 16 {top + plot_h / 2})">ball gain</text>',
        f'<text x="{left}" y="{top + plot_h + 18}" font-family="Arial" font-size="11">{x_min:.3f}</text>',
        f'<text x="{left + plot_w - 38}" y="{top + plot_h + 18}" font-family="Arial" font-size="11">{x_max:.3f}</text>',
        f'<text x="28" y="{sy(y_min)}" font-family="Arial" font-size="11">{y_min:.3f}</text>',
        f'<text x="28" y="{sy(y_max)}" font-family="Arial" font-size="11">{y_max:.3f}</text>',
    ]
    for row in rows:
        x = sx(float(row["proxy_value"]))
        y = sy(float(row["target_delta"]))
        label = str(row["dataset"])
        color = "#2f6fbb" if float(row["target_delta"]) >= 0 else "#b23b3b"
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}" opacity="0.85"/>')
        parts.append(f'<text x="{x + 6:.2f}" y="{y - 6:.2f}" font-family="Arial" font-size="11">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_report(path: Path, corr_rows: list[dict[str, object]], scatter_rows: list[dict[str, object]]) -> None:
    max_n = max([int(row["n"]) for row in corr_rows], default=0)
    strong = [
        row
        for row in corr_rows
        if int(row["n"]) >= 10 and _safe_float(row.get("spearman")) is not None and abs(float(row["spearman"])) >= 0.7
    ]
    lines = [
        "# Phase 38 Evidence Matrix: Frozen Deployable Proxy Correlations",
        "",
        "Only the three pre-registered deployable proxies are reported:",
        "",
        "- `mean_vote_margin_weighted`",
        "- `mean_vote_entropy_weighted`",
        "- `ball_gcn_disagree_confident_train`",
        "",
        "Clean labels are not used on the proxy side. Target deltas are downstream evaluation results.",
        "",
        "## Correlations",
        "",
        "| Method | Target | Proxy | n | Spearman | Pearson | Power note |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in corr_rows:
        lines.append(
            f"| {row['proxy_config']} | {row['target']} | {row['proxy']} | {row['n']} | {_fmt(row['spearman'])} | {_fmt(row['pearson'])} | {row['power_note']} |"
        )
    lines.extend(["", "## Judge", ""])
    if max_n < 10:
        lines.append(
            f"Maximum n is {max_n}, below the pre-registered n>=10 bar. Adaptive layer is not evaluated; keep proxy gating as future work."
        )
    elif strong:
        lines.append(
            "At least one frozen proxy reaches the pre-registered n>=10 and |Spearman|>=0.7 bar. A minimal binary switch may be evaluated next."
        )
    else:
        lines.append(
            f"n>=10 is available, but none of the frozen proxies reaches |Spearman|>=0.7. Adaptive trigger should be closed and discussed as future work."
        )
    lines.extend(
        [
            "",
            "Claim boundary: evidence can be described as preliminary deployable-proxy evidence. Do not claim a causal determining variable.",
            "",
            f"Scatter rows: {len(scatter_rows)}.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase 38 evidence matrix using frozen deployable proxies only.")
    parser.add_argument("--proxy_summary", default=str(REPO_ROOT / "runs" / "evidence_matrix_20260617" / "deployable_proxy" / "deployable_proxy_summary.csv"))
    parser.add_argument("--target_summary", default=str(REPO_ROOT / "runs" / "evidence_matrix_20260617" / "applicability_grid_summary.csv"))
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "runs" / "evidence_matrix_20260617"))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    scatter_rows = build_scatter_rows(_read_rows(Path(args.proxy_summary)), _read_rows(Path(args.target_summary)))
    corr_rows = build_correlation_rows(scatter_rows)
    _write_csv(output_dir / "frozen_proxy_scatter.csv", scatter_rows)
    _write_csv(output_dir / "frozen_proxy_correlations.csv", corr_rows)
    plot_dir = output_dir / "frozen_proxy_scatter_plots"
    for row in corr_rows:
        group = [
            srow
            for srow in scatter_rows
            if srow["proxy_config"] == row["proxy_config"] and srow["target"] == row["target"] and srow["proxy"] == row["proxy"]
        ]
        safe_name = "__".join([str(row["proxy_config"]), str(row["target"]), str(row["proxy"])]).replace("/", "_")
        write_svg_scatter(plot_dir / f"{safe_name}.svg", group, f"{row['proxy_config']} / {row['target']} / {row['proxy']}")
    write_report(output_dir / "evidence_matrix_report.md", corr_rows, scatter_rows)
    print(f"WROTE {output_dir}")


if __name__ == "__main__":
    main()
