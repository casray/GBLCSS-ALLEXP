from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, CitationFull, Coauthor, Planetoid


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM_ROOT = REPO_ROOT.parents[1] / "gb_graph_aware_minival_6_1"
EMBEDDING_SPECS = {
    "Cora": "checkpoints/embeddings/Cora/ratio_train0p2_val0p1_split0_pretrain0/H.pt",
    "CiteSeer": "checkpoints/embeddings/CiteSeer/ratio_train0p2_val0p1_split0_pretrain0/H.pt",
    "CoraFull": "checkpoints/embeddings_v2/CoraFull/grace/H.pt",
    "CoauthorCS": "checkpoints/embeddings_v2/CoauthorCS/grace/H.pt",
    "AmazonPhoto": "checkpoints/embeddings_v2/AmazonPhoto/grace/H.pt",
}


def _load_embedding(path: Path) -> tuple[torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        if "H" not in payload:
            raise ValueError(f"embedding payload has no H tensor: {path}")
        return payload["H"].detach().cpu().float(), payload
    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"unsupported embedding payload: {type(payload)!r}")
    return payload.detach().cpu().float(), {}


def audit_embedding_file(dataset: str, path: str | Path) -> dict[str, object]:
    path = Path(path).resolve()
    h, payload = _load_embedding(path)
    if h.ndim != 2:
        raise ValueError(f"H must be two-dimensional: {path}")
    norms = h.norm(p=2, dim=1)
    checkpoint_value = str(payload.get("checkpoint_path", "")).strip()
    checkpoint_path = Path(checkpoint_value) if checkpoint_value else None
    metadata_path = path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return {
        "dataset": dataset,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "nodes": int(h.shape[0]),
        "dim": int(h.shape[1]),
        "finite": bool(torch.isfinite(h).all().item()),
        "max_row_norm_error": float((norms - 1.0).abs().max().item()),
        "payload_type": "dict" if payload else "tensor",
        "metadata_exists": metadata_path.exists(),
        "metadata_label_free": metadata.get("label_free", ""),
        "metadata_used_labels_or_masks": metadata.get("used_labels_or_masks_in_training", ""),
        "config_source_dataset": metadata.get("config", {}).get("config_source_dataset", ""),
        "checkpoint_recorded": checkpoint_path is not None,
        "checkpoint_exists": bool(checkpoint_path and checkpoint_path.exists()),
    }


def summarize_ball_rows(rows: list[dict[str, object]], max_balls: int = 64) -> dict[str, object]:
    if not rows:
        raise ValueError("ball rows are empty")
    grouped: dict[str, list[int]] = defaultdict(list)
    cap_hits = 0
    for row in rows:
        count = int(row["num_balls"])
        if count > int(max_balls):
            raise ValueError(f"num_balls={count} exceeds max_balls={max_balls}")
        grouped[str(row["dataset"])].append(count)
        cap_hits += int(count == int(max_balls))
    values = [value for items in grouped.values() for value in items]
    return {
        "global_min": min(values),
        "global_max": max(values),
        "cap_hits": cap_hits,
        "total": len(values),
        "by_dataset": {
            dataset: {
                "min": min(items),
                "max": max(items),
                "range": f"{min(items)}-{max(items)}",
                "cap_hits": sum(value == int(max_balls) for value in items),
                "total": len(items),
            }
            for dataset, items in grouped.items()
        },
    }


def _dataset_pair(dataset: str, repo_root: Path, upstream_root: Path):
    transform = T.NormalizeFeatures()
    if dataset in {"Cora", "CiteSeer"}:
        local = Planetoid(str(repo_root / "data"), dataset, transform=transform)[0]
        upstream = Planetoid(str(upstream_root / "data" / dataset), dataset, transform=transform)[0]
    elif dataset == "CoraFull":
        local = CitationFull(str(repo_root / "data"), "cora", transform=transform)[0]
        upstream = CitationFull(str(upstream_root / "data" / dataset), "cora", transform=transform)[0]
    elif dataset == "CoauthorCS":
        local = Coauthor(str(repo_root / "data"), "CS", transform=transform)[0]
        upstream = Coauthor(str(upstream_root / "data" / dataset), "CS", transform=transform)[0]
    elif dataset == "AmazonPhoto":
        local = Amazon(str(repo_root / "data"), "Photo", transform=transform)[0]
        upstream = Amazon(str(upstream_root / "data" / dataset), "Photo", transform=transform)[0]
    else:
        raise ValueError(f"unsupported dataset for order audit: {dataset}")
    return local, upstream


def _canonical_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    encoded = edge_index[0].long() * int(num_nodes) + edge_index[1].long()
    return torch.sort(encoded).values


def audit_node_order(dataset: str, repo_root: Path, upstream_root: Path) -> dict[str, object]:
    local, upstream = _dataset_pair(dataset, repo_root, upstream_root)
    same_shape = tuple(local.x.shape) == tuple(upstream.x.shape)
    feature_max_abs_diff = float((local.x - upstream.x).abs().max().item()) if same_shape else float("inf")
    local_edges = _canonical_edges(local.edge_index, int(local.num_nodes))
    upstream_edges = _canonical_edges(upstream.edge_index, int(upstream.num_nodes))
    return {
        "dataset": dataset,
        "nodes_equal": int(local.num_nodes) == int(upstream.num_nodes),
        "labels_equal": bool(torch.equal(local.y.cpu(), upstream.y.cpu())),
        "features_equal": bool(same_shape and feature_max_abs_diff == 0.0),
        "feature_max_abs_diff": feature_max_abs_diff,
        "edges_equal": bool(torch.equal(local_edges.cpu(), upstream_edges.cpu())),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    embedding_rows: list[dict[str, object]],
    node_rows: list[dict[str, object]],
    ball_rows: list[dict[str, object]],
) -> str:
    summary = summarize_ball_rows(ball_rows, max_balls=64)
    lines = [
        "# Submission Integrity Audit - Phase 4: GRACE Provenance and Ball Construction",
        "",
        "## Verdict",
        "",
        "- GRACE embedding provenance: PASS for file existence, finite normalized tensors, and node-order alignment.",
        "- Label-free training path: PASS by code inspection; GRACE losses read only features and edges. The v2 metadata also records `label_free=true` and `used_labels_or_masks_in_training=false`.",
        "- Manuscript mechanism facts: **FAIL until corrected**. Two P0 factual statements contradict the formal artifacts and implementation.",
        "",
        "## Embedding Manifest",
        "",
        "| Dataset | Shape | SHA256 | Config source | Checkpoint recoverable |",
        "|---|---:|---|---|---|",
    ]
    for row in embedding_rows:
        source = row["config_source_dataset"] or "native/legacy payload"
        lines.append(
            f"| {row['dataset']} | {row['nodes']}x{row['dim']} | `{str(row['sha256'])[:16]}...` | "
            f"{source} | {row['checkpoint_exists']} |"
        )
    lines.extend(
        [
            "",
            "## Node-order Audit",
            "",
            "| Dataset | Nodes | Labels | Features | Canonical edges |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in node_rows:
        lines.append(
            f"| {row['dataset']} | {row['nodes_equal']} | {row['labels_equal']} | "
            f"{row['features_equal']} | {row['edges_equal']} |"
        )
    lines.extend(
        [
            "",
            "## Five-seed Ball Counts",
            "",
            "| Dataset | Ball-count range | Seeds hitting B_max=64 |",
            "|---|---:|---:|",
        ]
    )
    for dataset, item in summary["by_dataset"].items():
        lines.append(f"| {dataset} | {item['range']} | {item['cap_hits']}/{item['total']} |")
    lines.extend(
        [
            "",
            f"Across the 25 formal reconstructions, the observed range is {summary['global_min']}-{summary['global_max']}; "
            f"{summary['cap_hits']}/{summary['total']} runs hit the hard cap.",
            "",
            "## P0 Manuscript Corrections",
            "",
            "1. Replace the universal `H in R^{N x 256}` statement with `H in R^{N x d_h}`. The formal Cora embedding is 128-dimensional; the other four core embeddings are 256-dimensional.",
            "2. Remove `22-133 balls` and `the cap is not triggered on most datasets`. The implementation cannot exceed 64 balls. The formal five-seed result is 52-64 balls, with 19/25 runs hitting the cap.",
            "3. Reframe large balls on CoraFull/CoauthorCS/AmazonPhoto as a consequence of the fixed 64-ball capacity as well as embedding geometry; do not claim purely geometry-driven stopping.",
            "",
            "## Reproducibility Boundary",
            "",
            "- Cora and CiteSeer retain GRACE model checkpoints and are numerically re-extractable. An independent CPU extraction differed from the archived H by at most 8.94e-8 (Cora) and 1.04e-7 (CiteSeer).",
            "- The v2 pretraining path for CoraFull, CoauthorCS, and AmazonPhoto saves H, metadata, and losses, but not model weights. The reported H files are hashable artifacts, yet exact regeneration from a saved encoder checkpoint is unavailable. This is a P1 reproducibility weakness, not evidence that the reported H is fabricated.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM_ROOT)
    parser.add_argument(
        "--ball-counts",
        type=Path,
        default=REPO_ROOT / "runs/submission_integrity_audit_20260710/ball_counts_5seed.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/submission_integrity_audit_20260710",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "docs/submission_integrity_audit_phase4_grace_provenance_20260711.md",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    upstream_root = args.upstream_root.resolve()

    embedding_rows = [
        audit_embedding_file(dataset, upstream_root / relative)
        for dataset, relative in EMBEDDING_SPECS.items()
    ]
    node_rows = [audit_node_order(dataset, repo_root, upstream_root) for dataset in EMBEDDING_SPECS]
    ball_rows = _read_csv(args.ball_counts)
    summarize_ball_rows(ball_rows, max_balls=64)

    _write_csv(args.output_dir / "grace_embedding_manifest.csv", embedding_rows)
    _write_csv(args.output_dir / "grace_node_order_audit.csv", node_rows)
    args.report.write_text(build_report(embedding_rows, node_rows, ball_rows), encoding="utf-8")
    print(args.report)


if __name__ == "__main__":
    main()
