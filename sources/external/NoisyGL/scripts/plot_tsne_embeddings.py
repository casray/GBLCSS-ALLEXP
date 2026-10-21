from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.linear_probe_embeddings import _load_dataset, embedding_path_for
from utils.frozen_grace_targets import load_frozen_embeddings


def _parse_str_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def class_distance_summary(embeddings: torch.Tensor, labels: torch.Tensor) -> dict:
    z = F.normalize(embeddings.detach().float().cpu(), p=2, dim=1)
    labels = labels.detach().cpu().long()
    classes = sorted(torch.unique(labels).tolist())
    within_values = []
    between_values = []
    for cls in classes:
        idx = torch.nonzero(labels == int(cls), as_tuple=False).flatten()
        if idx.numel() >= 2:
            within_values.append(torch.pdist(z[idx], p=2).mean())
    for i, cls_a in enumerate(classes):
        idx_a = torch.nonzero(labels == int(cls_a), as_tuple=False).flatten()
        for cls_b in classes[i + 1 :]:
            idx_b = torch.nonzero(labels == int(cls_b), as_tuple=False).flatten()
            if idx_a.numel() and idx_b.numel():
                between_values.append(torch.cdist(z[idx_a], z[idx_b], p=2).mean())
    within = torch.stack(within_values).mean().item() if within_values else 0.0
    between = torch.stack(between_values).mean().item() if between_values else 0.0
    return {
        "mean_within_class_distance": float(within),
        "mean_between_class_distance": float(between),
        "between_within_ratio": float(between / max(within, 1e-12)),
    }


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_tsne(embeddings: torch.Tensor, perplexity: float, seed: int):
    z = F.normalize(embeddings.detach().float().cpu(), p=2, dim=1).numpy()
    return TSNE(
        n_components=2,
        perplexity=float(perplexity),
        init="pca",
        learning_rate="auto",
        random_state=int(seed),
        max_iter=1000,
    ).fit_transform(z)


def plot_tsne(panels: list[dict], labels, output_prefix: str | Path) -> None:
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.3), squeeze=False)
    labels_np = labels.detach().cpu().numpy()
    for ax, panel in zip(axes[0], panels):
        coords = panel["coords"]
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=labels_np,
            s=6,
            cmap="tab10",
            alpha=0.85,
            linewidths=0,
        )
        ax.set_title(panel["method"])
        ax.set_xticks([])
        ax.set_yticks([])
    handles, _ = scatter.legend_elements(num=None)
    axes[0][-1].legend(handles=handles, labels=[str(i) for i in sorted(set(labels_np.tolist()))], frameon=False, markerscale=2)
    fig.tight_layout()
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=240)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot t-SNE comparison for SSL embeddings.")
    parser.add_argument("--embedding_root", required=True)
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--methods", default="grace,ccassg")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split_type", default="percent")
    parser.add_argument("--train_percent", type=float, default=0.2)
    parser.add_argument("--val_percent", type=float, default=0.1)
    parser.add_argument("--test_percent", type=float, default=0.7)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="runs/figures_20260615")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _load_dataset(args.dataset, args.device, args.split_type, args.train_percent, args.val_percent, args.test_percent)
    labels = data.labels.detach().cpu() if isinstance(data.labels, torch.Tensor) else torch.as_tensor(data.labels)
    panels = []
    summary_rows = []
    for method in _parse_str_list(args.methods):
        path = embedding_path_for(args.embedding_root, args.dataset, method)
        if not path.exists():
            raise FileNotFoundError(f"missing embedding for {args.dataset}/{method}: {path}")
        embeddings = load_frozen_embeddings(path, expected_nodes=int(labels.shape[0]))
        coords = compute_tsne(embeddings, args.perplexity, args.seed)
        panels.append({"method": method, "coords": coords})
        summary_rows.append(
            {
                "dataset": str(args.dataset).lower(),
                "method": method,
                "embedding_path": str(path),
                "embedding_dim": int(embeddings.shape[1]),
                **class_distance_summary(embeddings, labels),
            }
        )
    write_csv(output_dir / "tsne_class_distance_summary.csv", summary_rows)
    plot_tsne(panels, labels, output_dir / f"tsne_{str(args.dataset).lower()}_grace_vs_ccassg")
    print(f"WROTE {output_dir / f'tsne_{str(args.dataset).lower()}_grace_vs_ccassg.png'}")


if __name__ == "__main__":
    main()
