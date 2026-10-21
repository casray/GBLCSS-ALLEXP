from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Ellipse
from sklearn.decomposition import PCA


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ssl_swap_downstream import _load_dataset
from utils.frozen_grace_targets import load_frozen_embeddings
from utils.ordinary_gb import build_ordinary_granular_balls
from utils.tools import setup_seed


OUT_STEM = REPO_ROOT / "paper_assets_20260706" / "figures" / "preview_ball_cover_styles_cora"
H_PATH = (
    REPO_ROOT.parent.parent
    / "gb_graph_aware_minival_6_1"
    / "checkpoints"
    / "embeddings"
    / "Cora"
    / "ratio_train0p2_val0p1_split0_pretrain0"
    / "H.pt"
)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 600,
        }
    )


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def color_for_class(cls: int) -> str:
    return mpl.colors.to_hex(plt.get_cmap("tab10")(int(cls) % 10))


def ball_stats(z2: np.ndarray, assignment: np.ndarray) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for ball_id in np.unique(assignment):
        idx = np.flatnonzero(assignment == ball_id)
        pts = z2[idx]
        center = pts.mean(axis=0)
        dist = np.linalg.norm(pts - center, axis=1)
        cov = np.cov((pts - center).T) if pts.shape[0] > 2 else np.eye(2) * 1e-4
        out.append(
            {
                "ball_id": int(ball_id),
                "idx": idx,
                "center": center,
                "max_radius": float(dist.max()),
                "q80_radius": float(np.quantile(dist, 0.80)),
                "cov": cov,
                "size": int(idx.size),
            }
        )
    return out


def draw_points(ax, z2: np.ndarray, labels: np.ndarray, n_classes: int) -> None:
    for cls in range(n_classes):
        mask = labels == cls
        ax.scatter(z2[mask, 0], z2[mask, 1], s=5, color=color_for_class(cls), alpha=0.50, linewidths=0)


def draw_centers(ax, stats: list[dict[str, object]]) -> None:
    centers = np.vstack([stat["center"] for stat in stats])
    ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=20, color="#111111", linewidths=0.85, alpha=0.95)


def draw_max_circles(ax, stats: list[dict[str, object]]) -> None:
    for stat in sorted(stats, key=lambda item: float(item["max_radius"]), reverse=True):
        ax.add_patch(
            Circle(
                tuple(stat["center"]),
                float(stat["max_radius"]),
                facecolor="none",
                edgecolor="#111111",
                linewidth=0.68,
                alpha=0.58,
            )
        )
    draw_centers(ax, stats)


def draw_quantile_circles(ax, stats: list[dict[str, object]]) -> None:
    for stat in sorted(stats, key=lambda item: float(item["q80_radius"]), reverse=True):
        ax.add_patch(
            Circle(
                tuple(stat["center"]),
                float(stat["q80_radius"]),
                facecolor="none",
                edgecolor="#111111",
                linewidth=0.76,
                alpha=0.72,
            )
        )
    draw_centers(ax, stats)


def draw_cov_ellipses(ax, stats: list[dict[str, object]]) -> None:
    for stat in stats:
        cov = np.asarray(stat["cov"], dtype=float)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = np.maximum(vals[order], 1e-8)
        vecs = vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        # About a local 80-90% visual contour in the 2D projection. This is
        # only a drawing convention; the algorithm still uses the original GB.
        width, height = 2.8 * np.sqrt(vals)
        ax.add_patch(
            Ellipse(
                xy=tuple(stat["center"]),
                width=float(width),
                height=float(height),
                angle=float(angle),
                facecolor="none",
                edgecolor="#111111",
                linewidth=0.76,
                alpha=0.72,
            )
        )
    draw_centers(ax, stats)


def draw_representative_quantile(ax, stats: list[dict[str, object]], keep: int = 18) -> None:
    selected = sorted(stats, key=lambda item: int(item["size"]), reverse=True)[:keep]
    draw_quantile_circles(ax, selected)


def format_panel(ax, title: str, subtitle: str) -> None:
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
    ax.text(
        0.02,
        0.02,
        subtitle,
        transform=ax.transAxes,
        fontsize=6.1,
        color="#4b5563",
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#d9dee7", "boxstyle": "round,pad=0.22", "alpha": 0.86},
    )
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    setup_style()
    setup_seed(0)
    data = _load_dataset(
        "cora",
        device="cpu",
        split_type="percent",
        train_percent=0.2,
        val_percent=0.1,
        test_percent=0.7,
    )
    h = load_frozen_embeddings(H_PATH, expected_nodes=int(data.n_nodes))
    h_np = l2_normalize(to_numpy(h).astype(np.float32))
    labels = to_numpy(data.labels).astype(np.int64)
    gb = build_ordinary_granular_balls(
        h,
        max_balls=64,
        min_ball_size=32,
        split_improvement=0.05,
        random_state=0,
    )
    assignment = np.asarray(gb.assignment, dtype=np.int64)
    z2 = PCA(n_components=2, random_state=0).fit_transform(h_np)
    stats = ball_stats(z2, assignment)

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.2), gridspec_kw={"hspace": 0.16, "wspace": 0.08})
    panels = [
        ("A  Current: max-radius circles", f"all {len(stats)} final balls; sensitive to projected outliers", draw_max_circles),
        ("B  Robust: 80% quantile circles", f"all {len(stats)} final balls; outlier-resistant visual contour", draw_quantile_circles),
        ("C  Shape-aware: covariance ellipses", f"all {len(stats)} final balls; follows local projected anisotropy", draw_cov_ellipses),
        ("D  Sparse: representative balls", "18 largest final balls only; explanation figure, not full cover", draw_representative_quantile),
    ]
    for ax, (title, subtitle, draw_fn) in zip(axes.flat, panels):
        draw_points(ax, z2, labels, int(data.n_classes))
        draw_fn(ax, stats)
        format_panel(ax, title, subtitle)

    fig.suptitle(
        "Preview alternatives for visualizing final granular balls in frozen GRACE-H space",
        x=0.035,
        y=0.992,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    fig.text(
        0.035,
        0.006,
        "All panels use the same final Cora balls from the current algorithm; only the drawing convention changes.",
        ha="left",
        va="bottom",
        fontsize=6.3,
        color="#4b5563",
    )
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT_STEM}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_STEM}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_STEM}.pdf")
    print(f"Wrote {OUT_STEM}.png")


if __name__ == "__main__":
    main()
