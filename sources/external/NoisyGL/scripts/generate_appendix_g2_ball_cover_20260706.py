from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle
from sklearn.decomposition import PCA


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ssl_swap_downstream import _load_dataset
from utils.frozen_grace_targets import load_frozen_embeddings
from utils.labelnoise import label_process
from utils.ordinary_gb import build_ordinary_granular_balls
from utils.tools import setup_seed


OUT_STEM = REPO_ROOT / "paper_assets_20260706" / "figures" / "appendix_G2_grace_ball_cover"
EMB_ROOT = REPO_ROOT.parent.parent / "gb_graph_aware_minival_6_1" / "checkpoints"

DATASETS = [
    ("cora", "Cora"),
    ("citeseer", "CiteSeer"),
    ("pubmed", "PubMed"),
    ("coauthorcs", "Coauthor-CS"),
    ("amazonpho", "Amazon-Photo"),
]

H_CANDIDATES = {
    "cora": [
        EMB_ROOT / "embeddings" / "Cora" / "ratio_train0p2_val0p1_split0_pretrain0" / "H.pt",
        EMB_ROOT / "embeddings_v2" / "Cora" / "grace" / "H.pt",
    ],
    "citeseer": [
        EMB_ROOT / "embeddings" / "CiteSeer" / "ratio_train0p2_val0p1_split0_pretrain0" / "H.pt",
        EMB_ROOT / "embeddings_v2" / "CiteSeer" / "grace" / "H.pt",
    ],
    "pubmed": [EMB_ROOT / "embeddings_v2" / "PubMed" / "grace" / "H.pt"],
    "coauthorcs": [EMB_ROOT / "embeddings_v2" / "CoauthorCS" / "grace" / "H.pt"],
    "amazonpho": [EMB_ROOT / "embeddings_v2" / "AmazonPhoto" / "grace" / "H.pt"],
}

DISPLAY_LIMIT = 3500
LOCAL_DISPLAY_LIMIT = 900


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 600,
        }
    )


def color_for_class(cls: int) -> str:
    return mpl.colors.to_hex(plt.get_cmap("tab20")(int(cls) % 20))


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def train_mask_to_bool(mask, n_nodes: int) -> np.ndarray:
    mask_np = to_numpy(mask)
    if mask_np.dtype == np.bool_:
        if mask_np.shape[0] != n_nodes:
            raise ValueError("Boolean train mask length does not match node count.")
        return mask_np.copy()
    out = np.zeros(n_nodes, dtype=bool)
    out[mask_np.astype(np.int64)] = True
    return out


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = "\n".join(str(path) for path in paths)
    raise FileNotFoundError(f"No embedding file found. Tried:\n{joined}")


def ball_cover_stats(z2: np.ndarray, assignment: np.ndarray) -> list[dict[str, object]]:
    stats: list[dict[str, object]] = []
    for ball_id in np.unique(assignment):
        idx = np.flatnonzero(assignment == ball_id)
        points = z2[idx]
        center = points.mean(axis=0)
        dist = np.linalg.norm(points - center, axis=1)
        radius = float(dist.max())
        visual_radius = float(np.quantile(dist, 0.80))
        stats.append(
            {
                "ball_id": int(ball_id),
                "idx": idx,
                "center": center,
                "radius": radius,
                "visual_radius": visual_radius,
                "size": int(idx.size),
            }
        )
    return stats


def select_zoom_ball(stats: list[dict[str, object]], train_mask: np.ndarray, noisy: np.ndarray, n_classes: int) -> int:
    best_id = int(stats[0]["ball_id"])
    best_score = -1.0
    for stat in stats:
        idx = stat["idx"]
        train_idx = idx[train_mask[idx]]
        if train_idx.size < 8:
            continue
        counts = np.bincount(noisy[train_idx], minlength=n_classes)
        margin = counts.max() / max(1, counts.sum())
        # Prefer enough votes and a non-trivial local distribution, so the zoom
        # panel explains voting rather than showing a single-class toy ball.
        score = train_idx.size - 25.0 * abs(margin - 0.65)
        if score > best_score:
            best_score = float(score)
            best_id = int(stat["ball_id"])
    return best_id


def sample_indices(indices: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if indices.size <= limit:
        return indices
    return np.sort(rng.choice(indices, size=limit, replace=False))


def draw_ball_covers(ax, stats: list[dict[str, object]], *, selected_ball: int | None = None) -> None:
    for stat in sorted(stats, key=lambda item: float(item["visual_radius"]), reverse=True):
        ball_id = int(stat["ball_id"])
        is_selected = selected_ball is not None and ball_id == selected_ball
        circle = Circle(
            tuple(stat["center"]),
            float(stat["visual_radius"]) * 1.02,
            facecolor="none",
            edgecolor="#111111",
            alpha=0.88 if is_selected else 0.52,
            linewidth=1.15 if is_selected else 0.55,
        )
        ax.add_patch(circle)
    centers = np.vstack([stat["center"] for stat in stats])
    ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=11, color="#111111", linewidths=0.65, alpha=0.9)


def load_panel_data(dataset_key: str) -> dict[str, object]:
    data = _load_dataset(
        dataset_key,
        device="cpu",
        split_type="percent",
        train_percent=0.2,
        val_percent=0.1,
        test_percent=0.7,
    )
    h_path = first_existing(H_CANDIDATES[dataset_key])
    h = load_frozen_embeddings(h_path, expected_nodes=int(data.n_nodes))
    h_np = l2_normalize(to_numpy(h).astype(np.float32))
    clean = to_numpy(data.labels).astype(np.int64)
    train_mask = train_mask_to_bool(data.train_masks, int(data.n_nodes))
    noisy_tensor, _ = label_process(
        labels=data.labels,
        features=data.feats,
        n_classes=data.n_classes,
        noise_type="uniform",
        noise_rate=0.4,
        random_seed=0,
        eligible_mask=data.train_masks,
        debug=False,
    )
    noisy = to_numpy(noisy_tensor).astype(np.int64)

    gb = build_ordinary_granular_balls(
        h,
        max_balls=64,
        min_ball_size=32,
        split_improvement=0.05,
        random_state=0,
    )
    assignment = np.asarray(gb.assignment, dtype=np.int64)
    z2 = PCA(n_components=2, random_state=0, svd_solver="randomized").fit_transform(h_np)
    stats = ball_cover_stats(z2, assignment)
    zoom_ball = select_zoom_ball(stats, train_mask, noisy, int(data.n_classes))

    return {
        "data": data,
        "h_path": h_path,
        "z2": z2,
        "stats": stats,
        "clean": clean,
        "noisy": noisy,
        "train_mask": train_mask,
        "zoom_ball": zoom_ball,
    }


def draw_full_panel(ax, panel: dict[str, object], dataset_label: str, row_seed: int) -> None:
    data = panel["data"]
    z2 = panel["z2"]
    stats = panel["stats"]
    clean = panel["clean"]
    rng = np.random.default_rng(20260706 + row_seed)
    show_idx = sample_indices(np.arange(int(data.n_nodes)), DISPLAY_LIMIT, rng)

    for cls in range(int(data.n_classes)):
        cls_idx = show_idx[clean[show_idx] == cls]
        if cls_idx.size:
            ax.scatter(
                z2[cls_idx, 0],
                z2[cls_idx, 1],
                s=2.7,
                color=color_for_class(cls),
                alpha=0.48,
                linewidths=0,
            )
    draw_ball_covers(ax, stats)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{dataset_label}: robust projected contours", loc="left", fontsize=7.2, fontweight="bold")
    sampled_note = "" if show_idx.size == int(data.n_nodes) else f"; {show_idx.size:,} points shown"
    ax.text(
        0.02,
        0.02,
        f"N={int(data.n_nodes):,}; balls={len(stats)}{sampled_note}",
        transform=ax.transAxes,
        fontsize=5.8,
        color="#4b5563",
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#d9dee7", "boxstyle": "round,pad=0.22", "alpha": 0.86},
    )
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_zoom_panel(ax, panel: dict[str, object], dataset_label: str, row_seed: int) -> None:
    data = panel["data"]
    z2 = panel["z2"]
    stats = panel["stats"]
    noisy = panel["noisy"]
    train_mask = panel["train_mask"]
    zoom_ball = int(panel["zoom_ball"])
    zoom_stat = next(stat for stat in stats if int(stat["ball_id"]) == zoom_ball)
    zoom_center = np.asarray(zoom_stat["center"])
    zoom_radius = float(zoom_stat["visual_radius"])
    x0, x1 = zoom_center[0] - 1.55 * zoom_radius, zoom_center[0] + 1.55 * zoom_radius
    y0, y1 = zoom_center[1] - 1.55 * zoom_radius, zoom_center[1] + 1.55 * zoom_radius

    local_ball_stats = [
        stat
        for stat in stats
        if x0 <= float(stat["center"][0]) <= x1 and y0 <= float(stat["center"][1]) <= y1
    ]
    local_mask = (z2[:, 0] >= x0) & (z2[:, 0] <= x1) & (z2[:, 1] >= y0) & (z2[:, 1] <= y1)
    local_idx = np.flatnonzero(local_mask)
    rng = np.random.default_rng(20260761 + row_seed)
    local_show_idx = sample_indices(local_idx, LOCAL_DISPLAY_LIMIT, rng)
    ax.scatter(z2[local_show_idx, 0], z2[local_show_idx, 1], c="#aeb6c2", s=5.2, alpha=0.35, linewidths=0)
    draw_ball_covers(ax, local_ball_stats, selected_ball=zoom_ball)

    zoom_idx = np.asarray(zoom_stat["idx"])
    train_idx = zoom_idx[train_mask[zoom_idx]]
    non_train_idx = zoom_idx[~train_mask[zoom_idx]]
    non_train_show_idx = sample_indices(non_train_idx, LOCAL_DISPLAY_LIMIT, rng)
    ax.scatter(z2[non_train_show_idx, 0], z2[non_train_show_idx, 1], c="#7d8793", s=7, alpha=0.32, linewidths=0)
    for cls in range(int(data.n_classes)):
        cls_idx = train_idx[noisy[train_idx] == cls]
        if cls_idx.size:
            ax.scatter(
                z2[cls_idx, 0],
                z2[cls_idx, 1],
                s=16,
                color=color_for_class(cls),
                edgecolors="white",
                linewidths=0.30,
                alpha=0.92,
            )

    counts = np.bincount(noisy[train_idx], minlength=int(data.n_classes))
    top = counts.argsort()[::-1][:3]
    vote_text = ", ".join(f"c{int(cls)}:{int(counts[cls])}" for cls in top if counts[cls] > 0)
    ax.text(
        0.02,
        0.02,
        f"zoomed ball\ntrain noisy votes: {vote_text}",
        transform=ax.transAxes,
        fontsize=5.8,
        color="#4b5563",
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#d9dee7", "boxstyle": "round,pad=0.22", "alpha": 0.88},
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{dataset_label}: zoomed voting signal", loc="left", fontsize=7.2, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    setup_style()
    setup_seed(0)
    panels = [(key, label, load_panel_data(key)) for key, label in DATASETS]

    fig, axes = plt.subplots(
        len(panels),
        2,
        figsize=(7.1, 10.8),
        gridspec_kw={"width_ratios": [1.08, 1.0], "hspace": 0.28, "wspace": 0.12},
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.962, bottom=0.052)
    for row, (dataset_key, dataset_label, panel) in enumerate(panels):
        draw_full_panel(axes[row, 0], panel, dataset_label, row)
        draw_zoom_panel(axes[row, 1], panel, dataset_label, row)

    fig.suptitle(
        "Granular balls as robust local contours in frozen GRACE representations",
        x=0.035,
        y=0.996,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    fig.text(
        0.035,
        0.004,
        "Balls are built label-free from L2-normalized GRACE H. Black contours show the 80% projected distance radius of each final ball "
        "for visualization only; large-dataset scatter points are sampled for display, while balls use all nodes.",
        ha="left",
        va="bottom",
        fontsize=6.1,
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
