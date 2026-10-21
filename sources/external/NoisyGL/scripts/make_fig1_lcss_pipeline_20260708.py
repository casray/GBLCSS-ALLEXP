from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.8,
    }
)


COL = {
    "ink": "#1f2937",
    "muted": "#64748b",
    "panel": "#f8fafc",
    "line": "#cbd5e1",
    "graph": "#475569",
    "blue": "#4C78A8",
    "teal": "#3BA99C",
    "orange": "#F28E2B",
    "green": "#59A14F",
    "red": "#E15759",
    "purple": "#8E6BBE",
    "yellow": "#F1CE63",
}


def box(ax, xy, wh, title, body=None, fc="#ffffff", ec=None, lw=0.9, title_color=None):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.014,rounding_size=0.022",
        facecolor=fc,
        edgecolor=ec or COL["line"],
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 0.032,
        title,
        ha="center",
        va="top",
        fontsize=7.6,
        fontweight="bold",
        color=title_color or COL["ink"],
    )
    if body:
        ax.text(
            x + w / 2,
            y + h / 2 - 0.018,
            body,
            ha="center",
            va="center",
            fontsize=6.2,
            color=COL["muted"],
            linespacing=1.28,
        )
    return patch


def arrow(ax, start, end, color=None, lw=1.1, rad=0.0, text=None, text_offset=(0, 0)):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=lw,
        color=color or COL["muted"],
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arr)
    if text:
        mx = (start[0] + end[0]) / 2 + text_offset[0]
        my = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, ha="center", va="center", fontsize=5.8, color=COL["muted"])


def draw_graph(ax, cx, cy, scale=1.0):
    pts = [
        (-0.035, 0.025),
        (0.025, 0.048),
        (0.078, 0.012),
        (-0.005, -0.025),
        (0.055, -0.055),
        (-0.07, -0.035),
    ]
    edges = [(0, 1), (1, 2), (1, 3), (3, 4), (0, 3), (3, 5), (2, 4)]
    pts = [(cx + x * scale, cy + y * scale) for x, y in pts]
    for i, j in edges:
        ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]], color=COL["graph"], lw=0.75, zorder=1)
    colors = [COL["blue"], COL["teal"], COL["orange"], COL["green"], COL["red"], COL["purple"]]
    for k, (x, y) in enumerate(pts):
        ax.add_patch(Circle((x, y), 0.011 * scale, fc=colors[k], ec="white", lw=0.6, zorder=2))
    for x, y in [pts[2], pts[5]]:
        ax.text(x + 0.011 * scale, y + 0.009 * scale, "×", fontsize=8, fontweight="bold", color=COL["red"], zorder=3)


def draw_embedding(ax, x0, y0, w, h):
    ax.text(x0 + w / 2, y0 + h - 0.022, "Frozen GRACE-H space", ha="center", va="top", fontsize=7.4, fontweight="bold", color=COL["ink"])
    centers = [
        (x0 + 0.28 * w, y0 + 0.47 * h, COL["blue"]),
        (x0 + 0.47 * w, y0 + 0.58 * h, COL["orange"]),
        (x0 + 0.66 * w, y0 + 0.43 * h, COL["green"]),
        (x0 + 0.52 * w, y0 + 0.31 * h, COL["purple"]),
    ]
    for cx, cy, c in centers:
        ax.add_patch(Circle((cx, cy), 0.092, fc="none", ec="#8a8f98", lw=0.8, alpha=0.78))
        ax.add_patch(Circle((cx, cy), 0.006, fc="#111827", ec="none"))
        offsets = [(-0.038, 0.018), (-0.018, -0.026), (0.025, -0.012), (0.036, 0.026), (-0.002, 0.036)]
        for dx, dy in offsets:
            ax.add_patch(Circle((cx + dx, cy + dy), 0.008, fc=c, ec="white", lw=0.35, alpha=0.9))
    ax.text(x0 + 0.04 * w, y0 + 0.105 * h, "label-free granular-ball grouping", fontsize=5.5, color=COL["muted"], ha="left")


def draw_votes(ax, x0, y0, w, h):
    ax.text(x0 + w / 2, y0 + h - 0.025, "Noisy train votes", ha="center", va="top", fontsize=7.4, fontweight="bold", color=COL["ink"])
    rows = [(COL["orange"], 0.085, "7"), (COL["green"], 0.045, "3"), (COL["blue"], 0.030, "2")]
    for idx, (c, bw, count) in enumerate(rows):
        yy = y0 + h - 0.080 - idx * 0.043
        ax.add_patch(Rectangle((x0 + 0.034, yy - 0.012), bw, 0.021, fc=c, ec="white", lw=0.35))
        ax.text(x0 + 0.034 + bw + 0.008, yy - 0.001, count, ha="left", va="center", fontsize=5.8, color=COL["muted"])


def draw_soft_target(ax, x0, y0, w, h):
    ax.text(x0 + w / 2, y0 + h - 0.025, "Soft target", ha="center", va="top", fontsize=7.4, fontweight="bold", color=COL["ink"])
    bx = x0 + 0.07
    by = y0 + 0.075
    bw = w - 0.10
    bh = 0.035
    bx = x0 + 0.05
    parts = [(0.55, COL["orange"], "q"), (0.25, COL["green"], ""), (0.12, COL["blue"], ""), (0.08, COL["purple"], "")]
    cur = bx
    for frac, c, label in parts:
        ax.add_patch(Rectangle((cur, by), bw * frac, bh, fc=c, ec="white", lw=0.4))
        cur += bw * frac
    ax.text(x0 + w / 2, y0 + 0.041, "semi-hard, beta=0.5", ha="center", fontsize=5.5, color=COL["muted"])


def draw_mlp(ax, x0, y0, w, h, title="X-only MLP"):
    ax.text(
        x0 + w / 2,
        y0 + h - 0.012,
        title,
        ha="center",
        va="top",
        fontsize=6.5,
        fontweight="bold",
        color=COL["ink"],
    )
    block_w = 0.25 * w
    block_h = 0.30 * h
    block_y = y0 + 0.33 * h
    layers = [
        (x0 + 0.02 * w, "Input\nX", "#eff6ff", COL["blue"]),
        (x0 + 0.375 * w, "Hidden\n64", "#eff6ff", COL["blue"]),
        (x0 + 0.73 * w, "Output\nC", "#f0fdf4", COL["green"]),
    ]
    for left, label, face, edge in layers:
        ax.add_patch(
            FancyBboxPatch(
                (left, block_y),
                block_w,
                block_h,
                boxstyle="round,pad=0.003,rounding_size=0.005",
                facecolor=face,
                edgecolor=edge,
                linewidth=0.8,
            )
        )
        ax.text(
            left + block_w / 2,
            block_y + block_h / 2,
            label,
            ha="center",
            va="center",
            fontsize=4.4,
            color=COL["ink"],
            linespacing=1.0,
        )
    for left, next_left in zip([item[0] for item in layers[:-1]], [item[0] for item in layers[1:]]):
        ax.add_patch(
            FancyArrowPatch(
                (left + block_w, block_y + block_h / 2),
                (next_left, block_y + block_h / 2),
                arrowstyle="-|>",
                mutation_scale=6,
                linewidth=0.8,
                color=COL["muted"],
            )
        )
    ax.text(x0 + w / 2, y0 + 0.10 * h, "soft cross-entropy", ha="center", fontsize=4.8, color=COL["muted"])


def make_figure():
    fig = plt.figure(figsize=(7.2, 4.15), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.965, "GB-LCSS turns graph structure into denoised supervision for graph-free inference",
            ha="left", va="top", fontsize=10.5, fontweight="bold", color=COL["ink"])
    ax.text(0.02, 0.925, "Training uses the graph only to build label-free contrastive geometry; inference keeps only an X-only MLP.",
            ha="left", va="top", fontsize=6.8, color=COL["muted"])

    # Stage ribbons
    ax.add_patch(Rectangle((0.02, 0.84), 0.79, 0.026, fc="#e0f2fe", ec="none"))
    ax.text(0.03, 0.853, "Training-time resources", ha="left", va="center", fontsize=6.5, fontweight="bold", color="#075985")
    ax.add_patch(Rectangle((0.825, 0.84), 0.155, 0.026, fc="#dcfce7", ec="none"))
    ax.text(0.836, 0.853, "Inference-time resource", ha="left", va="center", fontsize=6.5, fontweight="bold", color="#166534")

    # Panels
    box(ax, (0.025, 0.58), (0.145, 0.22), "Observed graph", "X, A + noisy\ntrain labels", fc="#ffffff")
    draw_graph(ax, 0.095, 0.675, scale=1.25)

    box(ax, (0.205, 0.58), (0.15, 0.22), "GRACE pretrain", None, fc="#ffffff")
    ax.text(0.28, 0.682, "H", ha="center", va="center", fontsize=16, fontweight="bold", color=COL["blue"])
    ax.text(0.28, 0.625, "label-free InfoNCE\nfrozen after pretraining", ha="center", fontsize=5.5, color=COL["muted"], linespacing=1.3)

    box(ax, (0.39, 0.535), (0.20, 0.28), "", fc="#ffffff")
    draw_embedding(ax, 0.402, 0.548, 0.176, 0.245)

    box(ax, (0.625, 0.61), (0.16, 0.205), "", fc="#ffffff")
    draw_votes(ax, 0.634, 0.625, 0.142, 0.17)

    box(ax, (0.625, 0.35), (0.16, 0.205), "", fc="#ffffff")
    draw_soft_target(ax, 0.634, 0.365, 0.142, 0.17)

    box(ax, (0.825, 0.58), (0.145, 0.22), "Deployable model", None, fc="#ffffff")
    draw_mlp(ax, 0.838, 0.608, 0.118, 0.145)

    # Inference strip
    box(ax, (0.035, 0.16), (0.21, 0.13), "Raw node feature", "x_v only", fc="#f0fdf4", ec="#86efac", title_color="#166534")
    box(ax, (0.335, 0.16), (0.23, 0.13), "X-only MLP", "same student as above", fc="#f0fdf4", ec="#86efac", title_color="#166534")
    box(ax, (0.655, 0.16), (0.245, 0.13), "Prediction", "no graph, no H,\nno grouping", fc="#f0fdf4", ec="#86efac", title_color="#166534")

    ax.text(0.02, 0.31, "At inference, all graph-derived objects are discarded.", ha="left", fontsize=6.5, color=COL["muted"])

    # Main arrows
    arrow(ax, (0.172, 0.69), (0.204, 0.69))
    arrow(ax, (0.356, 0.69), (0.389, 0.69))
    arrow(ax, (0.591, 0.68), (0.624, 0.71), text="train labels only", text_offset=(-0.012, 0.032))
    arrow(ax, (0.70, 0.61), (0.70, 0.557), color=COL["orange"])
    arrow(ax, (0.785, 0.455), (0.825, 0.655), color=COL["orange"], rad=-0.12, text="soft CE", text_offset=(0.025, 0.018))
    arrow(ax, (0.245, 0.225), (0.335, 0.225), color="#16a34a")
    arrow(ax, (0.565, 0.225), (0.655, 0.225), color="#16a34a")

    # Negative-resource callouts
    ax.text(0.82, 0.49, "Discarded after training:", ha="left", fontsize=6.2, color=COL["muted"], fontweight="bold")
    for idx, item in enumerate(["adjacency A", "GRACE encoder", "granular balls"]):
        ax.text(0.835, 0.465 - idx * 0.026, f"× {item}", ha="left", fontsize=6.0, color=COL["red"])

    # Panel labels
    labels = [
        (0.025, 0.815, "a"),
        (0.39, 0.815, "b"),
        (0.625, 0.815, "c"),
        (0.825, 0.815, "d"),
        (0.035, 0.296, "e"),
    ]
    for x, y, lab in labels:
        ax.text(x, y, lab, fontsize=8.5, fontweight="bold", color=COL["ink"], ha="left", va="top")

    caption = (
        "Fig. 1 | Granular-ball local-consensus soft supervision (GB-LCSS). "
        "The graph is used only during training to learn label-free GRACE representations and "
        "construct local consensus soft targets from noisy training labels. "
        "The deployed classifier is an ordinary MLP that takes raw node features only."
    )
    ax.text(0.02, 0.055, caption, ha="left", va="bottom", fontsize=6.25, color=COL["ink"], wrap=True)
    return fig


if __name__ == "__main__":
    fig = make_figure()
    stem = OUT_DIR / "fig1_lcss_pipeline_20260716"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(stem.with_suffix(".svg"))
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".png"))
