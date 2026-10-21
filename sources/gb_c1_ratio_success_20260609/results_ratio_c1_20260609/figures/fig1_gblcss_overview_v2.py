"""Submission figure for the GB-LCSS method overview.

Figure contract
---------------
Core conclusion:
    Training-time graph structure is converted into leave-one-out
    local-consensus supervision, enabling graph-free X-only inference.

Evidence chain:
    (1) (X, A) enters label-free GRACE while noisy labels are withheld.
    (2) Frozen row-normalized H is recursively partitioned into disjoint balls.
    (3) Leave-one-out, tie-safe votes form a node-specific soft target.
    (4) The target supervises an X-only MLP; deployment keeps only x_v and MLP.

Archetype: schematic-led composite with Stage 3 as the visual center.
Backend: Python / Matplotlib only.
Target size: 183 mm x 100 mm; vector SVG/PDF and 600-dpi raster exports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


WIDTH_MM = 183
HEIGHT_MM = 102
MM_PER_INCH = 25.4


def palette(grayscale: bool = False) -> dict[str, str]:
    if grayscale:
        return {
            "blue": "#4A4A4A",
            "blue_light": "#F1F1F1",
            "orange": "#777777",
            "orange_light": "#E6E6E6",
            "green": "#2F2F2F",
            "green_light": "#F6F6F6",
            "purple": "#A0A0A0",
            "red": "#111111",
            "dark": "#303030",
            "mid": "#737373",
            "line": "#555555",
            "pale": "#FAFAFA",
            "white": "#FFFFFF",
        }
    return {
        "blue": "#3B6EA8",
        "blue_light": "#EAF2F8",
        "orange": "#E69F00",
        "orange_light": "#FFF2D8",
        "green": "#009E73",
        "green_light": "#E6F5EF",
        "purple": "#8E6BBE",
        "red": "#D55E00",
        "dark": "#3F4650",
        "mid": "#87919C",
        "line": "#66717D",
        "pale": "#F7F9FB",
        "white": "#FFFFFF",
    }


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.8,
            "lines.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def rounded_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 0.8,
    radius: float = 0.008,
    linestyle: str = "-",
    zorder: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    *,
    linewidth: float = 0.9,
    mutation_scale: float = 7,
    linestyle: str = "-",
    connectionstyle: str = "arc3,rad=0",
    zorder: int = 5,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def stage_header(ax, x0: float, x1: float, number: int, title: str, color: str, p: dict[str, str]) -> None:
    y = 0.872
    ax.add_patch(Circle((x0 + 0.017, y), 0.014, facecolor=color, edgecolor=color, lw=0.8, zorder=8))
    ax.text(x0 + 0.017, y - 0.001, str(number), ha="center", va="center", color="white", fontsize=8.0, fontweight="bold", zorder=9)
    ax.text(
        x0 + 0.038,
        y,
        title,
        ha="left",
        va="center",
        color=p["dark"],
        fontsize=9.0,
        fontweight="bold",
        linespacing=0.93,
        zorder=9,
    )
    ax.plot([x0 + 0.006, x1 - 0.006], [0.813, 0.813], color=color, lw=1.0, solid_capstyle="round", zorder=8)


def draw_class_marker(ax, x: float, y: float, cls: int, p: dict[str, str], *, size: float = 22, alpha: float = 1.0, zorder: int = 6) -> None:
    markers = ["o", "s", "^"]
    colors = [p["blue"], p["orange"], p["purple"]]
    ax.scatter(
        [x],
        [y],
        s=size,
        marker=markers[cls % 3],
        c=[colors[cls % 3]],
        edgecolors=p["white"],
        linewidths=0.45,
        alpha=alpha,
        zorder=zorder,
    )


def draw_graph(ax, cx: float, cy: float, scale: float, p: dict[str, str], *, faded: bool = False) -> None:
    coords = np.array(
        [
            [-0.74, 0.16],
            [-0.42, 0.68],
            [0.02, 0.44],
            [0.52, 0.67],
            [0.76, 0.08],
            [0.31, -0.36],
            [-0.26, -0.58],
            [-0.69, -0.28],
        ]
    )
    edges = [(0, 1), (0, 2), (0, 7), (1, 2), (2, 3), (2, 5), (2, 6), (3, 4), (3, 5), (4, 5), (5, 6), (6, 7)]
    alpha = 0.35 if faded else 1.0
    for i, j in edges:
        ax.plot(
            [cx + scale * coords[i, 0], cx + scale * coords[j, 0]],
            [cy + scale * coords[i, 1], cy + scale * coords[j, 1]],
            color=p["mid"],
            lw=0.65,
            alpha=alpha,
            zorder=2,
        )
    for idx, (dx, dy) in enumerate(coords):
        face = p["blue_light"] if idx % 3 else p["white"]
        ax.add_patch(
            Circle(
                (cx + scale * dx, cy + scale * dy),
                0.105 * scale,
                facecolor=face,
                edgecolor=p["blue"],
                lw=0.65,
                alpha=alpha,
                zorder=3,
            )
        )


def draw_stage1(ax, bounds: tuple[float, float, float, float], p: dict[str, str]) -> None:
    x0, y0, x1, y1 = bounds
    rounded_box(ax, (x0, y0), x1 - x0, y1 - y0, facecolor=p["blue_light"], edgecolor=p["blue"])
    stage_header(ax, x0, x1, 1, "Label-free graph\nrepresentation", p["blue"], p)

    graph_x, graph_y = x0 + 0.036, 0.620
    draw_graph(ax, graph_x, graph_y, 0.031, p)
    ax.text(graph_x, 0.525, r"$G=(V,E,X)$", ha="center", va="top", color=p["dark"], fontsize=7.0)

    view_x = x0 + 0.090
    rounded_box(ax, (view_x - 0.019, 0.650), 0.038, 0.095, facecolor=p["white"], edgecolor=p["mid"], linewidth=0.65, radius=0.004)
    rounded_box(ax, (view_x - 0.019, 0.505), 0.038, 0.095, facecolor=p["white"], edgecolor=p["mid"], linewidth=0.65, radius=0.004)
    draw_graph(ax, view_x, 0.696, 0.016, p, faded=False)
    draw_graph(ax, view_x, 0.551, 0.016, p, faded=False)
    ax.text(view_x, 0.760, "edge drop", ha="center", va="bottom", color=p["mid"], fontsize=7.0)
    ax.text(view_x, 0.487, "feature mask", ha="center", va="top", color=p["mid"], fontsize=7.0)
    arrow(ax, (graph_x + 0.032, graph_y + 0.010), (view_x - 0.021, 0.690), p["blue"])
    arrow(ax, (graph_x + 0.032, graph_y - 0.010), (view_x - 0.021, 0.555), p["blue"])

    grace_x = x0 + 0.137
    rounded_box(ax, (grace_x - 0.023, 0.557), 0.046, 0.125, facecolor=p["white"], edgecolor=p["blue"], linewidth=0.9, radius=0.006)
    ax.text(grace_x, 0.628, "GRACE", ha="center", va="center", color=p["blue"], fontsize=7.6, fontweight="bold")
    ax.text(grace_x, 0.590, "encoder", ha="center", va="center", color=p["dark"], fontsize=7.0)
    arrow(ax, (view_x + 0.020, 0.696), (grace_x - 0.025, 0.646), p["blue"])
    arrow(ax, (view_x + 0.020, 0.551), (grace_x - 0.025, 0.594), p["blue"])

    h_x = x0 + 0.172
    arrow(ax, (grace_x + 0.024, 0.620), (h_x - 0.017, 0.620), p["blue"])
    ax.add_patch(Rectangle((h_x - 0.011, 0.565), 0.022, 0.110, facecolor=p["white"], edgecolor=p["blue"], lw=0.8, zorder=3))
    for yy in np.linspace(0.575, 0.663, 6):
        ax.plot([h_x - 0.007, h_x + 0.007], [yy, yy], color=p["blue"], lw=0.8)
    ax.text(h_x - 0.004, 0.704, r"frozen $\bar{H}$", ha="center", va="bottom", color=p["dark"], fontsize=7.0)
    ax.text(h_x - 0.003, 0.545, r"$\ell_2$-normalized", ha="center", va="top", color=p["mid"], fontsize=7.0)

    # Noisy labels are shown on an explicitly separate lane and do not feed GRACE.
    ax.plot([x0 + 0.018, x1 - 0.018], [0.365, 0.365], color=p["blue"], lw=0.65, linestyle=(0, (2, 2)))
    ax.text((x0 + x1) / 2, 0.407, "Noisy training labels", ha="center", va="center", color=p["red"], fontsize=7.0, fontweight="bold")
    ax.text((x0 + x1) / 2, 0.382, "withheld from GRACE", ha="center", va="center", color=p["red"], fontsize=7.0)
    label_xs = np.linspace(x0 + 0.045, x1 - 0.045, 4)
    for idx, xx in enumerate(label_xs):
        draw_class_marker(ax, xx, 0.292, idx % 3, p, size=25)
        if idx in (1, 3):
            ax.plot([xx - 0.006, xx + 0.006], [0.282, 0.302], color=p["red"], lw=1.2, zorder=8)
            ax.plot([xx - 0.006, xx + 0.006], [0.302, 0.282], color=p["red"], lw=1.2, zorder=8)
    ax.text((x0 + x1) / 2, 0.236, "used only after partitioning", ha="center", va="center", color=p["dark"], fontsize=7.0)
    rounded_box(ax, (x0 + 0.042, 0.115), x1 - x0 - 0.084, 0.055, facecolor=p["white"], edgecolor=p["blue"], linewidth=0.7, radius=0.004)
    ax.text((x0 + x1) / 2, 0.142, "labels are not used", ha="center", va="center", color=p["blue"], fontsize=7.0, fontweight="bold")


def draw_stage2(ax, bounds: tuple[float, float, float, float], p: dict[str, str]) -> None:
    x0, y0, x1, y1 = bounds
    rounded_box(ax, (x0, y0), x1 - x0, y1 - y0, facecolor=p["blue_light"], edgecolor=p["blue"])
    stage_header(ax, x0, x1, 2, "Granular-ball\npartition", p["blue"], p)

    rng = np.random.default_rng(14)
    parent_cx, parent_cy = x0 + 0.041, 0.625
    parent_pts = rng.normal(size=(18, 2))
    parent_pts[:, 0] = parent_cx + 0.017 * parent_pts[:, 0]
    parent_pts[:, 1] = parent_cy + 0.060 * parent_pts[:, 1]
    ax.add_patch(Ellipse((parent_cx, parent_cy), 0.070, 0.220, facecolor=p["white"], edgecolor=p["blue"], lw=0.8, zorder=2))
    ax.scatter(parent_pts[:, 0], parent_pts[:, 1], s=8, c=p["mid"], edgecolors="none", zorder=3)
    ax.text(parent_cx, 0.758, "parent ball", ha="center", va="bottom", color=p["dark"], fontsize=7.0)

    split_x = x0 + 0.092
    rounded_box(ax, (split_x - 0.021, 0.570), 0.042, 0.105, facecolor=p["white"], edgecolor=p["blue"], linewidth=0.8, radius=0.005)
    ax.text(split_x, 0.634, "2-means", ha="center", va="center", color=p["blue"], fontsize=7.0, fontweight="bold")
    ax.text(split_x, 0.598, "recurse", ha="center", va="center", color=p["dark"], fontsize=7.0)
    arrow(ax, (parent_cx + 0.038, parent_cy), (split_x - 0.028, parent_cy), p["blue"])

    # Final partition is drawn as non-overlapping regions, not overlapping circles.
    px0, py0, pw, ph = x0 + 0.139, 0.505, 0.040, 0.245
    rounded_box(ax, (px0, py0), pw, ph, facecolor=p["white"], edgecolor=p["blue"], linewidth=0.8, radius=0.004)
    regions = [
        ([(px0, py0 + ph), (px0 + 0.030, py0 + ph), (px0 + 0.026, py0 + 0.115), (px0, py0 + 0.095)], p["blue_light"], p["blue"]),
        ([(px0 + 0.030, py0 + ph), (px0 + pw, py0 + ph), (px0 + pw, py0 + 0.105), (px0 + 0.026, py0 + 0.115)], p["orange_light"], p["orange"]),
        ([(px0, py0 + 0.095), (px0 + 0.026, py0 + 0.115), (px0 + pw, py0 + 0.105), (px0 + pw, py0), (px0, py0)], p["green_light"], p["green"]),
    ]
    for verts, face, edge in regions:
        ax.add_patch(Polygon(verts, closed=True, facecolor=face, edgecolor=edge, lw=0.65, zorder=2))
    points = [
        (px0 + 0.008, py0 + 0.198, 0), (px0 + 0.016, py0 + 0.170, 0), (px0 + 0.011, py0 + 0.132, 0),
        (px0 + 0.030, py0 + 0.205, 1), (px0 + 0.034, py0 + 0.165, 1), (px0 + 0.028, py0 + 0.135, 1),
        (px0 + 0.009, py0 + 0.060, 2), (px0 + 0.021, py0 + 0.072, 2), (px0 + 0.032, py0 + 0.048, 2),
    ]
    for xx, yy, cls in points:
        draw_class_marker(ax, xx, yy, cls, p, size=12)
    arrow(ax, (split_x + 0.027, parent_cy), (px0 - 0.004, parent_cy), p["blue"])
    ax.text(px0 + pw / 2, 0.775, "disjoint balls", ha="center", va="bottom", color=p["dark"], fontsize=7.0)

    ax.text((x0 + x1) / 2, 0.440, "Accept a split only if", ha="center", va="center", color=p["dark"], fontsize=7.1, fontweight="bold")
    ax.text((x0 + x1) / 2, 0.397, r"dispersion improves by $\delta$", ha="center", va="center", color=p["dark"], fontsize=7.0)
    ax.text((x0 + x1) / 2, 0.356, r"each child has size $\geq n_{\min}$", ha="center", va="center", color=p["dark"], fontsize=7.0)
    ax.text((x0 + x1) / 2, 0.315, r"number of balls $\leq B_{\max}$", ha="center", va="center", color=p["dark"], fontsize=7.0)
    rounded_box(ax, (x0 + 0.028, 0.195), x1 - x0 - 0.056, 0.070, facecolor=p["white"], edgecolor=p["blue"], linewidth=0.7, radius=0.004)
    ax.text((x0 + x1) / 2, 0.230, "geometry only: no labels", ha="center", va="center", color=p["blue"], fontsize=7.0, fontweight="bold")


def draw_vote_bars(ax, x: float, y: float, width: float, height: float, p: dict[str, str]) -> None:
    values = [0.67, 0.22, 0.11]
    colors = [p["blue"], p["orange"], p["purple"]]
    for idx, (value, color) in enumerate(zip(values, colors)):
        yy = y + height - (idx + 1) * height / 3 + 0.006
        ax.add_patch(Rectangle((x, yy), width, height / 3 - 0.012, facecolor=p["white"], edgecolor=p["mid"], lw=0.45, zorder=2))
        ax.add_patch(Rectangle((x, yy), width * value, height / 3 - 0.012, facecolor=color, edgecolor="none", zorder=3))
        draw_class_marker(ax, x - 0.012, yy + (height / 3 - 0.012) / 2, idx, p, size=13)


def draw_probability_strip(ax, x: float, y: float, width: float, height: float, weights: list[float], p: dict[str, str]) -> None:
    colors = [p["blue"], p["orange"], p["purple"]]
    cursor = x
    for weight, color in zip(weights, colors):
        w = width * weight
        ax.add_patch(Rectangle((cursor, y), w, height, facecolor=color, edgecolor=p["white"], lw=0.35, zorder=4))
        cursor += w
    ax.add_patch(Rectangle((x, y), width, height, facecolor="none", edgecolor=p["dark"], lw=0.6, zorder=5))


def draw_stage3(ax, bounds: tuple[float, float, float, float], p: dict[str, str]) -> None:
    x0, y0, x1, y1 = bounds
    rounded_box(ax, (x0, y0), x1 - x0, y1 - y0, facecolor=p["orange_light"], edgecolor=p["orange"], linewidth=1.0)
    stage_header(ax, x0, x1, 3, "LOO, tie-safe\nlocal consensus", p["orange"], p)

    ball_cx, ball_cy = x0 + 0.061, 0.640
    ax.add_patch(Ellipse((ball_cx, ball_cy), 0.105, 0.250, facecolor=p["white"], edgecolor=p["orange"], lw=1.0, zorder=2))
    nodes = [
        (-0.030, 0.072, 0), (-0.005, 0.088, 0), (0.023, 0.055, 0), (-0.027, 0.018, 0),
        (0.010, 0.020, 0), (0.032, -0.025, 0), (-0.022, -0.052, 1), (0.013, -0.076, 2),
    ]
    for dx, dy, cls in nodes:
        draw_class_marker(ax, ball_cx + dx, ball_cy + dy, cls, p, size=24)

    target_x, target_y = ball_cx - 0.022, ball_cy - 0.052
    ax.add_patch(Circle((target_x, target_y), 0.0125, facecolor="none", edgecolor=p["red"], lw=1.1, zorder=8))
    ax.plot([target_x - 0.008, target_x + 0.008], [target_y - 0.012, target_y + 0.012], color=p["red"], lw=1.4, zorder=9)
    ax.plot([target_x - 0.008, target_x + 0.008], [target_y + 0.012, target_y - 0.012], color=p["red"], lw=1.4, zorder=9)
    ax.text(target_x - 0.006, target_y - 0.043, r"target $v$", ha="center", va="top", color=p["red"], fontsize=7.0, fontweight="bold")
    ax.text(ball_cx, 0.782, "zoomed ball", ha="center", va="bottom", color=p["dark"], fontsize=7.0)

    vote_x = x0 + 0.137
    arrow(ax, (ball_cx + 0.054, ball_cy), (vote_x - 0.018, ball_cy), p["orange"])
    ax.text(x0 + 0.116, 0.706, "exclude self", ha="center", va="bottom", color=p["red"], fontsize=7.0, fontweight="bold")
    ax.plot([target_x + 0.006, x0 + 0.117], [target_y, 0.704], color=p["red"], lw=0.7, linestyle=(0, (2, 2)), zorder=7)
    draw_vote_bars(ax, vote_x, 0.575, 0.050, 0.145, p)
    ax.text(vote_x + 0.025, 0.742, r"votes $\rightarrow\ \mathbf{p}_{j,-v}$", ha="center", va="bottom", color=p["dark"], fontsize=7.0)

    rule_x = x0 + 0.196
    rule_w = 0.098
    rounded_box(ax, (rule_x - 0.007, 0.626), rule_w, 0.125, facecolor=p["white"], edgecolor=p["orange"], linewidth=0.75, radius=0.004)
    rule_cx = rule_x - 0.007 + rule_w / 2
    ax.text(rule_cx, 0.727, "unique majority", ha="center", va="center", color=p["dark"], fontsize=7.0, fontweight="bold")
    ax.text(rule_cx, 0.696, r"$\mathbf{q}_{j,-v}=$", ha="center", va="center", color=p["dark"], fontsize=7.0)
    ax.text(rule_cx, 0.669, r"$(1-\beta)\mathbf{p}_{j,-v}$", ha="center", va="center", color=p["dark"], fontsize=7.0)
    ax.text(rule_cx, 0.642, r"$+\,\beta\mathbf{e}_{m}$", ha="center", va="center", color=p["dark"], fontsize=7.0)
    rounded_box(ax, (rule_x - 0.007, 0.486), rule_w, 0.105, facecolor=p["white"], edgecolor=p["orange"], linewidth=0.75, radius=0.004)
    ax.text(rule_cx, 0.566, "tie", ha="center", va="center", color=p["dark"], fontsize=7.0, fontweight="bold")
    ax.text(rule_cx, 0.532, r"$\mathbf{q}_{j,-v}=\mathbf{p}_{j,-v}$", ha="center", va="center", color=p["dark"], fontsize=7.0)
    ax.text(rule_cx, 0.500, "tie-safe", ha="center", va="center", color=p["orange"], fontsize=7.0, fontweight="bold")
    arrow(ax, (vote_x + 0.053, ball_cy), (rule_x - 0.010, ball_cy), p["orange"])

    rounded_box(ax, (x0 + 0.023, 0.388), x1 - x0 - 0.046, 0.066, facecolor=p["white"], edgecolor=p["orange"], linewidth=0.75, radius=0.004)
    ax.text((x0 + x1) / 2, 0.421, r"$\alpha_{j,-v}=\max\!\left(0,\frac{r_{j,-v}-1/C}{1-1/C}\right)$", ha="center", va="center", color=p["dark"], fontsize=7.0)

    rounded_box(ax, (x0 + 0.023, 0.281), x1 - x0 - 0.046, 0.078, facecolor=p["white"], edgecolor=p["orange"], linewidth=0.75, radius=0.004)
    ax.text((x0 + x1) / 2, 0.320, r"$\mathbf{t}_v=(1-\alpha_{j,-v})\mathbf{e}_{\tilde{y}_v}+\alpha_{j,-v}\mathbf{q}_{j,-v}$", ha="center", va="center", color=p["dark"], fontsize=7.0)

    ax.text(x0 + 0.052, 0.241, "noisy one-hot", ha="center", va="bottom", color=p["dark"], fontsize=7.0)
    draw_probability_strip(ax, x0 + 0.018, 0.191, 0.073, 0.032, [0.0, 1.0, 0.0], p)
    arrow(ax, (x0 + 0.098, 0.207), (x0 + 0.128, 0.207), p["orange"])
    ax.text(x0 + 0.113, 0.169, "soft correction", ha="center", va="top", color=p["orange"], fontsize=7.0, fontweight="bold")
    draw_probability_strip(ax, x0 + 0.134, 0.191, 0.073, 0.032, [0.63, 0.27, 0.10], p)
    ax.text(x0 + 0.1705, 0.241, r"target $\mathbf{t}_v$", ha="center", va="bottom", color=p["dark"], fontsize=7.0)
    ax.text((x0 + x1) / 2, 0.129, "node-specific soft supervision", ha="center", va="center", color=p["orange"], fontsize=7.1, fontweight="bold")


def draw_feature_vector(ax, x: float, y: float, p: dict[str, str], label: str) -> None:
    ax.add_patch(Rectangle((x - 0.012, y - 0.046), 0.024, 0.092, facecolor=p["white"], edgecolor=p["green"], lw=0.75, zorder=3))
    for yy, length in zip(np.linspace(y - 0.032, y + 0.032, 5), [0.013, 0.008, 0.016, 0.010, 0.014]):
        ax.plot([x - 0.008, x - 0.008 + length], [yy, yy], color=p["green"], lw=1.0, zorder=4)
    ax.text(x, y + 0.064, label, ha="center", va="bottom", color=p["dark"], fontsize=7.0, fontweight="bold")


def draw_mlp(ax, x: float, y: float, width: float, height: float, p: dict[str, str], label: str) -> None:
    rounded_box(ax, (x, y), width, height, facecolor=p["white"], edgecolor=p["green"], linewidth=0.9, radius=0.005)
    ax.plot([x + width * 0.37, x + width * 0.37], [y + 0.012, y + height - 0.012], color=p["green"], lw=0.65)
    ax.plot([x + width * 0.68, x + width * 0.68], [y + 0.012, y + height - 0.012], color=p["green"], lw=0.65)
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", color=p["dark"], fontsize=7.0, fontweight="bold")


def draw_stage4(ax, bounds: tuple[float, float, float, float], p: dict[str, str]) -> None:
    x0, y0, x1, y1 = bounds
    rounded_box(ax, (x0, y0), x1 - x0, y1 - y0, facecolor=p["pale"], edgecolor=p["green"], linewidth=1.0)
    stage_header(ax, x0, x1, 4, "X-only student\nand deployment", p["green"], p)

    # Training lane.
    rounded_box(ax, (x0 + 0.012, 0.457), x1 - x0 - 0.024, 0.346, facecolor=p["blue_light"], edgecolor=p["blue"], linewidth=0.65, radius=0.005)
    ax.text(x0 + 0.022, 0.774, "TRAINING", ha="left", va="center", color=p["blue"], fontsize=7.0, fontweight="bold")
    feature_x, train_y = x0 + 0.038, 0.645
    draw_feature_vector(ax, feature_x, train_y, p, r"raw $X$")
    mlp_x, mlp_y, mlp_w, mlp_h = x0 + 0.080, 0.592, 0.068, 0.106
    arrow(ax, (feature_x + 0.014, train_y), (mlp_x - 0.004, train_y), p["green"])
    draw_mlp(ax, mlp_x, mlp_y, mlp_w, mlp_h, p, "2-layer MLP")

    pred_x = x0 + 0.181
    arrow(ax, (mlp_x + mlp_w + 0.004, train_y), (pred_x - 0.018, train_y), p["green"])
    draw_probability_strip(ax, pred_x - 0.015, train_y - 0.017, 0.044, 0.034, [0.55, 0.30, 0.15], p)
    ax.text(pred_x + 0.007, train_y + 0.047, r"$f_{\theta}(x_v)$", ha="center", va="bottom", color=p["dark"], fontsize=7.0)

    loss_x, loss_y = x0 + 0.146, 0.500
    rounded_box(ax, (loss_x, loss_y), 0.083, 0.060, facecolor=p["white"], edgecolor=p["orange"], linewidth=0.85, radius=0.004)
    ax.text(loss_x + 0.0415, loss_y + 0.030, "soft\ncross-entropy", ha="center", va="center", color=p["dark"], fontsize=7.0, fontweight="bold", linespacing=0.92)
    arrow(ax, (pred_x + 0.007, train_y - 0.024), (loss_x + 0.060, loss_y + 0.063), p["green"])
    ax.text(x0 + 0.024, 0.514, r"$\mathbf{t}_v$", ha="center", va="center", color=p["orange"], fontsize=7.4, fontweight="bold")
    arrow(ax, (x0 + 0.039, 0.514), (loss_x - 0.005, 0.528), p["orange"])
    ax.text(x0 + 0.071, 0.483, "target enters loss only", ha="center", va="top", color=p["orange"], fontsize=7.0)

    # Explicit training/deployment boundary.
    ax.plot([x0 + 0.012, x1 - 0.012], [0.410, 0.410], color=p["dark"], lw=0.8, linestyle=(0, (3, 2)), zorder=5)
    ax.text((x0 + x1) / 2, 0.426, "training ends: discard A, H, GRACE, balls", ha="center", va="bottom", color=p["dark"], fontsize=7.0)
    ax.text(x0 + 0.028, 0.410, "//", ha="center", va="center", rotation=90, color=p["red"], fontsize=9.0, fontweight="bold", zorder=7)

    # Deployment lane.
    rounded_box(ax, (x0 + 0.012, 0.108), x1 - x0 - 0.024, 0.270, facecolor=p["green_light"], edgecolor=p["green"], linewidth=0.65, radius=0.005)
    ax.text(x0 + 0.022, 0.350, "INFERENCE", ha="left", va="center", color=p["green"], fontsize=7.0, fontweight="bold")
    infer_y = 0.245
    draw_feature_vector(ax, x0 + 0.040, infer_y, p, r"$x_v$")
    infer_mlp_x = x0 + 0.083
    arrow(ax, (x0 + 0.054, infer_y), (infer_mlp_x - 0.004, infer_y), p["green"])
    draw_mlp(ax, infer_mlp_x, infer_y - 0.052, 0.067, 0.104, p, "same MLP")
    arrow(ax, (infer_mlp_x + 0.071, infer_y), (x0 + 0.174, infer_y), p["green"])
    draw_probability_strip(ax, x0 + 0.178, infer_y - 0.017, 0.045, 0.034, [0.66, 0.20, 0.14], p)
    ax.text(x0 + 0.200, infer_y + 0.052, "class\nprobabilities", ha="center", va="bottom", color=p["dark"], fontsize=7.0, linespacing=0.95)
    ax.text((x0 + x1) / 2, 0.151, r"$X$ only", ha="center", va="center", color=p["green"], fontsize=7.1, fontweight="bold")
    ax.text((x0 + x1) / 2, 0.126, r"no $A$, no $H$, no neighbor fetching", ha="center", va="center", color=p["green"], fontsize=7.0)


def make_figure(grayscale: bool = False):
    setup_style()
    p = palette(grayscale)
    fig = plt.figure(figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(p["white"])
    ax.set_facecolor(p["white"])

    stage_y0, stage_y1 = 0.075, 0.920
    s1 = (0.012, stage_y0, 0.205, stage_y1)
    s2 = (0.215, stage_y0, 0.408, stage_y1)
    s3 = (0.418, stage_y0, 0.719, stage_y1)
    s4 = (0.729, stage_y0, 0.988, stage_y1)

    draw_stage1(ax, s1, p)
    draw_stage2(ax, s2, p)
    draw_stage3(ax, s3, p)
    draw_stage4(ax, s4, p)

    # Main-stage arrows stay above the content and never intersect secondary flows.
    arrow(ax, (s1[2] + 0.002, 0.793), (s2[0] - 0.003, 0.793), p["blue"], linewidth=1.0, mutation_scale=8)
    arrow(ax, (s2[2] + 0.002, 0.793), (s3[0] - 0.003, 0.793), p["blue"], linewidth=1.0, mutation_scale=8)
    arrow(ax, (s3[2] + 0.002, 0.793), (s4[0] - 0.003, 0.793), p["orange"], linewidth=1.0, mutation_scale=8)

    return fig


def export_all(output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig = make_figure(grayscale=False)
    fig.savefig(output_stem.with_suffix(".svg"), format="svg", dpi=600, facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), format="pdf", dpi=600, facecolor="white")
    fig.savefig(output_stem.with_suffix(".png"), format="png", dpi=600, facecolor="white")
    plt.close(fig)

    gray_fig = make_figure(grayscale=True)
    gray_path = output_stem.parent / f"{output_stem.name}_grayscale.png"
    gray_fig.savefig(gray_path, format="png", dpi=600, facecolor="white")
    plt.close(gray_fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the GB-LCSS Fig. 1 overview.")
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(__file__).with_name("fig1_gblcss_overview_v2"),
        help="Output path without extension.",
    )
    args = parser.parse_args()
    export_all(args.output_stem)


if __name__ == "__main__":
    main()
