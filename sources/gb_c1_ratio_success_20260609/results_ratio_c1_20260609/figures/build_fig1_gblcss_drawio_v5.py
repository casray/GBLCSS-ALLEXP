"""Build the compact 3+1 submission figure for GB-LCSS.

This version emphasizes the paper-level argument rather than implementation
detail: graph structure is converted into leave-one-out soft supervision,
then an X-only classifier is trained and deployed.
"""

from __future__ import annotations

from pathlib import Path

import build_fig1_gblcss_drawio_v4 as base


OUT = Path(__file__).with_name("fig1_gblcss_overview_drawio_v5.drawio")

INK = "#2F3943"
MUTED = "#6F7C86"
DIVIDER = "#CBD3DA"
NEUTRAL = "#6F8392"
NEUTRAL_LIGHT = "#DDE5EA"
BLUE = "#3B6EA8"
BLUE_LIGHT = "#EDF3F8"
GREEN = "#009E73"
GREEN_LIGHT = "#E8F6F1"
ORANGE = "#D99000"
ORANGE_LIGHT = "#FFF4D8"
PURPLE = "#7656A3"
PURPLE_LIGHT = "#F3EFF8"
RED = "#C94B3C"
WHITE = "#FFFFFF"


def section_header(d: base.Diagram, x: int, y: int, width: int, number: int, title: str, color: str) -> None:
    d.ellipse(x, y, 28, 28, str(number), fill=color, stroke=color, width=0.8, font_size=16, font_color=WHITE)
    d.text(x + 38, y - 2, width - 38, 30, title, font_size=18, bold=True, align="left", color=INK)
    d.line(x, y + 38, x + width, y + 38, color=DIVIDER, width=1.1)


def neutral_graph(d: base.Diagram, cx: float, cy: float, scale: float = 1.0, variant: int = 0) -> None:
    pts = [(-42, -7), (-26, -37), (3, -30), (33, -40), (43, -5), (18, 29), (-12, 36), (-43, 20)]
    edges = [(0, 1), (0, 2), (0, 7), (1, 2), (2, 3), (2, 5), (2, 6), (3, 4), (3, 5), (4, 5), (5, 6), (6, 7)]
    if variant == 1:
        edges = [e for i, e in enumerate(edges) if i not in (1, 8)]
    elif variant == 2:
        edges = [e for i, e in enumerate(edges) if i not in (3, 10)]
    for a, b in edges:
        d.line(cx + pts[a][0] * scale, cy + pts[a][1] * scale, cx + pts[b][0] * scale, cy + pts[b][1] * scale, color="#91A0AB", width=0.9)
    for i, (dx, dy) in enumerate(pts):
        color = NEUTRAL if i % 3 else "#A9B7C1"
        d.dot(cx + dx * scale, cy + dy * scale, color, radius=5.5 * scale, stroke=WHITE, width=0.5)


def compact_ball(
    d: base.Diagram,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    *,
    outline: str,
    points: list[tuple[float, float]],
    point_color: str = NEUTRAL,
    stroke_width: float = 1.5,
) -> None:
    d.ellipse(cx - rx, cy - ry, 2 * rx, 2 * ry, fill=WHITE, stroke=outline, width=stroke_width)
    for dx, dy in points:
        d.dot(cx + dx, cy + dy, point_color, radius=4.2, stroke=WHITE, width=0.4)


def compact_hist(
    d: base.Diagram,
    x: float,
    y: float,
    values: list[float],
    colors: list[str],
    *,
    width: int = 132,
    height: int = 78,
) -> None:
    d.line(x, y, x + width, y, color="#87939D", width=0.9)
    bar_w = 16
    gap = 9
    for i, (value, color) in enumerate(zip(values, colors)):
        h = value * height
        bx = x + 9 + i * (bar_w + gap)
        d.rect(bx, y - h, bar_w, h, fill=color, stroke=INK, width=0.45, rounded=0)
        d.text(bx - 2, y + 2, bar_w + 4, 16, str(i + 1), font_size=10, color=MUTED)


def compact_mlp(d: base.Diagram, x: float, y: float, color: str = PURPLE) -> None:
    columns = [(x, [y - 26, y, y + 26]), (x + 62, [y - 28, y, y + 28]), (x + 124, [y - 20, y + 20])]
    for ci in range(len(columns) - 1):
        left_x, left_y = columns[ci]
        right_x, right_y = columns[ci + 1]
        pairs = [(0, 0), (0, -1), (1, 0), (1, -1), (2, 0), (2, -1)]
        for li, ri in pairs:
            d.line(left_x + 7, left_y[li], right_x - 7, right_y[ri], color="#B9C3D8", width=0.65)
    for xx, ys in columns:
        for yy in ys:
            d.ellipse(xx - 7, yy - 7, 14, 14, fill=WHITE, stroke=color, width=1.1)


def probability_strip(d: base.Diagram, x: float, y: float, values: list[float]) -> None:
    colors = ["#D8CCE6", "#A98FC3", PURPLE, "#C8B7DA", "#E9E1F0"]
    d.rect(x, y, 28, 82, fill=WHITE, stroke=DIVIDER, width=1.0, rounded=1)
    cursor = y + 78
    for value, color in zip(values, colors):
        h = max(4, value * 72)
        cursor -= h
        d.rect(x + 4, cursor, 20, h, fill=color, stroke=WHITE, width=0.3, rounded=0)


def build() -> base.Diagram:
    base.OUT = OUT
    d = base.Diagram()
    d.model.set("pageHeight", "820")
    d.model.set("dy", "820")

    # Section guides: open editorial layout instead of dashboard cards.
    section_header(d, 24, 20, 336, 1, "Label-free representation", BLUE)
    section_header(d, 392, 20, 336, 2, "Granular-ball partition", GREEN)
    section_header(d, 760, 20, 816, 3, "Leave-one-out local consensus", ORANGE)
    d.line(376, 18, 376, 418, color=DIVIDER, width=0.9)
    d.line(744, 18, 744, 418, color=DIVIDER, width=0.9)
    d.line(20, 425, 1580, 425, color=DIVIDER, width=1.1)

    # Stage 1: graph -> augmented views -> shared GRACE -> frozen H.
    d.text(18, 78, 100, 28, "G = (V, E, X)", font_size=13, bold=True)
    neutral_graph(d, 62, 161, 0.62)
    d.ellipse(25, 230, 20, 20, "×", fill=WHITE, stroke=RED, width=1.4, font_size=14, font_color=RED)
    d.text(49, 225, 82, 28, "labels not used", font_size=11, color=RED, bold=True, align="left")
    d.arrow(101, 154, 120, 148, color=INK, width=1.15)
    d.arrow(101, 168, 120, 245, color=INK, width=1.15)

    d.text(108, 78, 104, 28, "augmented views", font_size=12, bold=True)
    d.rect(120, 108, 78, 82, fill=WHITE, stroke=DIVIDER, width=0.9)
    d.rect(120, 204, 78, 82, fill=WHITE, stroke=DIVIDER, width=0.9)
    neutral_graph(d, 159, 149, 0.40, variant=1)
    neutral_graph(d, 159, 245, 0.40, variant=2)
    d.text(111, 294, 96, 34, "edge dropping<br>feature masking", font_size=10, color=MUTED)
    d.arrow(198, 149, 218, 165, color=INK, width=1.15)
    d.arrow(198, 245, 218, 235, color=INK, width=1.15)

    d.rect(218, 132, 62, 136, "<b>Shared</b><br><b>GRACE</b><br><i>g</i><sub>ψ</sub>", fill=BLUE_LIGHT, stroke=BLUE, width=1.5, font_size=13)
    d.arrow(280, 200, 294, 200, color=INK, width=1.25)
    d.text(286, 76, 55, 34, r"\(\bar{H}\)", font_size=18, color=BLUE, bold=True)
    d.rect(294, 110, 38, 184, fill=WHITE, stroke=DIVIDER, width=0.9)
    for x, y in [(304, 130), (319, 145), (306, 163), (319, 184), (305, 214), (319, 233), (306, 255), (319, 274)]:
        d.dot(x, y, NEUTRAL if x < 312 else "#A9B7C1", radius=4.0, stroke=WHITE, width=0.4)
    d.text(28, 355, 330, 38, "The graph shapes a frozen representation;<br>no class label enters this stage.", font_size=12, color=MUTED)

    # Stage 2: compact parent -> split -> 2x3 final balls.
    d.text(398, 78, 92, 24, "parent", font_size=13, bold=True)
    compact_ball(d, 450, 183, 48, 70, outline=GREEN, points=[(-20, -28), (15, -35), (25, -4), (-26, 7), (4, 16), (-12, 42), (26, 38)])
    d.arrow(500, 183, 524, 183, color=INK, width=1.25)
    d.text(508, 127, 70, 42, "recursive<br>2-means", font_size=11, color=MUTED, bold=True)
    compact_ball(d, 570, 139, 39, 41, outline=GREEN, points=[(-14, -10), (8, -18), (17, 9), (-8, 16)])
    compact_ball(d, 570, 236, 39, 41, outline=GREEN, points=[(-17, -9), (7, -18), (18, 8), (-7, 17)])
    d.arrow(612, 183, 634, 183, color=INK, width=1.25)
    d.text(620, 78, 96, 24, "disjoint balls", font_size=13, bold=True)
    final_centers = [(650, 132), (695, 132), (650, 202), (695, 202), (650, 272), (695, 272)]
    for idx, (cx, cy) in enumerate(final_centers):
        selected = idx == 4
        compact_ball(
            d,
            cx,
            cy,
            19,
            24,
            outline=ORANGE if selected else GREEN,
            points=[(-6, -7), (7, -5), (0, 9)],
            stroke_width=2.0 if selected else 1.2,
        )
    d.text(620, 305, 68, 24, "selected Bⱼ", font_size=11, color=ORANGE, bold=True)
    d.text(398, 348, 320, 46, "Geometry only · mutually exclusive regions<br>fixed budget Bmax = 64", font_size=12, color=MUTED)
    d.arrow(650, 296, 804, 131, color=ORANGE, width=1.4, dashed=True, label="Bⱼ", label_size=12)

    # Stage 3: selected ball -> exclude self -> distribution -> consensus -> target.
    d.text(775, 76, 120, 24, "selected Bⱼ", font_size=13, bold=True)
    d.ellipse(780, 108, 122, 122, fill=WHITE, stroke=ORANGE, width=1.7)
    class_points = [
        (806, 134, base.BLUE), (838, 126, base.ORANGE), (869, 146, base.ORANGE),
        (805, 181, base.PURPLE), (839, 190, base.GREEN), (874, 184, base.GRAY),
        (829, 159, base.GREEN), (860, 164, base.ORANGE),
    ]
    for x, y, color in class_points:
        d.dot(x, y, color, radius=5, stroke=WHITE, width=0.45)
    d.dot(829, 159, base.GREEN, radius=7, stroke=RED, width=1.5)
    d.line(822, 152, 836, 166, color=RED, width=1.7)
    d.line(836, 152, 822, 166, color=RED, width=1.7)
    d.text(780, 236, 122, 34, "exclude v from<br>its own vote", font_size=11, color=RED, bold=True)
    d.arrow(907, 169, 937, 169, color=INK, width=1.3)

    d.text(930, 78, 146, 24, "leave-one-out votes", font_size=13, bold=True)
    compact_hist(d, 940, 210, [0.36, 0.82, 0.28, 0.18, 0.0], [base.BLUE, base.ORANGE, base.GREEN, base.PURPLE, base.GRAY])
    d.text(936, 226, 140, 28, "pⱼ,−ᵥ", font_size=15, bold=True)
    d.arrow(1080, 169, 1110, 169, color=INK, width=1.3)

    d.rect(1115, 104, 170, 64, "<b>Unique majority</b><br>semi-hard consensus", fill=GREEN_LIGHT, stroke=GREEN, width=1.2, font_size=12)
    d.rect(1115, 184, 170, 64, "<b>Tie</b><br>retain the distribution", fill=WHITE, stroke=DIVIDER, width=1.0, font_size=12)
    d.text(1130, 262, 140, 26, "tie-safe qⱼ,−ᵥ", font_size=13, color=GREEN, bold=True)
    d.arrow(1288, 169, 1320, 169, color=INK, width=1.3)

    d.text(1312, 78, 230, 24, "soft correction", font_size=13, bold=True)
    compact_hist(d, 1320, 210, [0.0, 0.72, 0.0, 0.0, 0.0], [base.BLUE, base.GREEN, base.ORANGE, base.PURPLE, base.GRAY], width=120, height=72)
    d.text(1323, 226, 116, 24, "noisy one-hot", font_size=11, color=MUTED)
    d.text(1440, 146, 30, 32, "+", font_size=22, bold=True)
    compact_hist(d, 1470, 210, [0.25, 0.76, 0.22, 0.12, 0.0], [base.BLUE, base.ORANGE, base.GREEN, base.PURPLE, base.GRAY], width=100, height=72)
    d.text(1466, 226, 108, 24, "local consensus", font_size=11, color=MUTED)
    d.rect(
        1305,
        278,
        270,
        66,
        r"\(t_v=(1-\alpha_{j,-v})e_{\tilde y_v}+\alpha_{j,-v}q_{j,-v}\)",
        fill=ORANGE_LIGHT,
        stroke=ORANGE,
        width=1.5,
        font_size=16,
        extra="fontFamily=Times New Roman;",
    )
    d.text(1312, 349, 255, 42, "αⱼ,−ᵥ controls how strongly local<br>consensus replaces the noisy label.", font_size=11, color=MUTED)

    # Cross-stage target flow.
    d.line(1440, 344, 1440, 493, color=ORANGE, width=1.4, dashed=True)
    d.line(1440, 493, 630, 493, color=ORANGE, width=1.4, dashed=True)
    d.text(1450, 398, 120, 24, "precomputed tᵥ", font_size=11, color=ORANGE, bold=True, align="left")

    # Stage 4: one training lane and one deployment lane.
    section_header(d, 24, 447, 1552, 4, "X-only classifier training and inference", PURPLE)
    d.text(42, 505, 320, 26, "Training with precomputed soft targets", font_size=14, bold=True, align="left", color=PURPLE)
    d.text(55, 587, 65, 28, "Xᴸ", font_size=17, bold=True)
    d.rect(110, 585, 96, 26, fill=PURPLE_LIGHT, stroke=PURPLE, width=1.0, rounded=0)
    for i in range(5):
        d.rect(115 + i * 17, 590, 13, 16, fill=PURPLE if i in (0, 3) else WHITE, stroke=DIVIDER, width=0.4, rounded=0)
    d.arrow(210, 598, 255, 598, color=INK, width=1.35)
    compact_mlp(d, 275, 598)
    d.text(278, 528, 120, 26, "2-layer MLP", font_size=13, bold=True)
    d.arrow(405, 598, 445, 598, color=INK, width=1.35)
    probability_strip(d, 455, 557, [0.15, 0.50, 0.17, 0.12, 0.06])
    d.text(425, 650, 94, 24, "predictions", font_size=11, color=MUTED)
    d.arrow(488, 598, 535, 598, color=INK, width=1.35)
    d.rect(545, 552, 170, 92, "<b>Soft cross-entropy</b><br>prediction vs. tᵥ", fill=ORANGE_LIGHT, stroke=ORANGE, width=1.4, font_size=13)
    d.arrow(630, 493, 630, 552, color=ORANGE, width=1.3, dashed=True)

    d.line(785, 510, 785, 742, color=DIVIDER, width=1.0, dashed=True)
    d.text(726, 548, 120, 44, "After<br>training", font_size=13, color=MUTED, bold=True)
    d.arrow(747, 621, 822, 621, color=MUTED, width=1.2)

    d.text(840, 505, 300, 26, "Inference on the same graph", font_size=14, bold=True, align="left", color=PURPLE)
    d.text(850, 587, 70, 28, "xₜₑₛₜ", font_size=16, bold=True)
    d.rect(918, 585, 96, 26, fill=PURPLE_LIGHT, stroke=PURPLE, width=1.0, rounded=0)
    for i in range(5):
        d.rect(923 + i * 17, 590, 13, 16, fill=PURPLE if i in (1, 4) else WHITE, stroke=DIVIDER, width=0.4, rounded=0)
    d.arrow(1018, 598, 1060, 598, color=INK, width=1.35)
    compact_mlp(d, 1080, 598)
    d.text(1080, 528, 126, 26, "trained MLP", font_size=13, bold=True)
    d.arrow(1210, 598, 1250, 598, color=INK, width=1.35)
    probability_strip(d, 1260, 557, [0.08, 0.63, 0.11, 0.10, 0.08])
    d.text(1233, 650, 90, 24, "class probabilities", font_size=11, color=MUTED)
    d.arrow(1290, 598, 1340, 598, color=INK, width=1.35)
    d.rect(1350, 548, 205, 100, "<b>X-only inference</b><br>no A · no H̄ · no neighbors", fill=PURPLE_LIGHT, stroke=PURPLE, width=1.3, font_size=14)

    d.text(
        42,
        716,
        1510,
        48,
        "Training-time graph structure is converted into node-specific soft supervision; the deployed classifier receives node attributes only.",
        font_size=13,
        color=INK,
        bold=True,
    )

    return d


if __name__ == "__main__":
    diagram = build()
    diagram.save()
    print(OUT)
