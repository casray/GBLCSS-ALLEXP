"""Native Draw.io reconstruction of the supplied GB-LCSS reference figure.

This version intentionally follows the reference composition and wording
without redesigning or simplifying it.
"""

from __future__ import annotations

from pathlib import Path

import build_fig1_gblcss_drawio_v4 as base


base.OUT = Path(__file__).with_name("fig1_gblcss_overview_v5_exact_replica.drawio")

INK = "#111111"
MID = "#60666D"
LIGHT = "#E9ECEF"
BLUE = "#1F77E5"
ORANGE = "#FF7F0E"
GREEN = "#2CA02C"
PURPLE = "#8C46C7"
GRAY = "#B8B8B8"
RED = "#FF2020"
VIOLET = "#6A1BA1"


def panel(d: base.Diagram, x: int, y: int, w: int, h: int, n: int, title: str) -> None:
    d.rect(x, y, w, h, fill="#FFFFFF", stroke=INK, width=1.2, rounded=1, extra="arcSize=5;")
    d.rect(x, y, w, 42, fill="#F5F5F5", stroke=INK, width=1.0, rounded=1, extra="arcSize=5;")
    d.ellipse(x + 10, y + 6, 31, 31, str(n), fill="#050505", stroke="#050505", width=1, font_size=17, font_color="#FFFFFF")
    d.text(x + 48, y + 4, w - 58, 34, title, font_size=20, align="left")


def small_graph(d: base.Diagram, cx: float, cy: float, scale: float = 1.0, square_last: bool = False) -> None:
    pts = [(-48, -18), (-25, -54), (7, -38), (42, -52), (54, -8), (24, 35), (-16, 48), (-52, 18)]
    edges = [(0, 1), (0, 2), (0, 7), (1, 2), (2, 3), (2, 5), (2, 6), (3, 4), (3, 5), (4, 5), (5, 6), (6, 7)]
    colors = [BLUE, ORANGE, GREEN, ORANGE, PURPLE, PURPLE, BLUE, ORANGE]
    for a, b in edges:
        d.line(cx + pts[a][0] * scale, cy + pts[a][1] * scale, cx + pts[b][0] * scale, cy + pts[b][1] * scale, color=INK, width=0.9)
    for i, ((dx, dy), color) in enumerate(zip(pts, colors)):
        if square_last and i == 4:
            d.rect(cx + dx * scale - 6, cy + dy * scale - 6, 12, 12, fill="#FFFFFF", stroke=INK, width=0.8, rounded=0)
        else:
            d.dot(cx + dx * scale, cy + dy * scale, color if i not in (5,) else GRAY, radius=6.5 * scale, stroke=INK, width=0.8)


def feature_matrix(d: base.Diagram, x: int, y: int, rows: int, cols: int, cw: int, ch: int, blue_cells: set[tuple[int, int]]) -> None:
    for r in range(rows):
        for c in range(cols):
            fill = "#2B70C9" if (r, c) in blue_cells else "#E8EEF8"
            d.rect(x + c * cw, y + r * ch, cw, ch, fill=fill, stroke="#65758B", width=0.55, rounded=0)


def adjacency_matrix(d: base.Diagram, x: int, y: int) -> None:
    vals = [[0, 1, "⋯", 0], [1, 0, "⋯", 1], ["⋮", "⋮", "⋱", "⋮"], [0, 1, "⋯", 0]]
    for r, row in enumerate(vals):
        for c, val in enumerate(row):
            d.rect(x + c * 21, y + r * 20, 21, 20, str(val), fill="#FFFFFF", stroke="#4D4D4D", width=0.55, rounded=0, font_size=12)
    for i, label in enumerate(["v₁", "v₂", "⋮", "vₙ"]):
        d.text(x - 27, y + i * 20, 24, 20, label, font_size=12, italic=True)
    for i, label in enumerate(["v₁", "v₂", "⋯", "vₙ"]):
        d.text(x + i * 21, y - 22, 21, 20, label, font_size=12, italic=True)


def embedding_table(d: base.Diagram, x: int, y: int, title: str) -> None:
    d.rect(x, y, 124, 143, fill="#FFFFFF", stroke="#6B6B6B", width=1, rounded=1)
    d.text(x + 5, y + 7, 114, 22, title, font_size=12)
    colors = [BLUE, ORANGE, GREEN, PURPLE, GRAY]
    for r, color in enumerate(colors):
        d.dot(x + 14, y + 39 + r * 18, color, radius=5, stroke=INK, width=0.6)
        for c in range(2):
            d.rect(x + 30 + c * 25, y + 33 + r * 18, 25, 11, fill="#ECE4FA", stroke="#7D66A8", width=0.55, rounded=0)
        d.text(x + 87, y + 29 + r * 18, 24, 20, "⋯", font_size=15)
    d.text(x + 6, y + 120, 18, 18, "⋮", font_size=15)
    d.text(x + 31, y + 120, 50, 18, "⋮   ⋮", font_size=15)


def representation_scatter(d: base.Diagram, x: int, y: int) -> None:
    d.rect(x, y, 135, 204, fill="#FFFFFF", stroke="#6B6B6B", width=1, rounded=1)
    clusters = [
        (BLUE, [(19, 31), (28, 45), (39, 26), (45, 49), (57, 39), (34, 61)]),
        (ORANGE, [(93, 34), (104, 50), (117, 28), (123, 61), (92, 68), (114, 80), (127, 92)]),
        (GREEN, [(34, 104), (49, 93), (55, 116), (27, 129), (66, 136)]),
        (PURPLE, [(89, 112), (104, 130), (120, 109), (94, 150), (116, 159)]),
        (GRAY, [(62, 82), (73, 98), (71, 124), (67, 153), (78, 177)]),
    ]
    for color, pts in clusters:
        for dx, dy in pts:
            d.dot(x + dx, y + dy, color, radius=3.7, stroke="#FFFFFF", width=0.4)


def ball_outline(d: base.Diagram, cx: int, cy: int, rx: int, ry: int, stroke: str, dots: list[tuple[int, int, str]], width: float = 1.2) -> None:
    d.ellipse(cx - rx, cy - ry, rx * 2, ry * 2, fill="#FFFFFF", stroke=stroke, width=width)
    for dx, dy, color in dots:
        d.dot(cx + dx, cy + dy, color, radius=3.8, stroke=INK, width=0.55)
    d.text(cx - 10, cy - 11, 20, 22, "×", font_size=19, bold=True)


def bar_chart(d: base.Diagram, x: int, y: int, values: list[float], colors: list[str], width: int = 190, height: int = 112) -> None:
    d.line(x, y, x + width, y, color=INK, width=0.8)
    d.line(x, y, x, y - height, color=INK, width=0.8)
    d.text(x - 20, y - height - 2, 20, 20, "1", font_size=12)
    d.text(x - 26, y - height // 2 - 4, 26, 20, "0.5", font_size=12)
    bw = 18
    gap = 18
    for i, (value, color) in enumerate(zip(values, colors)):
        h = int(height * value)
        bx = x + 28 + i * (bw + gap)
        d.rect(bx, y - h, bw, h, fill=color, stroke=INK, width=0.6, rounded=0)
        d.text(bx - 2, y + 5, bw + 4, 20, str(i + 1), font_size=12)


def mini_chart(d: base.Diagram, x: int, y: int, values: list[float], colors: list[str], height: int = 74) -> None:
    d.line(x, y, x + 104, y, color=INK, width=0.75)
    d.line(x, y, x, y - height, color=INK, width=0.75)
    d.text(x - 20, y - height - 2, 20, 18, "1", font_size=11)
    d.text(x - 20, y - 7, 20, 18, "0", font_size=11)
    for i, (value, color) in enumerate(zip(values, colors)):
        h = int(height * value)
        bx = x + 10 + i * 18
        d.rect(bx, y - h, 11, h, fill=color, stroke=INK, width=0.5, rounded=0)
        d.text(bx - 2, y + 3, 15, 18, str(i + 1), font_size=10)


def mlp_network(d: base.Diagram, x: int, y: int, scale: float = 1.0) -> None:
    xs = [x, x + int(60 * scale), x + int(105 * scale)]
    columns = [[-36, 0, 36], [-48, -16, 16, 48], [-18, 18]]
    for a, ysa in enumerate(columns[:-1]):
        for ya in ysa:
            for yb in columns[a + 1]:
                d.line(xs[a] + 8, y + ya, xs[a + 1] - 8, y + yb, color="#4169B2", width=0.7)
    for col, ys in enumerate(columns):
        for yy in ys:
            d.ellipse(xs[col] - 9, y + yy - 9, 18, 18, fill="#FFFFFF", stroke="#163B76", width=1.0)


def build() -> base.Diagram:
    d = base.Diagram()
    d.model.set("pageWidth", "1536")
    d.model.set("pageHeight", "1024")
    d.model.set("dx", "1536")
    d.model.set("dy", "1024")
    d.rect(0, 0, 1536, 1024, fill="#FFFFFF", stroke="none", width=0, rounded=0)

    panel(d, 6, 4, 798, 438, 1, "Label-free graph representation")
    panel(d, 812, 4, 716, 438, 2, "Granular-ball partition in representation space")
    panel(d, 6, 451, 949, 482, 3, "Leave-One-Out Ball-Level Label Consensus")
    panel(d, 965, 451, 563, 482, 4, "X-only MLP training and graph-free inference")

    # Panel 1: label-free GRACE representation
    d.rect(18, 56, 144, 368, fill="#FFFFFF", stroke="#707070", width=0.9, rounded=1)
    d.text(38, 58, 104, 42, "Input graph<br><i>G</i> = (<i>V</i>, <i>E</i>, <i>X</i>)", font_size=14)
    small_graph(d, 89, 148, 0.95)
    d.text(33, 196, 112, 25, "<i>X</i>  node features", font_size=13)
    feature_matrix(d, 34, 220, 3, 6, 19, 17, {(0, 0), (0, 3), (1, 2), (1, 3), (1, 4), (2, 1), (2, 4)})
    d.text(46, 278, 90, 22, "<i>A</i>  adjacency", font_size=13)
    adjacency_matrix(d, 51, 326)
    d.text(141, 130, 20, 25, "⋯", font_size=17)
    d.text(141, 228, 20, 25, "⋯", font_size=17)

    d.arrow(163, 220, 204, 220, color=INK, width=1.2)
    d.arrow(163, 291, 204, 291, color=INK, width=1.2)
    d.rect(212, 57, 145, 181, fill="#FFFFFF", stroke="#707070", width=0.9, rounded=1)
    d.text(232, 60, 105, 40, "Augmented views<br>View 1", font_size=13)
    small_graph(d, 285, 148, 0.85, square_last=True)
    d.text(224, 195, 122, 42, "edge dropping +<br>feature masking", font_size=12)
    d.rect(212, 251, 145, 164, fill="#FFFFFF", stroke="#707070", width=0.9, rounded=1)
    d.text(254, 255, 65, 24, "View 2", font_size=13)
    small_graph(d, 285, 333, 0.85, square_last=True)

    d.arrow(357, 136, 392, 176, color=INK, width=1.2)
    d.arrow(357, 336, 392, 298, color=INK, width=1.2)
    d.rect(362, 176, 66, 171, fill="#F6F2FA", stroke="#707070", width=1, rounded=1)
    d.text(369, 192, 52, 64, "Shared<br>GRACE<br>encoder", font_size=14)
    d.text(375, 266, 40, 28, "g<sub>ψ</sub>", font_size=15, italic=True)
    small_graph(d, 395, 305, 0.34)

    d.arrow(428, 232, 470, 191, color=INK, width=1.2)
    d.arrow(428, 274, 470, 320, color=INK, width=1.2)
    embedding_table(d, 477, 77, "Embeddings (View 1)")
    embedding_table(d, 477, 270, "Embeddings (View 2)")
    d.arrow(539, 246, 539, 265, color=VIOLET, width=1.5, dashed=True)
    d.text(548, 225, 78, 47, "Contrastive<br>alignment", font_size=12, color=VIOLET, align="left")
    d.arrow(602, 149, 652, 196, color=INK, width=1.2)
    d.arrow(602, 340, 652, 287, color=INK, width=1.2)
    d.text(671, 93, 116, 48, "Frozen normalized<br>representation H̄", font_size=13)
    representation_scatter(d, 662, 148)
    d.line(196, 292, 196, 437, color=INK, width=1, dashed=True)
    d.line(196, 401, 392, 401, color=INK, width=1, dashed=True, arrow="blockThin")
    d.text(231, 389, 64, 42, "labels<br>not used", font_size=12)

    # Panel 2: label-agnostic granular-ball partition
    d.text(825, 48, 360, 26, "label-agnostic partition (no labels used)", font_size=14, align="left")
    d.rect(821, 76, 182, 292, fill="#FFFFFF", stroke="#707070", width=0.9, rounded=1)
    d.text(845, 81, 126, 26, "(a)  Parent ball", font_size=14)
    parent_dots = [(-45, -45, BLUE), (-18, -62, BLUE), (20, -58, ORANGE), (46, -37, ORANGE), (-55, 1, BLUE), (-12, -5, ORANGE), (34, 0, ORANGE), (-46, 48, PURPLE), (-5, 58, GREEN), (43, 47, PURPLE)]
    ball_outline(d, 911, 230, 76, 87, INK, parent_dots)
    d.rect(1012, 76, 182, 292, fill="#FFFFFF", stroke="#707070", width=0.9, rounded=1)
    d.text(1032, 81, 142, 26, "(b)  2-means split", font_size=14)
    ball_outline(d, 1106, 165, 57, 58, INK, [(-30, -24, BLUE), (-5, -38, BLUE), (25, -22, ORANGE), (-28, 7, BLUE), (8, -2, BLUE), (30, 18, RED)])
    ball_outline(d, 1106, 294, 57, 58, GREEN, [(-31, -20, GREEN), (-5, -35, PURPLE), (26, -17, GREEN), (-25, 11, GREEN), (8, 5, PURPLE), (31, 27, PURPLE)])
    d.line(987, 206, 1047, 166, color=INK, width=1, dashed=True, arrow="blockThin")
    d.line(987, 252, 1047, 294, color=INK, width=1, dashed=True, arrow="blockThin")
    d.arrow(1194, 221, 1202, 221, color=INK, width=1.2)

    d.rect(1202, 76, 316, 292, fill="#FFFFFF", stroke="#707070", width=0.9, rounded=1)
    d.text(1235, 81, 250, 26, "(c)  Final granular-ball partition", font_size=14)
    balls = [
        (1243, 151, 31, 29, BLUE, [(-12, -8, BLUE), (10, -9, PURPLE), (0, 10, BLUE)]),
        (1304, 151, 35, 33, BLUE, [(-15, -10, BLUE), (8, -14, BLUE), (18, 8, ORANGE), (-5, 13, BLUE)]),
        (1417, 164, 63, 55, INK, [(-27, -17, ORANGE), (0, -26, ORANGE), (27, -9, ORANGE), (-18, 18, ORANGE), (20, 23, ORANGE)]),
        (1239, 226, 28, 27, PURPLE, [(-10, -7, PURPLE), (9, -5, PURPLE), (0, 11, PURPLE)]),
        (1290, 227, 30, 28, ORANGE, [(-10, -8, ORANGE), (10, 8, ORANGE), (-6, 10, GRAY)]),
        (1351, 241, 47, 43, INK, [(-16, -12, PURPLE), (10, -14, PURPLE), (21, 9, PURPLE), (-10, 17, PURPLE)]),
        (1260, 307, 34, 32, RED, [(-12, -8, PURPLE), (9, -12, PURPLE), (15, 10, PURPLE), (-8, 13, PURPLE)]),
        (1461, 306, 52, 49, GREEN, [(-23, -14, GREEN), (2, -21, GREEN), (25, -4, GREEN), (-15, 21, GREEN), (20, 21, GREEN)]),
    ]
    for cx, cy, rx, ry, color, dots in balls:
        ball_outline(d, cx, cy, rx, ry, color, dots, width=1.4 if color != RED else 2.0)
    d.text(1155, 371, 350, 49, "balls are disjoint: each node belongs to exactly one ball;<br>splitting stops at B<sub>max</sub> = 64 or by geometry", font_size=13)
    d.line(1260, 341, 1277, 363, color=RED, width=1.4, arrow="blockThin")
    d.line(1277, 363, 1506, 363, color=RED, width=1.4)
    d.line(1506, 363, 1506, 308, color=RED, width=1.4, arrow="blockThin")

    # Cross-panel selected-ball annotation, kept exactly as in the reference.
    d.line(1506, 363, 1506, 431, color=RED, width=1.4)
    d.line(1506, 431, 914, 431, color=RED, width=1.4)
    d.line(914, 431, 914, 493, color=RED, width=1.4)
    d.line(914, 493, 231, 493, color=RED, width=1.4)
    d.text(1324, 423, 130, 25, "selected ball Bⱼ", font_size=13, color=RED)

    # Panel 3: leave-one-out target construction
    d.text(27, 505, 202, 30, "(a)  Selected ball Bⱼ", font_size=15, align="left")
    d.text(285, 505, 238, 30, "(b)  Leave-one-out consensus", font_size=15, align="left")
    d.text(592, 505, 185, 30, "(c)  Soft target tᵥ", font_size=15, align="left")
    d.line(246, 505, 246, 920, color="#BFC4C9", width=1, dashed=True)
    d.line(553, 505, 553, 920, color="#BFC4C9", width=1, dashed=True)

    d.ellipse(18, 564, 214, 235, fill="#FFFFFF", stroke=INK, width=1.1)
    selected_points = [(48, 614, GRAY), (80, 607, BLUE), (126, 588, BLUE), (166, 611, ORANGE), (191, 650, GRAY), (73, 681, GREEN), (116, 675, GREEN), (164, 690, PURPLE), (44, 718, PURPLE), (95, 734, GREEN), (132, 748, PURPLE), (186, 739, GRAY), (73, 770, PURPLE), (164, 776, GRAY)]
    for x, y, color in selected_points:
        d.dot(x, y, color, radius=6, stroke=INK, width=0.7)
    d.dot(116, 672, GREEN, radius=7, stroke=INK, width=0.8)
    d.ellipse(99, 654, 34, 34, "", fill="none", stroke=RED, width=2.0)
    d.text(126, 636, 25, 24, "v", font_size=15, italic=True)
    d.ellipse(143, 657, 18, 18, "×", fill="#FFFFFF", stroke=RED, width=1.5, font_size=18, font_color=RED)
    d.line(231, 492, 158, 665, color=RED, width=1.3, arrow="blockThin")

    d.text(327, 554, 167, 28, "p<sub>j,−v</sub>   (exclude v)", font_size=14)
    d.arrow(384, 581, 384, 617, color=INK, width=1.0)
    bar_chart(d, 309, 706, [0.33, 0.73, 0.19, 0.17, 0.0], [BLUE, ORANGE, GREEN, PURPLE, GRAY], width=210, height=95)
    d.text(382, 725, 82, 24, "classes", font_size=13)
    d.rect(273, 756, 256, 62, "<i>q</i><sub>j,−v</sub> = (1 − β) <i>p</i><sub>j,−v</sub> + β <i>e</i><sub>m<sub>j,−v</sub></sub>", fill="#FFFFFF", stroke=GREEN, width=1.1, font_size=16)
    d.text(292, 824, 225, 36, "if tie: q<sub>j,−v</sub> = p<sub>j,−v</sub> (no one-hot)", font_size=13)

    d.text(574, 551, 72, 44, "noisy<br>one-hot", font_size=12)
    mini_chart(d, 600, 706, [0.0, 0.68, 0.0, 0.0, 0.0], [BLUE, GREEN, ORANGE, PURPLE, GRAY], height=86)
    d.text(671, 649, 30, 30, "+", font_size=26, bold=True)
    d.text(711, 551, 98, 44, "local<br>consensus<br>q<sub>j,−v</sub>", font_size=12)
    mini_chart(d, 722, 706, [0.22, 0.65, 0.12, 0.11, 0.0], [BLUE, ORANGE, GREEN, PURPLE, GRAY], height=86)
    d.text(804, 649, 30, 30, "=", font_size=25, bold=True)
    d.text(842, 551, 92, 44, "soft target<br>tᵥ", font_size=12)
    mini_chart(d, 845, 706, [0.15, 0.54, 0.18, 0.13, 0.0], [BLUE, ORANGE, GREEN, PURPLE, GRAY], height=86)
    d.rect(581, 758, 343, 61, "<i>t</i><sub>v</sub> = (1 − α<sub>j,−v</sub>) · <i>e</i><sub>ỹᵥ</sub> + α<sub>j,−v</sub> · <i>q</i><sub>j,−v</sub>", fill="#FFFFFF", stroke=RED, width=1.1, font_size=17)
    d.text(591, 827, 323, 32, "α<sub>j,−v</sub>: vote concentration (0 = keep noisy label)", font_size=13)

    # Panel 4: training and graph-free inference
    d.text(975, 500, 180, 29, "<b>Training</b> (uses graph)", font_size=14, align="left")
    d.text(994, 532, 115, 43, "node features<br>x<sub>T</sub>", font_size=12)
    feature_matrix(d, 979, 590, 1, 5, 20, 16, {(0, 0), (0, 2), (0, 4)})
    d.text(1075, 584, 28, 22, "⋯", font_size=16)
    d.arrow(1103, 598, 1130, 598, color=INK, width=1.1)
    d.text(1130, 518, 125, 28, "MLP (2-layer)", font_size=13)
    mlp_network(d, 1150, 592, 0.85)
    d.text(1173, 642, 65, 24, "f<sub>θ</sub>", font_size=14, italic=True)
    d.arrow(1266, 598, 1290, 598, color=INK, width=1.1)
    d.text(1275, 518, 105, 45, "prediction<br>probabilities", font_size=12)
    d.rect(1303, 562, 18, 78, fill="#E7EEF8", stroke="#5B6F8D", width=0.7, rounded=0)
    for i, color in enumerate(["#C8D8EF", "#AFC7E8", "#DDE6F2", "#B8CCE8"]):
        d.rect(1306, 566 + i * 18, 12, 15, fill=color, stroke="#FFFFFF", width=0.3, rounded=0)
    d.arrow(1324, 598, 1374, 598, color=INK, width=1.1)
    d.text(1361, 518, 126, 46, "soft cross-entropy<br>loss with t<sub>T</sub>", font_size=12)
    d.line(1390, 626, 1390, 568, color=INK, width=0.9, arrow="blockThin")
    for i, (h, color) in enumerate([(30, BLUE), (52, ORANGE), (20, GREEN), (16, PURPLE)]):
        d.rect(1401 + i * 15, 624 - h, 9, h, fill=color, stroke=INK, width=0.5, rounded=0)
    d.line(1381, 625, 1468, 625, color=INK, width=0.8)
    d.line(1314, 646, 1314, 661, color=RED, width=1.2)
    d.line(1314, 661, 1424, 661, color=RED, width=1.2)
    d.line(1424, 661, 1424, 628, color=RED, width=1.2, arrow="blockThin")

    d.line(975, 670, 1516, 670, color="#7C7C7C", width=0.9, dashed=True)
    d.text(975, 656, 104, 24, "After training", font_size=13, color="#174DFF", align="left")
    d.text(975, 692, 180, 27, "<b>Inference</b> (graph-free)", font_size=14, color="#174DFF", align="left")
    d.text(975, 730, 177, 48, "Test node in the same graph<br>(transductive)", font_size=12, align="left")
    d.text(969, 794, 44, 26, "x<sub>test</sub>", font_size=14, italic=True)
    feature_matrix(d, 1015, 807, 1, 4, 20, 16, {(0, 0), (0, 3)})
    d.text(1091, 801, 28, 22, "⋯", font_size=16)
    d.arrow(1120, 814, 1152, 814, color=INK, width=1.1)
    d.text(1180, 706, 118, 43, "Trained MLP<br>(same weights)", font_size=12)
    mlp_network(d, 1205, 808, 0.85)
    d.arrow(1320, 814, 1360, 814, color=INK, width=1.1)
    d.text(1361, 730, 102, 44, "class<br>probabilities", font_size=12)
    d.rect(1392, 778, 18, 72, fill="#E7EEF8", stroke="#5B6F8D", width=0.7, rounded=0)
    for i, color in enumerate(["#C8D8EF", "#AFC7E8", "#DDE6F2", "#B8CCE8"]):
        d.rect(1395, 782 + i * 16, 12, 13, fill=color, stroke="#FFFFFF", width=0.3, rounded=0)
    d.rect(1028, 868, 366, 55, "<i>X</i> only: no <i>A</i>, no H̄, no neighbor fetching", fill="#FFFFFF", stroke="#174DFF", width=1.1, rounded=1, dashed=True, font_size=15, font_color="#174DFF", extra="fontFamily=Times New Roman;")

    # Legend
    d.rect(6, 945, 1522, 65, fill="#FFFFFF", stroke=INK, width=1.1, rounded=1)
    legend = [(BLUE, "Class 1"), (ORANGE, "Class 2"), (GREEN, "Class 3"), (PURPLE, "Class 4"), (GRAY, "Class 5")]
    lx = 39
    for color, label in legend:
        d.dot(lx, 977, color, radius=10, stroke=INK, width=0.8)
        d.text(lx + 17, 960, 72, 34, label, font_size=12, align="left")
        lx += 116
    d.ellipse(615, 964, 25, 25, "", fill="#FFFFFF", stroke=INK, width=1)
    d.dot(627, 976, ORANGE, radius=5, stroke=BLUE, width=1)
    d.text(650, 951, 118, 49, "training node<br>(noisy label)", font_size=12, align="left")
    d.ellipse(770, 964, 24, 24, "", fill="#FFFFFF", stroke=INK, width=1)
    d.text(802, 960, 116, 34, "non-voting node", font_size=12, align="left")
    d.line(928, 977, 974, 977, color=INK, width=1.2, arrow="blockThin")
    d.text(981, 960, 76, 34, "Data flow", font_size=12, align="left")
    d.line(1072, 977, 1124, 977, color=VIOLET, width=1.2, dashed=True, arrow="blockThin")
    d.text(1130, 960, 148, 34, "Contrastive alignment", font_size=12, align="left")
    d.ellipse(1338, 957, 38, 38, "×", fill="#FFFFFF", stroke=RED, width=1.4, font_size=25, font_color=RED)
    d.text(1384, 960, 130, 34, "Excluded self-vote", font_size=12, align="left")
    return d


if __name__ == "__main__":
    diagram = build()
    diagram.save()
    print(base.OUT)
