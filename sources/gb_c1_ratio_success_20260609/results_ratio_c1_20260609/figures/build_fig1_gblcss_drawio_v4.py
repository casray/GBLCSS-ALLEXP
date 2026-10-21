"""Build the editable Draw.io overview figure for GB-LCSS.

The diagram is intentionally assembled from native Draw.io cells so every
panel, label, connector, and visual motif remains editable.
"""

from __future__ import annotations

import math
from pathlib import Path
from xml.etree import ElementTree as ET


OUT = Path(__file__).with_name("fig1_gblcss_overview_drawio_v9.drawio")

CANVAS_W = 1900
CANVAS_H = 1280

INK = "#25313C"
MUTED = "#64717D"
LINE = "#AAB4BD"
PANEL = "#FFFFFF"
HEADER = "#F2F4F6"
BLUE = "#0072B2"
BLUE_LIGHT = "#E8F3FA"
ORANGE = "#E69F00"
ORANGE_LIGHT = "#FFF4D6"
GREEN = "#009E73"
GREEN_LIGHT = "#E6F6F1"
PURPLE = "#CC79A7"
PURPLE_LIGHT = "#FAEDF5"
GRAY = "#999999"
GRAY_LIGHT = "#EFF1F3"
RED = "#D55E00"


class Diagram:
    def __init__(self) -> None:
        self.mxfile = ET.Element("mxfile", host="drawio", version="31.1.5")
        self.diagram = ET.SubElement(self.mxfile, "diagram", name="GB-LCSS overview", id="gblcss-v4")
        self.model = ET.SubElement(
            self.diagram,
            "mxGraphModel",
            dx=str(CANVAS_W),
            dy=str(CANVAS_H),
            grid="1",
            gridSize="10",
            guides="1",
            tooltips="1",
            connect="1",
            arrows="1",
            fold="1",
            page="1",
            pageScale="1",
            pageWidth=str(CANVAS_W),
            pageHeight=str(CANVAS_H),
            math="1",
            shadow="0",
            background="#FFFFFF",
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", id="0")
        ET.SubElement(self.root, "mxCell", id="1", parent="0")
        self.counter = 2

    def _id(self, prefix: str = "c") -> str:
        value = f"{prefix}{self.counter}"
        self.counter += 1
        return value

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        value: str = "",
        *,
        fill: str = PANEL,
        stroke: str = LINE,
        width: float = 1.2,
        rounded: int = 1,
        font_size: int = 15,
        font_color: str = INK,
        align: str = "center",
        valign: str = "middle",
        dashed: bool = False,
        opacity: int = 100,
        extra: str = "",
        parent: str = "1",
        cid: str | None = None,
    ) -> str:
        cid = cid or self._id("r")
        style = (
            f"rounded={rounded};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            f"strokeWidth={width};fontFamily=Helvetica;fontSize={font_size};fontColor={font_color};"
            f"align={align};verticalAlign={valign};spacing=6;opacity={opacity};"
            f"dashed={1 if dashed else 0};{extra}"
        )
        cell = ET.SubElement(self.root, "mxCell", id=cid, value=value, style=style, vertex="1", parent=parent)
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})
        return cid

    def text(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        value: str,
        *,
        font_size: int = 15,
        color: str = INK,
        bold: bool = False,
        align: str = "center",
        valign: str = "middle",
        italic: bool = False,
        parent: str = "1",
        cid: str | None = None,
    ) -> str:
        flags = (1 if bold else 0) + (2 if italic else 0)
        return self.rect(
            x,
            y,
            w,
            h,
            value,
            fill="none",
            stroke="none",
            width=0,
            rounded=0,
            font_size=font_size,
            font_color=color,
            align=align,
            valign=valign,
            extra=f"fontStyle={flags};",
            parent=parent,
            cid=cid,
        )

    def ellipse(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        value: str = "",
        *,
        fill: str = PANEL,
        stroke: str = INK,
        width: float = 1.2,
        font_size: int = 14,
        font_color: str = INK,
        dashed: bool = False,
        opacity: int = 100,
        parent: str = "1",
        cid: str | None = None,
    ) -> str:
        cid = cid or self._id("e")
        style = (
            f"ellipse;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={width};"
            f"fontFamily=Helvetica;fontSize={font_size};fontColor={font_color};dashed={1 if dashed else 0};opacity={opacity};"
        )
        cell = ET.SubElement(self.root, "mxCell", id=cid, value=value, style=style, vertex="1", parent=parent)
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})
        return cid

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = INK,
        width: float = 1.2,
        dashed: bool = False,
        arrow: str = "none",
        label: str = "",
        label_size: int = 13,
        curved: bool = False,
        cid: str | None = None,
    ) -> str:
        cid = cid or self._id("l")
        style = (
            "edgeStyle=none;rounded=0;orthogonalLoop=1;jettySize=auto;"
            f"html=1;strokeColor={color};strokeWidth={width};dashed={1 if dashed else 0};"
            f"endArrow={arrow};endFill=1;fontFamily=Helvetica;fontSize={label_size};fontColor={color};"
            f"curved={1 if curved else 0};"
        )
        cell = ET.SubElement(self.root, "mxCell", id=cid, value=label, style=style, edge="1", parent="1")
        geom = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
        ET.SubElement(geom, "mxPoint", x=str(x1), y=str(y1), **{"as": "sourcePoint"})
        ET.SubElement(geom, "mxPoint", x=str(x2), y=str(y2), **{"as": "targetPoint"})
        return cid

    def arrow(self, x1: float, y1: float, x2: float, y2: float, **kwargs) -> str:
        return self.line(x1, y1, x2, y2, arrow="blockThin", **kwargs)

    def path_arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        points: list[tuple[float, float]],
        *,
        color: str = INK,
        width: float = 1.8,
        dashed: bool = False,
    ) -> str:
        cid = self._id("p")
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
            f"strokeColor={color};strokeWidth={width};dashed={1 if dashed else 0};"
            "endArrow=blockThin;endFill=1;"
        )
        cell = ET.SubElement(self.root, "mxCell", id=cid, value="", style=style, edge="1", parent="1")
        geom = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
        ET.SubElement(geom, "mxPoint", x=str(x1), y=str(y1), **{"as": "sourcePoint"})
        ET.SubElement(geom, "mxPoint", x=str(x2), y=str(y2), **{"as": "targetPoint"})
        point_array = ET.SubElement(geom, "Array", **{"as": "points"})
        for px, py in points:
            ET.SubElement(point_array, "mxPoint", x=str(px), y=str(py))
        return cid

    def dot(
        self,
        x: float,
        y: float,
        color: str,
        *,
        radius: float = 7,
        stroke: str = INK,
        width: float = 0.8,
        opacity: int = 100,
    ) -> str:
        return self.ellipse(
            x - radius,
            y - radius,
            radius * 2,
            radius * 2,
            fill=color,
            stroke=stroke,
            width=width,
            opacity=opacity,
        )

    def save(self) -> None:
        ET.indent(self.mxfile, space="  ")
        tree = ET.ElementTree(self.mxfile)
        tree.write(OUT, encoding="utf-8", xml_declaration=True)


def panel(d: Diagram, x: int, y: int, w: int, h: int, number: int, title: str) -> None:
    d.rect(x, y, w, h, fill=PANEL, stroke=INK, width=1.4, rounded=1, extra="arcSize=8;")
    d.rect(x, y, w, 44, fill=HEADER, stroke=INK, width=1.2, rounded=1, extra="arcSize=8;")
    d.ellipse(x + 12, y + 7, 30, 30, str(number), fill=INK, stroke=INK, width=1, font_size=17, font_color="#FFFFFF")
    d.text(x + 50, y + 5, w - 60, 34, title, font_size=19, bold=True, align="left")


def graph_icon(d: Diagram, cx: float, cy: float, scale: float = 1.0) -> None:
    pts = [(-48, -12), (-28, -48), (8, -38), (42, -52), (54, -8), (24, 32), (-12, 44), (-50, 24)]
    edges = [(0, 1), (0, 2), (0, 7), (1, 2), (2, 3), (2, 5), (2, 6), (3, 4), (3, 5), (4, 5), (5, 6), (6, 7)]
    colors = [BLUE, ORANGE, GREEN, PURPLE, GRAY, BLUE, PURPLE, ORANGE]
    for a, b in edges:
        d.line(cx + pts[a][0] * scale, cy + pts[a][1] * scale, cx + pts[b][0] * scale, cy + pts[b][1] * scale, color=MUTED, width=1)
    for (dx, dy), color in zip(pts, colors):
        d.dot(cx + dx * scale, cy + dy * scale, color, radius=7 * scale, width=0.8)


def matrix_icon(d: Diagram, x: float, y: float, rows: int, cols: int, cell: int, accent: str, label: str) -> None:
    d.text(x, y - 28, cols * cell, 22, label, font_size=13, bold=True)
    for r in range(rows):
        for c in range(cols):
            fill = accent if (r + 2 * c) % 4 == 0 else "#FFFFFF"
            d.rect(x + c * cell, y + r * cell, cell, cell, fill=fill, stroke=LINE, width=0.6, rounded=0)


def adjacency_icon(d: Diagram, x: float, y: float, cell: int = 15) -> None:
    """Draw an adjacency matrix with the node-index labels used in the reference figure."""
    labels = ["v₁", "v₂", "⋯", "vₙ"]
    values = [
        [0, 1, "⋯", 0],
        [1, 0, "⋯", 1],
        ["⋮", "⋮", "⋱", "⋮"],
        [0, 1, "⋯", 0],
    ]
    d.text(x - 4, y - 48, cell * 4 + 8, 22, "Adjacency A", font_size=13, bold=True)
    for c, label in enumerate(labels):
        d.text(x + c * cell, y - 23, cell, 18, label, font_size=10, color=MUTED)
    for r, label in enumerate(labels):
        d.text(x - 24, y + r * cell, 20, cell, label, font_size=10, color=MUTED, align="right")
        for c in range(4):
            d.rect(
                x + c * cell,
                y + r * cell,
                cell,
                cell,
                str(values[r][c]),
                fill="#FFFFFF" if values[r][c] in (0, "⋯", "⋮", "⋱") else GRAY_LIGHT,
                stroke=LINE,
                width=0.6,
                rounded=0,
                font_size=9,
            )


def packed_dots(rx: float, ry: float, count: int, colors: list[str]) -> list[tuple[float, float, str]]:
    """Deterministic sunflower packing inside an ellipse."""
    points: list[tuple[float, float, str]] = []
    golden_angle = math.pi * (3 - math.sqrt(5))
    for i in range(count):
        radius = math.sqrt((i + 0.8) / (count + 1.5)) * 0.82
        angle = i * golden_angle
        points.append((rx * radius * math.cos(angle), ry * radius * math.sin(angle), colors[i % len(colors)]))
    return points


def scatter(d: Diagram, x: float, y: float, w: float, h: float) -> None:
    d.rect(x, y, w, h, fill="#FFFFFF", stroke=LINE, width=1, rounded=1)
    clusters = [
        (BLUE, [(12, 22), (21, 16), (29, 28), (18, 36), (36, 20), (43, 33), (50, 18), (54, 43), (31, 47)]),
        (ORANGE, [(81, 24), (91, 17), (101, 29), (87, 40), (112, 20), (121, 36), (128, 25), (105, 48), (118, 53)]),
        (GREEN, [(20, 84), (31, 74), (42, 88), (15, 101), (50, 105), (61, 92), (37, 113), (24, 119)]),
        (PURPLE, [(86, 79), (98, 69), (111, 85), (80, 99), (105, 108), (121, 96), (128, 115), (92, 119)]),
        (GRAY, [(64, 50), (73, 61), (60, 70), (75, 83), (69, 96), (83, 53), (55, 61)]),
    ]
    sx = w / 140
    sy = h / 135
    for color, pts in clusters:
        for dx, dy in pts:
            d.dot(x + dx * sx, y + dy * sy, color, radius=4.2, stroke="#FFFFFF", width=0.5)


def ball(
    d: Diagram,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    color: str,
    dots: list[tuple[float, float, str]],
    *,
    width: float = 1.8,
    dot_radius: float = 4.5,
    show_center: bool = True,
    dashed: bool = False,
    fill: str = "#FFFFFF",
) -> None:
    d.ellipse(cx - rx, cy - ry, rx * 2, ry * 2, fill=fill, stroke=color, width=width, dashed=dashed)
    for dx, dy, dot_color in dots:
        d.dot(
            cx + dx,
            cy + dy,
            dot_color,
            radius=dot_radius,
            stroke="#FFFFFF",
            width=0.5,
            opacity=48 if dot_color == GRAY else 100,
        )
    if show_center:
        d.text(cx - 9, cy - 9, 18, 18, "×", font_size=18, bold=True)


def histogram(d: Diagram, x: float, y: float, vals: list[float], colors: list[str], labels: list[str], max_h: float = 88) -> None:
    d.line(x - 8, y, x + 145, y, color=MUTED, width=1)
    d.line(x, y, x, y - max_h - 8, color=MUTED, width=1)
    bw = 19
    gap = 9
    for i, (v, color, label) in enumerate(zip(vals, colors, labels)):
        h = max_h * v
        bx = x + 10 + i * (bw + gap)
        d.rect(bx, y - h, bw, h, fill=color, stroke=INK, width=0.6, rounded=0)
        d.text(bx - 2, y + 3, bw + 4, 18, label, font_size=11)


def neural_net(d: Diagram, x: float, y: float, scale: float = 1.0) -> None:
    cols = [3, 4, 3]
    xs = [x, x + 72 * scale, x + 144 * scale]
    ys_by_col: list[list[float]] = []
    for col, count in enumerate(cols):
        ys = [y + (i - (count - 1) / 2) * 35 * scale for i in range(count)]
        ys_by_col.append(ys)
        for yy in ys:
            d.ellipse(xs[col] - 9 * scale, yy - 9 * scale, 18 * scale, 18 * scale, fill="#FFFFFF", stroke=PURPLE, width=1.3)
    for col in range(2):
        for ya in ys_by_col[col]:
            for yb in ys_by_col[col + 1]:
                d.line(xs[col] + 10 * scale, ya, xs[col + 1] - 10 * scale, yb, color="#8EA3C5", width=0.7)


def build() -> Diagram:
    d = Diagram()

    # Panel geometry
    panel(d, 10, 10, 930, 550, 1, "Label-free graph representation")
    panel(d, 950, 10, 940, 550, 2, "Label-agnostic granular-ball partition")
    panel(d, 10, 570, 1050, 630, 3, "Leave-one-out ball-level label consensus")
    panel(d, 1070, 570, 820, 630, 4, "X-only MLP training and inference")

    # Stage 1
    d.text(25, 58, 150, 34, "Input graph", font_size=15, bold=True)
    d.text(25, 84, 150, 25, "G = (V, E, X)", font_size=14, italic=True)
    graph_icon(d, 105, 180, 0.9)
    matrix_icon(d, 48, 292, 4, 6, 17, BLUE, "Node features X")
    adjacency_icon(d, 78, 442, 15)

    d.path_arrow(178, 205, 198, 180, [(188, 205), (188, 180)], color=INK, width=1.6)
    d.path_arrow(178, 205, 198, 380, [(188, 205), (188, 380)], color=INK, width=1.6)
    d.text(190, 70, 190, 28, "Two augmented views", font_size=15, bold=True)
    d.rect(198, 110, 180, 150, fill="#FFFFFF", stroke=LINE, width=1.1)
    d.rect(198, 305, 180, 150, fill="#FFFFFF", stroke=LINE, width=1.1)
    graph_icon(d, 288, 166, 0.60)
    graph_icon(d, 288, 361, 0.60)
    d.text(207, 216, 162, 36, "edge dropping +<br>feature masking", font_size=11, color=MUTED)
    d.text(207, 414, 162, 26, "independent augmentation", font_size=11, color=MUTED)

    d.arrow(378, 185, 400, 185, color=INK, width=1.6)
    d.arrow(378, 380, 400, 380, color=INK, width=1.6)
    d.rect(400, 165, 112, 245, "<b>Shared</b><br><b>GRACE</b><br>encoder<br><br><i>g</i><sub>ψ</sub>", fill=BLUE_LIGHT, stroke=BLUE, width=1.8, font_size=14)
    d.text(395, 438, 122, 28, "labels are not used", font_size=12, color=RED, bold=True)

    d.arrow(512, 185, 540, 185, color=INK, width=1.6)
    d.arrow(512, 380, 540, 380, color=INK, width=1.6)
    matrix_icon(d, 542, 137, 5, 5, 15, BLUE, "Embedding H¹")
    matrix_icon(d, 542, 332, 5, 5, 15, PURPLE, "Embedding H²")
    d.arrow(583, 225, 583, 304, color=PURPLE, width=1.7, dashed=True)
    d.text(591, 244, 70, 42, "contrastive<br>alignment", font_size=10, color=PURPLE, align="left")
    d.arrow(617, 175, 672, 175, color=INK, width=1.6)
    d.arrow(617, 370, 672, 370, color=INK, width=1.6)
    scatter(d, 675, 105, 250, 405)
    d.text(675, 62, 250, 38, "Frozen normalized representation H̄", font_size=14, bold=True)

    # Stage 2
    d.text(970, 56, 520, 25, "Geometry-only construction in frozen embedding space H̄", font_size=13, color=GREEN, bold=True, align="left")

    # Four-column construction sequence, matching the granular-ball literature style.
    columns = [(970, 96, 155, 424), (1140, 96, 155, 424), (1310, 96, 180, 424)]
    for x, y, w, h in columns:
        d.rect(x, y, w, h, fill="#FFFFFF", stroke="#4E8A58", width=1.1, rounded=1, dashed=True)
    d.rect(1510, 84, 365, 446, fill="#FCFDFF", stroke="#4267D5", width=1.5, rounded=1)

    # Colors reveal latent class structure only for illustration. They are never read by
    # the recursive 2-means partition.
    class_blue = [BLUE, BLUE, BLUE, BLUE, BLUE, BLUE, PURPLE, GREEN]
    class_orange = [ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, ORANGE, PURPLE, GREEN]
    class_green = [GREEN, GREEN, GREEN, GREEN, GREEN, GREEN, BLUE, ORANGE]
    class_purple = [PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE, BLUE, ORANGE]
    all_classes = [BLUE, ORANGE, GREEN, PURPLE]

    d.text(978, 106, 139, 38, "Initialization<br>B₀ = V", font_size=13, color=GREEN, bold=True)
    initial_dots = packed_dots(63, 63, 72, all_classes)
    ball(d, 1047, 305, 72, 72, INK, initial_dots, width=1.2, dot_radius=3.1, dashed=True)

    d.arrow(1125, 302, 1140, 302, color=INK, width=1.5)
    d.text(1148, 106, 139, 38, "Recursive<br>bi-partition", font_size=13, color=GREEN, bold=True)
    ball(d, 1183, 305, 46, 46, BLUE, packed_dots(40, 40, 34, class_blue), width=1.2, dot_radius=3.0, dashed=True)
    ball(d, 1252, 305, 46, 46, RED, packed_dots(40, 40, 34, class_orange), width=1.2, dot_radius=3.0, dashed=True)

    d.arrow(1295, 302, 1310, 302, color=INK, width=1.5)
    d.text(1318, 106, 164, 38, "Further recursive<br>splitting", font_size=13, color=GREEN, bold=True)
    split_specs = [
        (1355, 245, 42, BLUE, class_blue),
        (1440, 245, 42, RED, class_orange),
        (1355, 385, 42, PURPLE, class_purple),
        (1440, 385, 42, GREEN, class_green),
    ]
    for cx, cy, radius, color, palette in split_specs:
        ball(d, cx, cy, radius, radius, color, packed_dots(radius, radius, 28, palette), width=1.2, dot_radius=2.9, dashed=True)

    d.arrow(1490, 302, 1510, 302, color=INK, width=1.5)
    d.text(1520, 92, 345, 42, "Embedding space H̄ covered by granular balls<br>B = {Bⱼ}ⱼ₌₁ᴮ", font_size=13, color="#2447A8", bold=True)

    # Dense projected covers in the final H space. Each cover is a true circle with no
    # fill, so complete dashed circumferences remain visible through overlaps.
    cover_specs = [
        (1555, 190, 38, BLUE, 32, class_blue),
        (1635, 183, 40, PURPLE, 34, class_purple),
        (1715, 190, 42, PURPLE, 36, class_purple),
        (1800, 190, 37, ORANGE, 31, class_orange),
        (1552, 292, 42, GREEN, 36, class_green),
        (1637, 284, 44, GREEN, 38, class_green),
        (1722, 294, 43, PURPLE, 37, class_purple),
        (1800, 290, 37, ORANGE, 31, class_orange),
        (1555, 410, 43, PURPLE, 37, class_purple),
        (1725, 410, 41, ORANGE, 35, class_orange),
        (1802, 410, 35, ORANGE, 29, class_orange),
    ]
    for cx, cy, radius, color, count, palette in cover_specs:
        ball(
            d,
            cx,
            cy,
            radius,
            radius,
            color,
            packed_dots(radius, radius, count, palette),
            width=1.1,
            dot_radius=2.35,
            dashed=True,
            fill="none",
        )

    # Use one shared node layout for the selected ball in Stages 2 and 3.
    selected_unit = packed_dots(1, 1, 100, [GREEN, GREEN, GREEN, GREEN, GREEN, GREEN, GREEN, BLUE, ORANGE, PURPLE])
    selected_stage2 = [(dx * 45, dy * 45, color) for dx, dy, color in selected_unit]
    ball(d, 1638, 410, 45, 45, GREEN, selected_stage2, width=1.1, dot_radius=1.75, show_center=True, dashed=True, fill="none")
    d.ellipse(1593, 365, 90, 90, fill="none", stroke=RED, width=2.3)
    d.text(1580, 463, 116, 25, "selected ball Bⱼ", font_size=11, color=RED, bold=True)
    d.text(970, 534, 895, 18, "Label-agnostic recursive 2-means; colors are shown only for visual interpretation", font_size=11, color=MUTED)

    # The selected ball is carried into Stage 3 rather than redrawn as an unrelated example.
    d.path_arrow(1638, 456, 195, 682, [(1638, 565), (195, 565)], color=RED, width=1.8)

    # Stage 3. The geometry is identical to the selected Stage-2 ball. Only one in
    # five nodes retains a visible noisy training label; the remaining 80% are gray.
    d.text(28, 622, 270, 26, "(a) Select a granular ball Bⱼ", font_size=15, bold=True)
    noisy_vote_colors = [GREEN, ORANGE, BLUE, GREEN, PURPLE]
    selected = [
        (dx * 138, dy * 138, noisy_vote_colors[(idx // 5) % len(noisy_vote_colors)] if idx % 5 == 0 else GRAY)
        for idx, (dx, dy, _) in enumerate(selected_unit)
    ]
    ball(d, 195, 825, 140, 140, RED, selected, width=2.2, dot_radius=3.9, show_center=False, fill="none")
    vote_dx, vote_dy, _ = next(item for item in selected if item[2] != GRAY)
    vote_x, vote_y = 195 + vote_dx, 825 + vote_dy
    d.ellipse(vote_x - 15, vote_y - 15, 30, 30, "v", fill=GREEN_LIGHT, stroke=RED, width=2.4, font_size=14, font_color=INK)
    d.line(vote_x - 15, vote_y - 15, vote_x + 15, vote_y + 15, color=RED, width=2)
    d.line(vote_x + 15, vote_y - 15, vote_x - 15, vote_y + 15, color=RED, width=2)
    d.text(65, 980, 260, 48, "Remove node v from its own vote<br>before computing local consensus", font_size=13, color=RED, bold=True)

    d.arrow(345, 825, 390, 825, color=INK, width=1.6)
    d.text(380, 622, 285, 26, "(b) Leave-one-out consensus", font_size=15, bold=True)
    histogram(d, 415, 820, [0.37, 0.82, 0.23, 0.19, 0.0], [BLUE, ORANGE, GREEN, PURPLE, GRAY], ["1", "2", "3", "4", "5"])
    d.text(395, 680, 205, 32, "pⱼ,−ᵥ  (exclude v)", font_size=16, bold=True)
    d.rect(380, 865, 270, 66, "<i>q</i><sub>j,−v</sub> = (1 − β)<i>p</i><sub>j,−v</sub> + β<i>e</i><sub>m</sub>", fill=GREEN_LIGHT, stroke=GREEN, width=1.5, font_size=16)
    d.text(385, 940, 260, 44, "If the majority ties: qⱼ,−ᵥ = pⱼ,−ᵥ<br>(tie-safe; no arbitrary one-hot class)", font_size=12, color=MUTED)

    d.arrow(660, 825, 700, 825, color=INK, width=1.6)
    d.text(685, 622, 345, 26, "(c) Construct a soft target tᵥ", font_size=15, bold=True)
    histogram(d, 700, 810, [0.0, 0.72, 0.0, 0.0, 0.0], [BLUE, GREEN, ORANGE, PURPLE, GRAY], ["1", "2", "3", "4", "5"], max_h=75)
    d.text(688, 696, 175, 28, "noisy one-hot", font_size=13, color=MUTED)
    d.text(847, 776, 34, 34, "+", font_size=25, bold=True)
    histogram(d, 875, 810, [0.28, 0.76, 0.18, 0.12, 0.0], [BLUE, ORANGE, GREEN, PURPLE, GRAY], ["1", "2", "3", "4", "5"], max_h=75)
    d.text(862, 696, 175, 28, "local consensus", font_size=13, color=MUTED)
    d.rect(680, 865, 345, 68, "<i>t</i><sub>v</sub> = (1 − α<sub>j,−v</sub>)<i>e</i><sub>ỹᵥ</sub> + α<sub>j,−v</sub><i>q</i><sub>j,−v</sub>", fill=ORANGE_LIGHT, stroke=ORANGE, width=1.8, font_size=17)
    d.text(695, 940, 315, 44, "αⱼ,−ᵥ measures vote concentration<br>β = 0.5 in the fixed configuration", font_size=13, color=MUTED)
    d.rect(
        340,
        1090,
        680,
        48,
        "Output: one node-specific, leave-one-out soft target for each noisy training node",
        fill="#F7F8F9",
        stroke=LINE,
        width=1.2,
        font_size=14,
        extra="fontStyle=1;",
    )

    # Stage 4
    d.text(1088, 622, 290, 27, "Training with graph-derived targets", font_size=15, bold=True, align="left")
    matrix_icon(d, 1100, 720, 1, 5, 16, BLUE, "node feature xᵥ")
    d.arrow(1185, 758, 1220, 758, color=INK, width=1.6)
    neural_net(d, 1235, 758, 0.75)
    d.text(1225, 664, 170, 28, "2-layer MLP  fθ", font_size=14, bold=True)
    d.arrow(1350, 758, 1405, 758, color=INK, width=1.6)
    d.rect(1411, 702, 40, 110, fill="#FFFFFF", stroke=LINE, width=1.1)
    for i, (c, ht) in enumerate([(BLUE, 22), (ORANGE, 46), (GREEN, 13), (PURPLE, 18)]):
        d.rect(1419, 797 - sum([22, 46, 13, 18][:i + 1]), 24, ht, fill=c, stroke="#FFFFFF", width=0.4, rounded=0)
    d.arrow(1457, 758, 1535, 758, color=INK, width=1.6)
    d.rect(1545, 708, 285, 98, "<b>Soft cross-entropy</b><br>prediction vs. tᵥ", fill=ORANGE_LIGHT, stroke=ORANGE, width=1.5, font_size=14)
    d.rect(1620, 650, 140, 34, "precomputed tᵥ", fill="#FFFFFF", stroke=ORANGE, width=1.2, font_size=12)
    d.arrow(1690, 684, 1690, 708, color=ORANGE, width=1.5, dashed=True)

    d.line(1090, 870, 1870, 870, color=LINE, width=1.2, dashed=True)
    d.text(1088, 888, 350, 28, "Inference after training (transductive)", font_size=15, bold=True, align="left", color=PURPLE)
    matrix_icon(d, 1100, 1015, 1, 5, 16, BLUE, "node feature xₜₑₛₜ")
    d.arrow(1185, 1053, 1220, 1053, color=INK, width=1.6)
    neural_net(d, 1235, 1053, 0.75)
    d.text(1230, 958, 165, 28, "trained MLP", font_size=14, bold=True)
    d.arrow(1350, 1053, 1425, 1053, color=INK, width=1.6)
    d.rect(1435, 1008, 40, 90, fill="#FFFFFF", stroke=LINE, width=1.1)
    for i, c in enumerate([BLUE, ORANGE, GREEN, PURPLE]):
        d.rect(1440, 1013 + i * 20, 30, 16, fill=c if i == 1 else "#EDF0F2", stroke="#FFFFFF", width=0.4, rounded=0)
    d.text(1495, 1016, 260, 58, "class probabilities<br><b>no graph access</b>", font_size=14, align="left")
    d.rect(1570, 1125, 285, 42, "<b>X-only inference:</b> no A, no H̄, no neighbors", fill=PURPLE_LIGHT, stroke=PURPLE, width=1.2, font_size=12)

    # Legend and inference callout
    d.rect(10, 1210, 1880, 60, fill="#FAFBFC", stroke=INK, width=1.2, rounded=1)
    legend_items = [(BLUE, "Class 1"), (ORANGE, "Class 2"), (GREEN, "Class 3"), (PURPLE, "Class 4")]
    lx = 35
    for color, label in legend_items:
        d.dot(lx, 1240, color, radius=8)
        d.text(lx + 14, 1225, 72, 30, label, font_size=12, align="left")
        lx += 100
    d.dot(470, 1240, GRAY, radius=8, opacity=48)
    d.text(486, 1225, 112, 30, "unlabeled node", font_size=12, align="left")
    d.ellipse(620, 1231, 18, 18, "", fill="none", stroke=GREEN, width=1.3, dashed=True)
    d.text(644, 1225, 105, 30, "granular ball", font_size=12, align="left")
    d.line(775, 1240, 825, 1240, color=INK, width=1.5, arrow="blockThin")
    d.text(833, 1225, 72, 30, "data flow", font_size=12, align="left")
    d.line(925, 1240, 975, 1240, color=PURPLE, width=1.5, dashed=True, arrow="blockThin")
    d.text(983, 1225, 126, 30, "contrastive alignment", font_size=12, align="left")
    d.ellipse(1130, 1229, 22, 22, "×", fill="#FFFFFF", stroke=RED, width=1.6, font_size=17, font_color=RED)
    d.text(1159, 1225, 135, 30, "excluded self-vote", font_size=12, align="left")
    d.rect(1360, 1223, 490, 34, "Gray nodes are unlabeled (about 80%); colors in Stage 2 are visualization only", fill=GRAY_LIGHT, stroke=LINE, width=1.0, font_size=12)

    return d


if __name__ == "__main__":
    diagram = build()
    diagram.save()
    print(OUT)
