"""Compact submission schematic for GB-LCSS.

Core claim: training-time graph structure is converted into leave-one-out
local-consensus supervision, enabling X-only inference.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle


WIDTH_MM = 183
HEIGHT_MM = 94
MM_PER_INCH = 25.4

BLUE = "#3B6EA8"
BLUE_BG = "#EEF4FA"
GREEN = "#4C956C"
GREEN_BG = "#EEF7F0"
AMBER = "#D99A00"
AMBER_BG = "#FFF7E5"
PURPLE = "#7656A3"
PURPLE_BG = "#F4F0F8"
DARK = "#3F4650"
MID = "#7B858F"
LIGHT = "#D8DDE2"
RED = "#C44E52"
WHITE = "#FFFFFF"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "lines.solid_capstyle": "round",
        }
    )


def arrow(ax, x0, y0, x1, y1, color=DARK, lw=0.85, ms=7, style="-") -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            linestyle=style,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=12,
        )
    )


def stage(ax, x0, x1, number, title, color, background, title_size=8.1) -> None:
    ax.add_patch(Rectangle((x0, 0.08), x1 - x0, 0.84, facecolor=background, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((x0, 0.875), x1 - x0, 0.045, facecolor=color, edgecolor="none", zorder=2))
    ax.text(x0 + 0.014, 0.897, str(number), ha="center", va="center", color=WHITE, fontsize=7.8, fontweight="bold")
    ax.text(x0 + 0.029, 0.897, title, ha="left", va="center", color=WHITE, fontsize=title_size, fontweight="bold")


def node(ax, x, y, cls=0, size=20, noisy=False, target=False, alpha=1.0) -> None:
    colors = [BLUE, AMBER, PURPLE]
    markers = ["o", "s", "^"]
    ax.scatter([x], [y], s=size, marker=markers[cls % 3], c=[colors[cls % 3]], edgecolors=WHITE, linewidths=0.55, alpha=alpha, zorder=8)
    if noisy:
        ax.plot([x - 0.006, x + 0.006], [y - 0.009, y + 0.009], color=RED, lw=1.2, zorder=10)
        ax.plot([x - 0.006, x + 0.006], [y + 0.009, y - 0.009], color=RED, lw=1.2, zorder=10)
    if target:
        ax.add_patch(Circle((x, y), 0.012, fill=False, edgecolor=RED, lw=1.0, zorder=9))


def graph(ax, cx, cy, scale=1.0, faded=False) -> None:
    pts = np.array([[-.75,.05],[-.48,.55],[-.05,.40],[.47,.61],[.74,.04],[.25,-.42],[-.30,-.57],[-.68,-.30]])
    edges = [(0,1),(0,2),(0,7),(1,2),(2,3),(2,5),(2,6),(3,4),(3,5),(4,5),(5,6),(6,7)]
    a = 0.35 if faded else 1.0
    for i, j in edges:
        ax.plot([cx + pts[i,0]*0.035*scale, cx + pts[j,0]*0.035*scale], [cy + pts[i,1]*0.070*scale, cy + pts[j,1]*0.070*scale], color=MID, lw=0.55, alpha=a, zorder=3)
    for i, (dx, dy) in enumerate(pts):
        ax.add_patch(Circle((cx + dx*0.035*scale, cy + dy*0.070*scale), 0.0045*scale, facecolor=WHITE if i % 3 else BLUE_BG, edgecolor=BLUE, lw=0.55, alpha=a, zorder=4))


def bars(ax, x, y, values, colors, width=0.045, height=0.075, labels=False) -> None:
    gap = width / (len(values) * 3.2)
    bw = (width - gap * (len(values) - 1)) / len(values)
    for i, (value, color) in enumerate(zip(values, colors)):
        bx = x + i * (bw + gap)
        ax.add_patch(Rectangle((bx, y), bw, height * value, facecolor=color, edgecolor=DARK, lw=0.35, zorder=8))
        if labels:
            ax.text(bx + bw / 2, y - 0.012, str(i + 1), ha="center", va="top", color=MID, fontsize=5.7)
    ax.plot([x - 0.003, x + width + 0.003], [y, y], color=DARK, lw=0.45, zorder=7)


def prob_strip(ax, x, y, values, width=0.065, height=0.022) -> None:
    colors = [BLUE, AMBER, PURPLE]
    cur = x
    for v, c in zip(values, colors):
        w = width * v
        ax.add_patch(Rectangle((cur, y), w, height, facecolor=c, edgecolor=WHITE, lw=0.35, zorder=8))
        cur += w
    ax.add_patch(Rectangle((x, y), width, height, facecolor="none", edgecolor=DARK, lw=0.5, zorder=9))


def draw_stage1(ax, x0, x1) -> None:
    stage(ax, x0, x1, 1, "Label-free GRACE", BLUE, BLUE_BG)
    graph(ax, x0 + 0.035, 0.655, 1.05)
    ax.text(x0 + 0.035, 0.588, r"$G=(V,E,X)$", ha="center", va="top", color=DARK, fontsize=6.5)

    view_x = x0 + 0.091
    graph(ax, view_x, 0.705, 0.70)
    graph(ax, view_x, 0.602, 0.70)
    ax.text(view_x, 0.753, "edge drop", ha="center", va="bottom", color=MID, fontsize=5.9)
    ax.text(view_x, 0.551, "feature mask", ha="center", va="top", color=MID, fontsize=5.9)
    arrow(ax, x0 + 0.066, 0.661, view_x - 0.028, 0.690, BLUE, ms=6)
    arrow(ax, x0 + 0.066, 0.649, view_x - 0.028, 0.617, BLUE, ms=6)

    enc_x = x0 + 0.133
    ax.add_patch(Polygon([(enc_x - 0.016, 0.574), (enc_x - 0.016, 0.726), (enc_x + 0.021, 0.691), (enc_x + 0.021, 0.609)], closed=True, facecolor=WHITE, edgecolor=BLUE, lw=0.8, zorder=4))
    ax.text(enc_x + 0.001, 0.650, "GRACE", ha="center", va="center", color=BLUE, fontsize=6.4, fontweight="bold")
    arrow(ax, view_x + 0.027, 0.650, enc_x - 0.019, 0.650, BLUE, ms=6)

    hx = x1 - 0.024
    arrow(ax, enc_x + 0.024, 0.650, hx - 0.013, 0.650, BLUE, ms=6)
    for i in range(5):
        ax.add_patch(Rectangle((hx - 0.009, 0.606 + i*0.019), 0.018, 0.011, facecolor=BLUE if i % 2 else WHITE, edgecolor=BLUE, lw=0.45, zorder=5))
    ax.text(hx, 0.726, r"frozen $\bar{H}$", ha="center", va="bottom", color=DARK, fontsize=6.3)

    ax.text((x0+x1)/2, 0.460, "noisy labels", ha="center", va="center", color=RED, fontsize=6.4, fontweight="bold")
    for i, xx in enumerate(np.linspace(x0 + 0.050, x1 - 0.050, 4)):
        node(ax, xx, 0.405, i % 3, 22, noisy=i in (1,3))
    ax.plot([x0 + 0.034, x1 - 0.034], [0.360, 0.360], color=RED, lw=0.65, linestyle=(0,(2,2)))
    ax.text((x0+x1)/2, 0.329, "labels not used", ha="center", va="center", color=RED, fontsize=6.2)
    ax.text((x0+x1)/2, 0.160, r"only $X$ and $A$ enter GRACE", ha="center", va="center", color=BLUE, fontsize=6.2, fontweight="bold")


def draw_stage2(ax, x0, x1) -> None:
    stage(ax, x0, x1, 2, "Granular-ball partition", GREEN, GREEN_BG, title_size=7.7)
    rng = np.random.default_rng(8)
    pcx, pcy = x0 + 0.041, 0.650
    pts = rng.normal(size=(19,2))
    ax.add_patch(Ellipse((pcx,pcy), 0.070,0.205, facecolor=WHITE, edgecolor=GREEN, lw=0.8))
    ax.scatter(pcx + pts[:,0]*0.015, pcy + pts[:,1]*0.043, s=8, c=GREEN, edgecolors=WHITE, linewidths=0.3, zorder=6)
    ax.text(pcx, 0.767, r"one ball in $\bar{H}$", ha="center", va="bottom", color=DARK, fontsize=6.2)

    sx = x0 + 0.091
    arrow(ax, pcx + 0.038, pcy, sx - 0.018, pcy, GREEN, ms=6)
    ax.text(sx + 0.003, 0.724, "2-means", ha="center", va="bottom", color=GREEN, fontsize=5.9, fontweight="bold")
    arrow(ax, sx + 0.018, pcy, x0 + 0.124, pcy, GREEN, ms=6)

    final_x = x0 + 0.139
    centers = [(final_x,0.708),(final_x+0.034,0.708),(final_x,0.584),(final_x+0.034,0.584)]
    for k,(cx,cy) in enumerate(centers):
        edge = [BLUE,GREEN,AMBER,PURPLE][k]
        ax.add_patch(Ellipse((cx,cy),0.027,0.082,facecolor=WHITE,edgecolor=edge,lw=0.75))
        local = rng.normal(size=(5,2))
        ax.scatter(cx + local[:,0]*0.005, cy + local[:,1]*0.016, s=5.5, c=edge, edgecolors="none", zorder=6)
    ax.text(final_x + 0.017, 0.767, "disjoint balls", ha="center", va="bottom", color=DARK, fontsize=6.0)

    ax.plot([x0 + 0.034, x1 - 0.034], [0.425,0.425], color=GREEN, lw=0.65)
    ax.text((x0+x1)/2, 0.365, "geometry only", ha="center", va="center", color=GREEN, fontsize=6.5, fontweight="bold")
    ax.text((x0+x1)/2, 0.308, r"$B_{\max}=64$", ha="center", va="center", color=DARK, fontsize=6.2)
    ax.text((x0+x1)/2, 0.264, r"$n_{\min}=32$", ha="center", va="center", color=DARK, fontsize=6.2)
    ax.text((x0+x1)/2, 0.220, r"$\delta=0.05$", ha="center", va="center", color=DARK, fontsize=6.2)
    ax.text((x0+x1)/2, 0.160, "no labels", ha="center", va="center", color=GREEN, fontsize=6.2, fontweight="bold")


def draw_stage3(ax, x0, x1) -> None:
    stage(ax, x0, x1, 3, "Leave-one-out local consensus", AMBER, AMBER_BG)
    bx, by = x0 + 0.050, 0.665
    ax.add_patch(Ellipse((bx,by),0.090,0.205,facecolor=WHITE,edgecolor=AMBER,lw=0.9))
    data = [(-.026,.059,0),(-.004,.076,0),(.022,.051,0),(-.029,.009,0),(.005,.012,0),(.028,-.026,0),(-.018,-.058,1),(.015,-.069,2)]
    for dx,dy,cls in data:
        node(ax,bx+dx,by+dy,cls,20,target=(dx,dy)==(-.018,-.058))
    tvx,tvy=bx-.018,by-.058
    ax.plot([tvx-.006,tvx+.006],[tvy-.009,tvy+.009],color=RED,lw=1.3,zorder=11)
    ax.plot([tvx-.006,tvx+.006],[tvy+.009,tvy-.009],color=RED,lw=1.3,zorder=11)
    ax.text(tvx,by-.118,"exclude self",ha="center",va="top",color=RED,fontsize=6.1,fontweight="bold")

    vx=x0+0.118
    arrow(ax,bx+0.050,by,vx-0.010,by,AMBER,ms=6)
    bars(ax,vx,0.618,[.68,.21,.11],[BLUE,AMBER,PURPLE],width=.050,height=.105,labels=True)
    ax.text(vx+.025,0.752,r"$\mathbf{p}_{j,-v}$",ha="center",va="bottom",color=DARK,fontsize=6.4,fontweight="bold")

    rx=x0+0.218
    arrow(ax,vx+0.055,by,rx-0.042,by,AMBER,ms=6)
    ax.text(rx,0.745,"unique majority",ha="center",va="bottom",color=DARK,fontsize=5.8,fontweight="bold")
    ax.text(rx,0.692,r"$\mathbf{q}_{j,-v}=(1-\beta)\mathbf{p}_{j,-v}+\beta\mathbf{e}_m$",ha="center",va="center",color=DARK,fontsize=5.6)
    ax.text(rx,0.615,"tie",ha="center",va="bottom",color=DARK,fontsize=5.8,fontweight="bold")
    ax.text(rx,0.576,r"$\mathbf{q}_{j,-v}=\mathbf{p}_{j,-v}$",ha="center",va="center",color=DARK,fontsize=5.9)
    ax.text(rx,0.536,"tie-safe",ha="center",va="center",color=AMBER,fontsize=5.9,fontweight="bold")

    ax.plot([x0+.026,x1-.026],[0.475,0.475],color=AMBER,lw=.65)
    ax.text((x0+x1)/2,0.422,r"$\alpha_{j,-v}=\max\!\left(0,\frac{r_{j,-v}-1/C}{1-1/C}\right)$",ha="center",va="center",color=DARK,fontsize=6.2)
    ax.text((x0+x1)/2,0.343,r"$\mathbf{t}_v=(1-\alpha_{j,-v})\mathbf{e}_{\tilde{y}_v}+\alpha_{j,-v}\mathbf{q}_{j,-v}$",ha="center",va="center",color=DARK,fontsize=6.3,fontweight="bold")

    prob_strip(ax,x0+.052,0.230,[0,1,0],width=.064,height=.024)
    arrow(ax,x0+.124,0.242,x0+.165,0.242,AMBER,ms=6)
    prob_strip(ax,x0+.173,0.230,[.63,.27,.10],width=.064,height=.024)
    ax.text(x0+.084,0.273,"observed label",ha="center",va="bottom",color=MID,fontsize=5.8)
    ax.text(x0+.205,0.273,"soft target",ha="center",va="bottom",color=MID,fontsize=5.8)
    ax.text((x0+x1)/2,0.184,"soft correction",ha="center",va="center",color=AMBER,fontsize=6.4,fontweight="bold")


def feature(ax, x, y, label) -> None:
    ax.add_patch(Rectangle((x-.010,y-.036),.020,.072,facecolor=WHITE,edgecolor=PURPLE,lw=.65,zorder=7))
    for k,w in enumerate([.012,.007,.015,.009]):
        ax.plot([x-.007,x-.007+w],[y-.024+k*.016,y-.024+k*.016],color=PURPLE,lw=.8,zorder=8)
    ax.text(x,y+.052,label,ha="center",va="bottom",color=DARK,fontsize=6.1,fontweight="bold")


def mlp(ax, x, y, width=.058, height=.080, label="MLP") -> None:
    ax.add_patch(Rectangle((x,y),width,height,facecolor=WHITE,edgecolor=PURPLE,lw=.75,zorder=6))
    for frac in (.34,.67):
        ax.plot([x+width*frac,x+width*frac],[y+.010,y+height-.010],color=PURPLE,lw=.6,zorder=7)
    ax.text(x+width/2,y+height/2,label,ha="center",va="center",color=DARK,fontsize=6.3,fontweight="bold")


def draw_stage4(ax, x0, x1) -> None:
    stage(ax,x0,x1,4,"X-only student",PURPLE,PURPLE_BG)
    ax.text(x0+.018,0.792,"TRAINING",ha="left",va="center",color=PURPLE,fontsize=6.2,fontweight="bold")
    fy=.665
    feature(ax,x0+.034,fy,r"$x_v$")
    mx=x0+.071
    arrow(ax,x0+.046,fy,mx-.005,fy,PURPLE,ms=6)
    mlp(ax,mx,fy-.040,label="MLP")
    ax.text(mx+.029,fy+.054,"2-layer",ha="center",va="bottom",color=MID,fontsize=5.8)
    px=x0+.160
    arrow(ax,mx+.063,fy,px-.009,fy,PURPLE,ms=6)
    prob_strip(ax,px,fy-.012,[.58,.28,.14],width=.055,height=.024)

    lossx=x0+.111
    ax.add_patch(Rectangle((lossx,0.500),.086,.060,facecolor=WHITE,edgecolor=AMBER,lw=.75,zorder=6))
    ax.text(lossx+.043,.530,"soft CE loss",ha="center",va="center",color=DARK,fontsize=6.0,fontweight="bold",zorder=9)
    ax.text(x0+.028,.530,r"$\mathbf{t}_v$",ha="center",va="center",color=AMBER,fontsize=6.7,fontweight="bold")
    arrow(ax,x0+.040,.530,lossx-.006,.530,AMBER,ms=6)
    arrow(ax,px+.025,fy-.022,lossx+.065,.564,PURPLE,ms=6)
    ax.text(x0+.064,0.478,"supervision only",ha="center",va="top",color=AMBER,fontsize=5.9)

    ax.plot([x0+.016,x1-.016],[0.420,0.420],color=DARK,lw=.75,linestyle=(0,(3,2)))
    ax.text((x0+x1)/2,0.438,"DEPLOYMENT",ha="center",va="bottom",color=DARK,fontsize=6.2,fontweight="bold")
    ax.text((x0+x1)/2,0.390,r"discard $A$, $\bar{H}$, GRACE and balls",ha="center",va="center",color=MID,fontsize=5.9)

    iy=.248
    feature(ax,x0+.034,iy,r"$x_v$")
    imx=x0+.071
    arrow(ax,x0+.046,iy,imx-.005,iy,PURPLE,ms=6)
    mlp(ax,imx,iy-.040,label="MLP")
    ax.text(imx+.029,iy+.054,"same model",ha="center",va="bottom",color=MID,fontsize=5.8)
    arrow(ax,imx+.063,iy,x0+.160,iy,PURPLE,ms=6)
    prob_strip(ax,x0+.165,iy-.012,[.67,.20,.13],width=.055,height=.024)
    ax.text(x0+.192,0.293,"class probabilities",ha="center",va="bottom",color=DARK,fontsize=5.9)
    ax.text((x0+x1)/2,0.135,r"$X$ only · no $A$ · no $\bar{H}$ · no neighbors",ha="center",va="center",color=PURPLE,fontsize=6.1,fontweight="bold")


def make_figure():
    setup_style()
    fig = plt.figure(figsize=(WIDTH_MM/MM_PER_INCH,HEIGHT_MM/MM_PER_INCH),dpi=150)
    ax=fig.add_axes([0,0,1,1])
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    s1=(.010,.200); s2=(.210,.405); s3=(.415,.705); s4=(.715,.990)
    draw_stage1(ax,*s1); draw_stage2(ax,*s2); draw_stage3(ax,*s3); draw_stage4(ax,*s4)
    arrow(ax,s1[1]+.002,.842,s2[0]-.003,.842,BLUE,lw=1.0,ms=8)
    arrow(ax,s2[1]+.002,.842,s3[0]-.003,.842,GREEN,lw=1.0,ms=8)
    arrow(ax,s3[1]+.002,.842,s4[0]-.003,.842,AMBER,lw=1.0,ms=8)
    return fig


def main() -> None:
    out=Path(__file__).with_name("fig1_gblcss_overview_v3.pdf")
    fig=make_figure()
    fig.savefig(out,format="pdf",bbox_inches=None,pad_inches=0,facecolor=WHITE)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
