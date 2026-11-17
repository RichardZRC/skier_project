"""Final report Figure 1(b): the pose model, HRNet-W32 with a new 24-landmark output layer, and what was trained when.

Facts (MMPose config td-hm_hrnet-w32_8xb64-210e_coco-256x192 and skier.pose.model / skier.pose.train):
- stem: two 3x3 stride-2 convs (input 256x192 -> 1/4 resolution, 64x48);
- stage 1: 4 bottleneck blocks, 64 channels, 1/4 resolution;
- stages 2-4: 1 / 4 / 3 modules, adding branches at 1/8, 1/16, 1/32 (32 / 64 / 128 / 256 channels); each module ends
  with a multi-resolution fusion; the output is the 1/4-resolution branch (32 channels);
- head: one 1x1 conv, 32 -> 24 channels (792 parameters), new and randomly initialized (COCO's had 17 outputs);
- backbone: 28.5 M parameters from the COCO checkpoint, frozen (incl. BatchNorm statistics) for the 3 warm-up epochs,
  then fine-tuned at lr 1e-4; the head trains from the start (lr 1e-3, then 5e-4).

Usage (any matplotlib): python final_report/figures/hrnet_fig.py final_report/figures/hrnet.pdf preview.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

out_pdf, out_png = Path(sys.argv[1]), Path(sys.argv[2])
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42})

W, H = 5.5, 1.2                         # inches: the full NeurIPS text width, like the pipeline strip
GREY = "#555555"
PRE_FC, PRE_EC = "#dbe8f5", "#3b6ea5"   # COCO-pretrained (the pipeline figure's HRNet blue)
NEW_FC, NEW_EC = "#fff0b3", "#b8860b"   # new layer
BOX_Y, BOX_H = 0.06, 0.74               # boxes span the lane area
LANES = [  # center y, label, first module that has the branch (1-based over the 8 modules of stages 2-4)
    (0.70, "1/4, 32 ch", 1),
    (0.505, "1/8, 64 ch", 1),
    (0.31, "1/16, 128 ch", 2),
    (0.115, "1/32, 256 ch", 6),
]
LANE_H = 0.115
MODULES = [2, 3, 3, 3, 3, 4, 4, 4]      # stage of each module: stage 2 has 1, stage 3 has 4, stage 4 has 3

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
arrow = dict(arrowstyle="-|>", color=GREY, lw=0.8, mutation_scale=6, shrinkA=0, shrinkB=0)


def box(x, w, title, detail, fc, ec, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, BOX_Y), w, BOX_H, boxstyle="round,pad=0,rounding_size=0.05", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, BOX_Y + BOX_H - 0.12, title, ha="center", va="center", fontsize=7.5, fontweight="bold")
    ax.text(x + w / 2, BOX_Y + BOX_H / 2 - 0.1, detail, ha="center", va="center", fontsize=6.5, linespacing=1.15)
    return x, x + w


# --- boxes before the multi-resolution stages -------------------------------------------------
x_in = box(0.02, 0.56, "Input crop", "256$\\times$192\nRGB", "white", GREY)
x_stem = box(0.70, 0.58, "Stem", "2 conv 3$\\times$3,\nstride 2\n(to 1/4)", PRE_FC, PRE_EC)
x_s1 = box(1.40, 0.56, "Stage 1", "4 bottleneck\nblocks,\n64 ch, 1/4", PRE_FC, PRE_EC)
y_mid = BOX_Y + BOX_H / 2
for a, b in ((x_in, x_stem), (x_stem, x_s1)):
    ax.annotate("", xy=(b[0], y_mid), xytext=(a[1], y_mid), arrowprops=arrow)

# --- stages 2-4: parallel branches with repeated fusion ---------------------------------------
L0, L1 = 2.11, 4.24
mw = (L1 - L0) / len(MODULES)
for y, label, first in LANES:
    x0 = L0 + (first - 1) * mw
    ax.add_patch(Rectangle((x0, y - LANE_H / 2), L1 - x0, LANE_H, fc=PRE_FC, ec=PRE_EC, lw=0.8))
    # the label sits on a lane-colored patch drawn above the fusion lines, so they don't cross it
    ax.text(x0 + 0.04, y, label, ha="left", va="center", fontsize=6, zorder=5,
            bbox=dict(boxstyle="square,pad=0.15", fc=PRE_FC, ec="none"))
# the transition from stage 1 into the first two branches
for y, _, first in LANES[:2]:
    ax.annotate("", xy=(L0, y), xytext=(x_s1[1], y_mid), arrowprops=arrow)
# a fusion after every module: a vertical exchange across the branches that exist
for k in range(1, len(MODULES) + 1):
    xf = L0 + k * mw - 0.035
    live = [y for y, _, first in LANES if first <= k]
    ax.plot([xf, xf], [min(live), max(live)], color=PRE_EC, lw=1.0, solid_capstyle="round")
    ax.plot([xf] * len(live), live, "o", ms=2.2, color=PRE_EC)
# stage labels above the lanes
for stage, name in ((2, "Stage 2"), (3, "Stage 3 ($\\times$4)"), (4, "Stage 4 ($\\times$3)")):
    ks = [i for i, s in enumerate(MODULES) if s == stage]
    xa, xb = L0 + ks[0] * mw, L0 + (ks[-1] + 1) * mw
    ax.text((xa + xb) / 2, 0.79, name, ha="center", va="bottom", fontsize=6.5)
    if stage > 2:
        ax.plot([xa, xa], [0.03, 0.83], color=GREY, lw=0.5, ls=(0, (2, 2)))

# --- the new head and the output ---------------------------------------------------------------
x_head = box(4.35, 0.50, "1$\\times$1 conv", "32 $\\to$ 24 ch\nnew, random\ninit.", NEW_FC, NEW_EC, lw=1.4)
x_out = box(4.95, 0.53, "Output", "24 heatmaps,\n64$\\times$48;\npeaks $\\to$\nlandmarks", "white", GREY)
ax.annotate("", xy=(x_head[0], LANES[0][0]), xytext=(L1, LANES[0][0]), arrowprops=arrow)
ax.annotate("", xy=(x_out[0], y_mid), xytext=(x_head[1], y_mid), arrowprops=arrow)

# --- what was trained when ----------------------------------------------------------------------
yb = 0.94
ax.plot([x_stem[0], x_stem[0], L1, L1], [yb - 0.03, yb, yb, yb - 0.03], color=PRE_EC, lw=0.8)
ax.text((x_stem[0] + L1) / 2, yb + 0.012, "HRNet-W32 backbone", ha="center", va="bottom", fontsize=6, color=PRE_EC)
# one legend line: each item's text is measured so the next item starts after it
renderer = fig.canvas.get_renderer()
x, yl = 0.05, 1.12
for fc, ec, text in ((PRE_FC, PRE_EC, "COCO-pretrained (28.5 M params): frozen for 3 warm-up epochs, then fine-tuned"),
                     (NEW_FC, NEW_EC, "new (792 params): trained throughout")):
    ax.add_patch(Rectangle((x, yl - 0.04), 0.13, 0.08, fc=fc, ec=ec, lw=0.9))
    t = ax.text(x + 0.17, yl, text, ha="left", va="center", fontsize=6.5)
    bb = t.get_window_extent(renderer).transformed(ax.transData.inverted())
    x = bb.x1 + 0.2

fig.savefig(out_pdf)
fig.savefig(out_png, dpi=300)
print(out_pdf, out_png)
