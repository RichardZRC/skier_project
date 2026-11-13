"""Milestone Figure 1: block diagram of the per-frame pipeline. Writes a vector PDF for LaTeX and a PNG preview.

Usage (any matplotlib): python milestone/figures/pipeline_fig.py milestone/figures/pipeline.pdf preview.png"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out_pdf, out_png = Path(sys.argv[1]), Path(sys.argv[2])
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42})

W, H = 5.5, 0.77           # inches: the full NeurIPS text width
Y0, BH = 0.03, 0.47        # box bottom and height
GREY = "#555555"
# title, detail, width, face, edge, line width. Colors match the strata figure: tracker orange, refinement brown.
BOXES = [
    ("Video frame", "skier box given\nin frame 1 only", 0.68, "white", GREY, 0.8),
    ("SiamFC tracker", "fixed, pretrained;\nbox + confidence", 0.84, "#fde3cc", "#d9822b", 0.8),
    ("Crop + HRNet-W32", "region around box,\n256$\\times$192; fine-tuned", 1.00, "#dbe8f5", "#3b6ea5", 0.8),
    ("Refinement (ours)", "re-crop around pass-1\nlandmarks; HRNet again", 1.00, "#ecd9c6", "#8c510a", 1.6),
    ("24 landmarks", "14 body, 8 ski,\n2 pole", 0.72, "white", GREY, 0.8),
    ("Replay", "smoothed window,\nskeleton drawn", 0.76, "#e3efdc", "#4f7f3a", 0.8),
]

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

margin = 0.015
gap = (W - 2 * margin - sum(b[2] for b in BOXES)) / (len(BOXES) - 1)
x, spans = margin, []
for title, detail, w, fc, ec, lw in BOXES:
    ax.add_patch(FancyBboxPatch((x, Y0), w, BH, boxstyle="round,pad=0,rounding_size=0.05", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, Y0 + BH - 0.105, title, ha="center", va="center", fontsize=7.5, fontweight="bold")
    ax.text(x + w / 2, Y0 + 0.165, detail, ha="center", va="center", fontsize=6.5, linespacing=1.15)
    spans.append((x, x + w))
    x += w + gap

arrow = dict(arrowstyle="-|>", color=GREY, lw=0.8, mutation_scale=6, shrinkA=0, shrinkB=0)
ymid = Y0 + BH / 2
for (_, a_end), (b_start, _) in zip(spans, spans[1:]):
    ax.annotate("", xy=(b_start, ymid), xytext=(a_end, ymid), arrowprops=arrow)

# The tracker box and its confidence also drive the replay window.
top, ytop = Y0 + BH, 0.61
c2, c6 = sum(spans[1]) / 2, sum(spans[5]) / 2
ax.plot([c2, c2, c6], [top, ytop, ytop], color=GREY, lw=0.8, solid_capstyle="butt")
ax.annotate("", xy=(c6, top), xytext=(c6, ytop + 0.004), arrowprops=arrow)
ax.text((c2 + c6) / 2, ytop + 0.02, "tracker box and confidence set the replay window (it zooms out when the tracker is unsure)",
        ha="center", va="bottom", fontsize=6.5)

fig.savefig(out_pdf)
fig.savefig(out_png, dpi=300)
print(out_pdf, out_png)
