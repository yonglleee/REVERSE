#!/usr/bin/env python3
"""Generate fig_tool_dist.png — grouped bar chart of tool usage across training stages.
Publication style following academic-plotting skill guidelines (NeurIPS, single column).
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Publication defaults ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
})

# --- Ocean Dusk palette (colorblind-safe) ---
# Zoom: slate gray (recedes), ImgSearch: teal, TxtSearch: coral (our contribution emphasis)
COLOR_ZOOM   = "#B0BEC5"   # cool gray — less important tool
COLOR_IMG    = "#2A9D8F"   # teal
COLOR_TXT    = "#E76F51"   # coral — most prominent in REVERSE

# Data
stages = ["Base", "Cold Start", "REVERSE"]
zoom = np.array([1.53, 0.20, 0.07])
img  = np.array([1.31, 1.04, 1.56])
txt  = np.array([1.07, 1.45, 2.75])

x = np.arange(len(stages))
width = 0.22

fig, ax = plt.subplots(figsize=(3.25, 2.5))

bars1 = ax.bar(x - width, zoom, width, label='Zoom',         color=COLOR_ZOOM, edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x,          img,  width, label='Image Search', color=COLOR_IMG,  edgecolor='white', linewidth=0.5)
bars3 = ax.bar(x + width,  txt,  width, label='Text Search',  color=COLOR_TXT,  edgecolor='white', linewidth=0.5)

# Value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.08:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=7, color='#444')

ax.set_ylabel('Avg. calls per sample')
ax.set_xticks(x)
ax.set_xticklabels(stages)
ax.set_ylim(0, 3.5)
ax.legend(loc='upper left', ncol=1)

# Highlight REVERSE column with subtle background
ax.axvspan(1.5, 2.5, color='#FFF3EE', alpha=0.5, zorder=0)

plt.tight_layout()
out = '/mnt/sh/mmvision/home/jonahli/projects/tusou/overleaf/Figure/fig_tool_dist.png'
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=300)
# Also save PDF for LaTeX
plt.savefig(out.replace('.png', '.pdf'))
print(f'Saved: {out}')
