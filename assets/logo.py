"""Draw the circuitGNN wordmark with a blue -> red gradient.

Writes logo.svg and logo.png (transparent background) next to this script.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

OUT = Path(__file__).parent
# blue and red hold for most of the word; the blend is confined to a short stretch at "t|G"
CMAP = LinearSegmentedColormap.from_list(
    "logo", [(0, "#3B82F6"), (0.5, "#2563EB"), (0.58, "#2563EB"), (0.64, "#DC2626"), (1, "#EF4444")]
)

fig, ax = plt.subplots(figsize=(6, 1.4))
ax.set_aspect("equal")
ax.axis("off")

text = TextPath((0, 0), "CircuitGNN", size=1.5, prop=dict(family="DejaVu Sans", weight="bold"))
box = text.get_extents()
patch = PathPatch(text, transform=Affine2D().translate(-box.x0, -box.y0) + ax.transData, lw=0, fc="none")
ax.add_patch(patch)
grad = ax.imshow(
    np.linspace(0, 1, 256)[None, :], cmap=CMAP, extent=(0, box.width, 0, box.height), aspect="equal"
)
grad.set_clip_path(patch)

ax.set_xlim(0, box.width)
ax.set_ylim(0, box.height)

fig.savefig(OUT / "logo.svg", transparent=True, bbox_inches="tight", pad_inches=0.1)
fig.savefig(OUT / "logo.png", transparent=True, bbox_inches="tight", pad_inches=0.1, dpi=300)
