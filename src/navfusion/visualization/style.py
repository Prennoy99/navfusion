"""One colour per entity, shared by every figure, so a method looks the same in every plot.

The method colours are the first three slots of a colour-blind-checked categorical palette
(checked for protan/deutan/tritan separation and >= 3:1 contrast on white). Ground truth
and GPS are reference data, not methods, so they get neutral ink tones.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

TRUTH = "#0b0b0b"
GPS = "#8a8985"
EKF = "#2a78d6"
DEAD_RECKONING = "#eb6834"
EKF_LSTM = "#1baf7a"

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
OUTAGE_SHADE = "#f0efec"


def style_axes(ax: plt.Axes, xlabel: str, ylabel: str, title: str | None = None) -> None:
    """Recessive grid and axes, no box: the data carries the contrast."""
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.set_ylabel(ylabel, color=INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, color=INK, loc="left", fontsize=11)
