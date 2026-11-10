"""Visualize the SVD result: coverages in PC space + the leading component maps."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from . import decompose as D
from .visualize import SOFTSPOT_CMAP, _draw_field, plt as _styled_plt  # reuse dark style

MAN = D.MAN


def plot(out="outputs/svd_overview.png"):
    df = pd.read_parquet("outputs/route_separations_throw.parquet")
    covs, coords, comps, edges = D.run()
    xe, ye = edges

    fig = plt.figure(figsize=(16, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1])

    # (1) coverages in PC1-PC2 space
    ax0 = fig.add_subplot(gs[0, 0])
    for i, c in enumerate(covs):
        is_man = c in MAN
        ax0.scatter(coords[i, 0], coords[i, 1], s=160,
                    c="#ff6b6b" if is_man else "#2ecc40",
                    ec="white", zorder=5)
        ax0.annotate(c, (coords[i, 0], coords[i, 1]),
                     textcoords="offset points", xytext=(8, 6),
                     color="#e6edf3", fontsize=9)
    ax0.axhline(0, color="#30363d", lw=0.8)
    ax0.axvline(0, color="#30363d", lw=0.8)
    ax0.set_xlabel("PC1  (man  <-->  zone;  75% var)")
    ax0.set_ylabel("PC2  (zone-family shape; 17% var)")
    ax0.set_title("Coverages in separation-surface PC space\n"
                  "(unsupervised: labels never given)", color="#e6edf3", fontsize=11)

    # (2) PC1 loading map, (3) PC2 loading map
    for j, axpos in enumerate([gs[0, 1], gs[0, 2]]):
        ax = fig.add_subplot(axpos)
        load = comps[j]
        lim = np.nanpercentile(np.abs(load[np.isfinite(load)]), 96)
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
        im = ax.pcolormesh(xe, ye, load, cmap=SOFTSPOT_CMAP, norm=norm, shading="flat")
        _draw_field(ax, xe, ye)
        ax.set_title(f"PC{j+1} spatial loading", color="#e6edf3", fontsize=11)
        ax.set_xlabel("yards downfield of LOS")
        if j == 0:
            ax.set_ylabel("lateral from ball (yd)")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("SVD of coverage separation surfaces  —  structure emerges from geometry, not a model",
                 color="#e6edf3", fontsize=14, y=1.04)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(plot())
