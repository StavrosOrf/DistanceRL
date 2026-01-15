#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot T_ij = 1 - 2 (Δ_ij)^λ for multiple λ values on the same figure.
Paper-ready styling + export to PDF/PNG.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

def T(delta: np.ndarray, lam: float) -> np.ndarray:
    return 1.0 - 2.0 * np.power(delta, lam)

def main() -> None:
    # Domain: Δ_ij ∈ [0, 1] ensures T_ij ∈ [-1, 1]
    delta = np.linspace(0.0, 1.0, 1200)

    lambdas = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
    palette = sns.color_palette("muted", n_colors=len(lambdas))

    # --- Paper-ready matplotlib style ---
    plt.rcParams.update({
        "figure.figsize": (5, 2.5),
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
        "grid.alpha": 0.35,
    })

    fig, ax = plt.subplots(constrained_layout=True)

    # A readable set of line styles (works in grayscale)
    linestyles = ["-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1))]

    for lam, ls, color in zip(lambdas, linestyles, palette):
        ax.plot(
            delta,
            T(delta, lam),
            color=color,
            linestyle=ls,
            label=rf"$\lambda={lam:g}$",
        )

    # Reference guides (major only, light gray)
    ax.axhline(0.0, color="0.82", lw=0.9, zorder=0)
    ax.axvline(0.0, color="0.82", lw=0.9, zorder=0)
    ax.axvline(1.0, color="0.82", lw=0.9, zorder=0)

    # Labels and title
    ax.set_xlabel(r"Normalized Value Gap $\Delta_{i,j}$")
    ax.set_ylabel(r"Target Cos. Sim. $Y_{i,j}$")

    # Limits, ticks, and grid
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(-1.05, 1.05)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="major", color="0.82", linewidth=0.9)
    ax.grid(False, which="minor")

    # Legend
    leg = ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        ncol=2,
        columnspacing=0.7,
        framealpha=0.85,
        borderpad=0.6,
        handlelength=1.3,
    )
    leg.get_frame().set_edgecolor("0.75")
    leg.get_frame().set_linewidth(0.9)
    
    #remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    # make arrow heads
    # ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    # ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    # Small annotation: endpoints
    ax.scatter([0.0, 1.0], [1.0, -1.0], s=18, color="0.15", zorder=5)

    # Export (vector + raster)
    fig.savefig("cos_sim_curves.pdf")
    fig.savefig("cos_sim_curves.png")

    # plt.show()
    # plt.savefigures("Tij_vs_Delta_lambdas.pdf")

if __name__ == "__main__":
    main()
