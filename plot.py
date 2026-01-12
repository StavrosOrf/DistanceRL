import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
from matplotlib.legend_handler import HandlerTuple

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "legend.fontsize": 13,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.linewidth": 1.0,
    "lines.linewidth": 3.0,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
})

palette = sns.color_palette("muted", 6)
magma_colors = cm.get_cmap("magma")(np.linspace(0.25, 0.85, 3))

rng = np.random.default_rng(2)
# initial anchor action from current policy
a0 = np.array([-1.5, 1.0], dtype=float)
T = 4

# ----------------------------
# Q landscape (only for contour context + Q-coloring of sampled actions)
# ----------------------------
a1 = np.linspace(-2, 2, 1000)
a2 = np.linspace(-2, 2, 1000)
A1, A2 = np.meshgrid(a1, a2)

def Q_landscape(A1, A2):
    q = (
        -0.08 * (A1**2 + A2**2)
        + 0.55 * np.sin(1.4 * A1) * np.cos(1.2 * A2)
        + 0.22 * np.sin(2.3 * A2)
        + 0.16 * np.cos(2.0 * A1 * A2)
        + 1.35 * np.exp(-((A1-1.2)**2 + (A2+0.8)**2) / 0.9)
        + 0.85 * np.exp(-((A1+1.6)**2 + (A2-1.4)**2) / 0.6)
        - 0.45 * np.exp(-((A1-0.4)**2 + (A2-1.9)**2) / 0.45)
    )
    
    # normalize to [0, 1]
    q_min = q.min()
    q_max = q.max()
    q_norm = (q - q_min) / (q_max - q_min + 1e-8)
    return q_norm

Q = Q_landscape(A1, A2)

def Q_at(x, y):
    ix = np.clip(np.searchsorted(a1, x), 0, len(a1) - 1)
    iy = np.clip(np.searchsorted(a2, y), 0, len(a2) - 1)
    return float(Q[iy, ix])

# ----------------------------
# Distance-RL update (neighbors sampled each step)
# ----------------------------
def step_update(a_anch, K=32, sigma=0.9, tau=0.85, beta=1, lam=0.99):    
    A_nb = a_anch + rng.normal(0.0, sigma, size=(K, 2))
    A_nb[:, 0] = np.clip(A_nb[:, 0], a1[0], a1[-1])
    A_nb[:, 1] = np.clip(A_nb[:, 1], a2[0], a2[-1])

    q = np.array([Q_at(p[0], p[1]) for p in A_nb])

    d2 = np.sum((A_nb - a_anch[None, :])**2, axis=1)
    sim = np.exp(-d2 / (2 * tau**2))

    q_norm = (q - q.mean()) / (q.std() + 1e-8)
    # logits = beta * q_norm + (1 - beta) * sim
    logits = beta * q_norm + (1 - beta) * sim
    logits = logits - logits.max()
    w = np.exp(logits)
    w = w / (w.sum() + 1e-12)

    a_w = (w[:, None] * A_nb).sum(axis=0)
    a_new = (1 - lam) * a_anch + lam * a_w
    return a_new, A_nb, q, a_w


anchors = [a0.copy()]

# keep LAST-step neighbors for drawing only
last_neighbors = None
last_q = None
last_weighted = None

a = a0.copy()
for t in range(T):
    a_new, A_nb, q, a_w = step_update(a)

    if t == T - 1:
        last_neighbors = A_nb
        last_q = q
        last_weighted = a_w.copy()

    anchors.append(a_new.copy())
    a = a_new

anchors = np.array(anchors)
a_opt = anchors[-1]

# ----------------------------
# Plot: clean trajectory + connected arrows; show sampled actions ONLY for LAST step
# ----------------------------
fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

contours = ax.contour(
    A1,
    A2,
    Q,
    levels=15,
    linewidths=0.9,
    alpha=0.55,
    cmap="magma",
)

traj_color = palette[0]
arrow_color = "black"  # palette[1]
sample_color = "limegreen"  #palette[5]
start_color = palette[3]

# Connected trajectory (soft line)
ax.plot(
    anchors[:, 0],
    anchors[:, 1],
    linewidth=2.4,
    alpha=0.2,
    zorder=2,
    color=traj_color,
)

# Direction arrows along the trajectory
stride = max(1, T // 22)  # ~20 arrows
x = anchors[:-1:stride, 0]
y = anchors[:-1:stride, 1]
dx = anchors[1::stride, 0] - anchors[:-1:stride, 0]
dy = anchors[1::stride, 1] - anchors[:-1:stride, 1]
# exclude last arrow to avoid overlap with sampled actions
x = x[:-1]
y = y[:-1]
dx = dx[:-1]
dy = dy[:-1]

ax.quiver(
    x,
    y,
    dx,
    dy,
    angles="xy",
    scale_units="xy",
    scale=1,
    width=0.007,
    alpha=0.9,
    headwidth=5,
    zorder=4,
    color=arrow_color,
)

# LAST step sampled actions (colored by Q)
a_last_anchor = anchors[-2]
for p in last_neighbors:
    ax.plot(
        [a_last_anchor[0], p[0]],
        [a_last_anchor[1], p[1]],
        linewidth=0.9,
        alpha=0.25,
        color=sample_color,
    )

sc = ax.scatter(
    last_neighbors[:, 0],
    last_neighbors[:, 1],
    s=28,
    c=last_q,
    alpha=0.9,
    zorder=3,
    cmap="magma",
    edgecolor="white",
    linewidth=0.4,
)

cbar = fig.colorbar(sc, ax=ax, shrink=1.0, pad=0.02)
cbar.set_label(r"$Q(s_{t},a_{t})$", labelpad=6)
cbar.ax.tick_params(labelsize=12)
#set ticks from 0 to 1 with step 0.2
cbar.set_ticks(np.arange(0, 1.01, 0.2))

# add a marker at the starting point
# ax.scatter([a0[0]], [a0[1]], s=26, marker="o", color=start_color, zorder=5, edgecolor="white", linewidth=0.7)

# have yticks every 1 unit
ax.set_yticks(np.arange(a2[0], a2[-1] + 1, 1))
ax.set_xticks(np.arange(a1[0], a1[-1] + 1, 1))
ax.tick_params(axis="both", which="major", labelsize=12)
ax.grid(True, color="0.9", linewidth=0.8, alpha=0.6)

# Weighted direction at the last step
ax.annotate(
    "",
    xy=last_weighted,
    xytext=a_last_anchor,
    arrowprops=dict(
        arrowstyle="-|>",
        linewidth=2.5,
        alpha=1,
        color=sample_color,
        zorder=1,
    ),
)

from matplotlib.lines import Line2D
# ---------- Legend (proxy artists) ----------
legend_handles = [
    Line2D(
        [0],
        [0],
        color=arrow_color,
        lw=2,
        marker=">",
        markersize=8,
        markevery=[1],
        label="Past trajectory",
    ),
    (
     Line2D([0], [0], marker="o", color="w", markerfacecolor=magma_colors[0],
               markeredgecolor="white", markersize=6, linestyle=""),
     Line2D([0], [0], marker="o", color="w", markerfacecolor=magma_colors[1],
               markeredgecolor="white", markersize=6, linestyle=""),
     Line2D([0], [0], marker="o", color="w", markerfacecolor=magma_colors[2],
               markeredgecolor="white", markersize=6, linestyle=""),
    ),
        Line2D(
        [0],
        [0],
        color=sample_color,
        lw=2,
        marker=">",
        markersize=8,
        markevery=[1],
           label="Weighted direction (last step)"),
]
legend_labels = [
    "Policy update steps",
    "Sampled actions (final step)",
    "Weighted update direction",
]
ax.legend(
    handles=legend_handles,
    labels=legend_labels,
    loc="lower left",
    frameon=True,
    framealpha=0.9,
    borderpad=0.4,
    labelspacing=0.4,
    handler_map={tuple: HandlerTuple(ndivide=None)},
)

# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

# Final point marker (optional)
# ax.scatter([a_opt[0]], [a_opt[1]], s=260, marker="s")

ax.set_xlabel(r"Action Dim. 1")
ax.set_ylabel(r"Action Dim. 2")
ax.set_xlim(a1[0], a1[-1])
ax.set_ylim(a2[0], a2[-1])

plt.show()

fig.savefig("ov.png", dpi=300, bbox_inches="tight")
fig.savefig("ov.pdf", dpi=300, bbox_inches="tight")
