import numpy as np
import matplotlib.pyplot as plt

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
fig, ax = plt.subplots(figsize=(4,3))

ax.contour(A1, A2, Q, levels=15, linewidths=0.8, alpha=0.5, cmap="viridis")

# Connected trajectory (no dots)
ax.plot(anchors[:, 0], anchors[:, 1], linewidth=2.5, alpha=0.15,
        zorder=2, color='red')

# Direction arrows along the trajectory
stride = max(1, T // 22)  # ~20 arrows
x = anchors[:-1:stride, 0]
y = anchors[:-1:stride, 1]
dx = anchors[1::stride, 0] - anchors[:-1:stride, 0]
dy = anchors[1::stride, 1] - anchors[:-1:stride, 1]
#exclude last arrow to avoid overlap with sampled actions
x = x[:-1]
y = y[:-1]
dx = dx[:-1]
dy = dy[:-1]

ax.quiver(x, y, dx, dy, angles='xy',
          scale_units='xy', scale=1,
          width=0.005, 
          alpha=1,
          headwidth=5, 
        # headlength=7, headaxislength=5,
        zorder=4,
          color='red',
          )
# LAST step sampled actions (colored by Q)
a_last_anchor = anchors[-2]
for p in last_neighbors:
    ax.plot([a_last_anchor[0], p[0]], [a_last_anchor[1], p[1]], linewidth=0.9, alpha=0.24)

sc = ax.scatter(last_neighbors[:, 0], last_neighbors[:, 1],
                s=20,
                c=last_q,
                alpha=0.95,
                zorder=3,)

cbar = fig.colorbar(sc, ax=ax, 
                    # pad=0.02,
                    shrink=1)
cbar.set_label(r"$Q(s_{t},a_{t})$")

#cbar fontsize
cbar.ax.tick_params(labelsize=7)

#add a marker at the starting point
ax.scatter([a0[0]], [a0[1]], s=20, marker="o", color="r", zorder=5)

#have yticks every 1 unit
ax.set_yticks(np.arange(a2[0], a2[-1] + 1, 1))
#reduce font size of ticks
ax.tick_params(axis='both', which='major', labelsize=8)

# Weighted direction at the last step
ax.annotate(
    "", xy=last_weighted, xytext=a_last_anchor,
    arrowprops=dict(arrowstyle="->",
                    linewidth=1.7,
                    alpha=0.95,
                    zorder=0,)
)

from matplotlib.lines import Line2D
# ---------- Legend (proxy artists) ----------
legend_handles = [
    Line2D([0], [0], color="red", lw=1.2, label="Policy Update"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="g", markersize=6,
           label="Sampled actions (last step)"),
    
]
ax.legend(handles=legend_handles, loc="lower left",
          frameon=True, framealpha=0.9, fontsize=9)


# Final point marker
# ax.scatter([a_opt[0]], [a_opt[1]], s=260, marker="s")

# ax.set_title("Distance-RL: multi-step improvement (neighbors shown only at the last step)")
ax.set_xlabel(r"Action Dim. 1")
ax.set_ylabel(r"Action Dim. 2")
ax.set_xlim(a1[0], a1[-1])
ax.set_ylim(a2[0], a2[-1])

plt.tight_layout()
plt.show()

# Optional save:
fig.savefig("distance_rl_multistep_last_samples_clean.png", dpi=300, bbox_inches="tight")
