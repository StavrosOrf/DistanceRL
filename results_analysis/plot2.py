import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# ----------------------------
# Match your 2D plot styling + colors
# ----------------------------
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
    "lines.linewidth": 3.0,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
})

palette = sns.color_palette("muted", 6)
magma = cm.get_cmap("inferno")
magma_colors = magma(np.linspace(0.25, 0.85, 3))

arrow_color  = "black"      # anchor
sample_color = "limegreen"  # weighted direction
traj_color   = palette[0]   # curve color (subtle)

# ----------------------------
# Helpers
# ----------------------------
def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / (n + eps)

def softmax(x, tau=1.0):
    x = np.asarray(x, dtype=float) / max(tau, 1e-12)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)

def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])

# 3D->2D arrow patch for curved arrows in 3D
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj())
        self.set_positions((xs[0], ys[0]), (xs[-1], ys[-1]))
        return float(np.min(zs))

def spherical_arc(a, b, n=90):
    """
    Great-circle (slerp) arc on unit sphere from a to b (both unit vectors).
    Returns (n,3) points.
    """
    a = normalize(a)
    b = normalize(b)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    theta = np.arccos(dot)
    if theta < 1e-8:
        return np.repeat(a[None, :], n, axis=0)

    t = np.linspace(0, 1, n)
    sin_theta = np.sin(theta)
    p = (np.sin((1 - t) * theta)[:, None] / sin_theta) * a[None, :] + \
        (np.sin(t * theta)[:, None] / sin_theta) * b[None, :]
    return np.array([normalize(pi) for pi in p])

# ----------------------------
# Synthetic sphere embeddings + Q utilities (replace with your real data)
# ----------------------------
rng = np.random.default_rng(9)
K = 32

Z = rng.normal(size=(K, 3))
Z = np.array([normalize(v) for v in Z])

preferred = normalize(np.array([0.25, 0.85, 0.45]))
q = 1.8 * (Z @ preferred) + 0.25 * rng.normal(size=K)

# Normalize Q to [0,1]
q01 = (q - q.min()) / (q.max() - q.min() + 1e-12)
norm01 = colors.Normalize(vmin=0.0, vmax=1.0)

# Weights from Q (utility-weighted direction)
tau = 0.35
w = softmax(q, tau=tau)
v_sum = (w[:, None] * Z).sum(axis=0)
v_sum_unit = normalize(v_sum)

# Anchor (fixed to a visible front-facing spot)
z_anchor = normalize(np.array([0,0, 1]))

# ----------------------------
# Plot
# ----------------------------
fig = plt.figure(figsize=(4.5,3), constrained_layout=False)
ax = fig.add_subplot(111, projection="3d")

# Unit sphere wireframe
u = np.linspace(0, 2*np.pi, 72)
v = np.linspace(0, np.pi, 36)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))
ax.plot_wireframe(xs, ys, zs, rstride=2, cstride=2, linewidth=0.3, alpha=0.16, color="0.01")

# Candidate arrows from origin, colored by Q (magma)
for i in range(K):
    col = magma(norm01(q01[i]))
    lw = 0.9 + 2.6 * q01[i]
    a  = 0.25 + 0.75 * q01[i]
    ax.quiver(
        0, 0, 0,
        Z[i, 0], Z[i, 1], Z[i, 2],
        length=0.95,
        normalize=False,
        linewidth=lw, 
        alpha=a, color=col,
        # headlength=8,
        # headwidth=6,
        arrow_length_ratio=0.2
    )

# Candidate tips
ax.scatter(
    Z[:, 0], Z[:, 1], Z[:, 2],
    s=12, c=q01, cmap="inferno", norm=norm01,
    edgecolors="grey", linewidths=0.45,
    alpha=0.95, zorder=0
)

#put a big sphere around the origin
u_sphere, v_sphere = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
x_sphere = 1.0 * np.cos(u_sphere) * np.sin(v_sphere)
y_sphere = 1.0 * np.sin(u_sphere) * np.sin(v_sphere)
z_sphere = 1.0 * np.cos(v_sphere)
ax.plot_surface(
    x_sphere, y_sphere, z_sphere,
    rstride=4, cstride=4,
    color="grey", alpha=0.08,
    edgecolor="none",
    zorder=0
)

# Anchor (black)
ax.quiver(0, 0, 0, *z_anchor, length=0.95, normalize=False,
          linewidth=3.2, alpha=0.98, color=arrow_color,
          arrow_length_ratio=0.26,
          zorder=10)
# ax.scatter([z_anchor[0]],[z_anchor[1]],[z_anchor[2]],
#            s=55, color=arrow_color, edgecolors="white", linewidths=0.6, zorder=6)

# Weighted direction (limegreen)
ax.quiver(0, 0, 0, *v_sum_unit, length=0.95, normalize=False,
          linewidth=3.4, alpha=0.95, color=sample_color,
          arrow_length_ratio=0.26,
          zorder=20)
# ax.scatter([v_sum_unit[0]],[v_sum_unit[1]],[v_sum_unit[2]],
#            s=55, color=sample_color, edgecolors="white", linewidths=0.6, zorder=6)

# ---- Curved arrow on the sphere: anchor -> weighted direction (great-circle arc) ----
arc = spherical_arc(z_anchor, v_sum_unit, n=120)
arc = arc * 0.6  # Scale to radius 0.2
ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=traj_color, alpha=0.9, linewidth=2.6, zorder=40)

# Add a small arrowhead along the arc (as a 3D projected patch)
j0, j1 = 95, 119  # last segment defines arrow direction
arr = Arrow3D(
    [arc[j0, 0], arc[j1, 0]],
    [arc[j0, 1], arc[j1, 1]],
    [arc[j0, 2], arc[j1, 2]],
    mutation_scale=16,
    lw=1.0,
    arrowstyle="-|>",
    color=traj_color,
    alpha=0.95,
    zorder=400
)
ax.add_artist(arr)

# Origin marker
ax.scatter([0], [0], [0], s=45, color="white", edgecolors="black", linewidths=1.0, zorder=7)

# Legend (same magma swatches style)
legend_handles = [
    Line2D([0],[0], color=arrow_color, lw=3, marker=">", markersize=9, markevery=[1],
           label="Anchor embedding"),
    (Line2D([0],[0], marker="o", color="w", markerfacecolor=magma_colors[0],
            markeredgecolor="white", markersize=7, linestyle=""),
     Line2D([0],[0], marker="o", color="w", markerfacecolor=magma_colors[1],
            markeredgecolor="white", markersize=7, linestyle=""),
     Line2D([0],[0], marker="o", color="w", markerfacecolor=magma_colors[2],
            markeredgecolor="white", markersize=7, linestyle="")),
    Line2D([0],[0], color=sample_color, lw=3, marker=">", markersize=9, markevery=[1],
           label="Weighted direction"),
    Line2D([0],[0], color=traj_color, lw=3, label="Update path on sphere"),
]
legend_labels = [
    "Anchor action embedding",
    "Candidate embeddings",
    "Weighted update direction",
    "Gradient of update",
]
ax.legend(
    legend_handles, legend_labels,
    loc="lower left",
    bbox_to_anchor=(-0.43, 0.84),
    frameon=True, framealpha=0.92,
    borderpad=0.4, labelspacing=0.2,
    ncol=2,
    handler_map={tuple: plt.matplotlib.legend_handler.HandlerTuple(ndivide=None)}
)

# Labels / view
ax.set_xlabel("Dim. 1")
ax.set_ylabel("Dim. 2")
ax.set_zlabel("Dim. 3")

ax.view_init(elev=18, azim=45)
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)
ax.set_zlim(-1.0, 1.0)
set_axes_equal(ax)

# Colorbar: normalized Q in [0,1]
mappable = cm.ScalarMappable(norm=norm01, cmap=magma)
mappable.set_array([])
cbar = fig.colorbar(mappable, ax=ax, fraction=0.035, pad=0.1, shrink=0.65)
cbar.set_label(r"$Q(s_{t},a_{t})$", labelpad=8)
cbar.set_ticks(np.arange(0, 1.01, 0.2))
cbar.ax.tick_params(labelsize=10)

ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.set_zticks([-1, 0, 1])
# ax.tick_params(pad=-2)

#add minor ticks every at 0.5 intervals with dotted grid lines
ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
ax.zaxis.set_minor_locator(plt.MultipleLocator(0.5))

# Save
fig.savefig("sphere_geometry_curved_update.png", dpi=300, bbox_inches="tight")
fig.savefig("sphere_geometry_curved_update.pdf", dpi=300 )#, bbox_inches="tight")
# plt.show()
