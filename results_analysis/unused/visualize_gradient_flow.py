import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize this:
# Delta = (G / beta).clamp(0., 1.)
# T = 1.0 - 2.0 * (Delta ** float(gamma_shape))
# alive = (1.0 - dones.view(-1, 1)).to(S.dtype)
# Y = (1.0 - lam) * T + lam * alive * (discount * S_next)

lam = 0.5
gamma_shape_values = [0.5, 1.0, 2.0]
discount = 0.99
alive = 1.0

# Create 1D arrays
Delta_1d = np.linspace(0, 1, 200)
S_next_1d = np.linspace(-1, 1, 200)

# Create 2D meshgrid
Delta, S_next = np.meshgrid(Delta_1d, S_next_1d)

# Create figure with nice styling
plt.style.use('seaborn-v0_8-paper')
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, gamma_shape in enumerate(gamma_shape_values):
    ax = axes[idx]
    
    # Compute T and cosine_similarity_now on the 2D grid
    T = 1.0 - 2.0 * (Delta ** float(gamma_shape))
    cosine_similarity_now = (1.0 - lam) * T + lam * alive * (discount * S_next)
    
    # Plot heatmap
    im = ax.imshow(cosine_similarity_now, extent=[0, 1, -1, 1], origin='lower', 
                   cmap='RdBu_r', aspect='auto', alpha=0.9)
    
    # Add contour lines
    contour_levels = np.linspace(cosine_similarity_now.min(), cosine_similarity_now.max(), 15)
    contours = ax.contour(Delta, S_next, cosine_similarity_now, levels=contour_levels, 
                          colors='black', linewidths=0.8, alpha=0.4)
    ax.clabel(contours, inline=True, fontsize=7, fmt='%.2f')
    
    # Add zero contour with emphasis
    zero_contour = ax.contour(Delta, S_next, cosine_similarity_now, levels=[0], 
                              colors='yellow', linewidths=2.5, linestyles='--')
    ax.clabel(zero_contour, inline=True, fontsize=9, fmt='%.1f')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=15)
    cbar.set_label('$\\cos(now)$', fontsize=11, rotation=270, labelpad=20)
    
    # Labels and title
    ax.set_xlabel('$\\Delta$ (Normalized Q-diff)', fontsize=12, fontweight='bold')
    ax.set_ylabel('$S_{next}$ (Cosine sim. at next step)', fontsize=12, fontweight='bold')
    ax.set_title(f'$\\gamma_{{shape}} = {gamma_shape}$\n',
                fontsize=13, fontweight='bold', pad=10)
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)

# Overall title
fig.suptitle(f'Target Function: $\\cos(now) = (1-\\lambda)T + \\lambda \\cdot discount \\cdot S_{{next}}$, ' +
            f'where $T = 1 - 2\\Delta^{{\\gamma_{{shape}}}}$\n' +
            f'$\\lambda={lam}, discount={discount}$',
            fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('./results_analysis/plots/visualize_gradient_flow.png', dpi=300, bbox_inches='tight')
print(f'\n✓ Saved: ./results_analysis/plots/visualize_gradient_flow.png')
for gamma_shape in gamma_shape_values:
    T = 1.0 - 2.0 * (Delta ** float(gamma_shape))
    cosine_similarity_now = (1.0 - lam) * T + lam * alive * (discount * S_next)
    print(f'γ_shape={gamma_shape}: cosine_similarity_now range [{cosine_similarity_now.min():.3f}, {cosine_similarity_now.max():.3f}]')
plt.show()
