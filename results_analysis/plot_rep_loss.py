import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import torch

def plot_representation_loss_analysis():
    """
    Visualize the recursive n-step cosine loss mechanism.
    
    The loss trains embeddings such that:
    - States with similar Q-values have high cosine similarity
    - States with different Q-values have low/negative cosine similarity
    - The target similarity Y = (1-λ)*T + λ*(1-done)*γ*S_next
      where T = 1 - 2*(|Q_i - Q_j|/β)^γ_shape
    """
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (20, 12)
    
    # Parameters from the loss function
    gamma_shape = 1.0
    lam = 0.5
    discount = 0.99
    beta = 100.0  # Example beta value (EMA of 95th percentile return gap)
    
    # Create range of normalized differences (Delta) and cosine similarities
    delta_range = np.linspace(0, 1, 200)
    cos_sim_range = np.linspace(-1, 1, 200)
    
    Delta, Cos_sim = np.meshgrid(delta_range, cos_sim_range)
    
    # Compute T (terminal state case for simplicity)
    T = 1.0 - 2.0 * (Delta ** gamma_shape)
    
    # Target similarity (assuming terminal state, so no S_next term)
    Y_terminal = (1.0 - lam) * T
    
    # Target similarity (non-terminal, assuming S_next = current similarity for visualization)
    Y_nonterminal = (1.0 - lam) * T + lam * discount * Cos_sim
    
    # Compute loss (Smooth L1 / Huber loss)
    huber_delta = 0.2
    
    def smooth_l1(x, delta=0.2):
        """Smooth L1 loss element"""
        abs_x = np.abs(x)
        return np.where(abs_x < delta, 
                       0.5 * x**2 / delta,
                       abs_x - 0.5 * delta)
    
    loss_terminal = smooth_l1(Cos_sim - Y_terminal, huber_delta)
    loss_nonterminal = smooth_l1(Cos_sim - Y_nonterminal, huber_delta)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(22, 14))
    
    # 1. Target Similarity T vs normalized difference Delta
    ax1 = plt.subplot(3, 4, 1)
    delta_1d = np.linspace(0, 1, 1000)
    t_1d = 1.0 - 2.0 * (delta_1d ** gamma_shape)
    ax1.plot(delta_1d, t_1d, linewidth=3, color='#2E86AB')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Δ=0.5')
    ax1.fill_between(delta_1d, t_1d, -1, alpha=0.2, color='red', 
                     label='Push apart (T < 0)')
    ax1.fill_between(delta_1d, t_1d, 1, alpha=0.2, color='green',
                     label='Pull together (T > 0)')
    ax1.set_xlabel('Δ (Normalized Q-difference)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Target Similarity T', fontsize=12, fontweight='bold')
    ax1.set_title('Target Similarity Function\nT = 1 - 2*Δ^γ', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.1, 1.1)
    
    # 2. Heatmap: Loss for Terminal States
    ax2 = plt.subplot(3, 4, 2)
    im1 = ax2.contourf(Delta, Cos_sim, loss_terminal, levels=30, cmap='RdYlGn_r')
    ax2.contour(Delta, Cos_sim, loss_terminal, levels=[0.1, 0.5, 1.0], 
                colors='black', linewidths=1, alpha=0.4)
    plt.colorbar(im1, ax=ax2, label='Loss')
    ax2.set_xlabel('Δ (Normalized Q-difference)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cosine Similarity z_i·z_j', fontsize=11, fontweight='bold')
    ax2.set_title('Loss Landscape (Terminal States)\nλ=0 component only', 
                  fontsize=12, fontweight='bold')
    
    # 3. Heatmap: Loss for Non-Terminal States
    ax3 = plt.subplot(3, 4, 3)
    im2 = ax3.contourf(Delta, Cos_sim, loss_nonterminal, levels=30, cmap='RdYlGn_r')
    ax3.contour(Delta, Cos_sim, loss_nonterminal, levels=[0.1, 0.5, 1.0],
                colors='black', linewidths=1, alpha=0.4)
    plt.colorbar(im2, ax=ax3, label='Loss')
    ax3.set_xlabel('Δ (Normalized Q-difference)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Cosine Similarity z_i·z_j', fontsize=11, fontweight='bold')
    ax3.set_title('Loss Landscape (Non-Terminal States)\nWith bootstrapping term', 
                  fontsize=12, fontweight='bold')
    
    # 4. Target vs Current Similarity (Terminal)
    ax4 = plt.subplot(3, 4, 4)
    im3 = ax4.contourf(Delta, Cos_sim, Y_terminal, levels=30, cmap='RdBu_r')
    ax4.contour(Delta, Cos_sim, Y_terminal, levels=[-0.5, 0, 0.5],
                colors='black', linewidths=1.5, alpha=0.5)
    plt.colorbar(im3, ax=ax4, label='Target Y')
    ax4.set_xlabel('Δ (Normalized Q-difference)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Cosine Similarity z_i·z_j', fontsize=11, fontweight='bold')
    ax4.set_title('Target Similarity Y (Terminal)\nWhere embeddings should be pushed', 
                  fontsize=12, fontweight='bold')
    
    # 5. Delta is already normalized (identity plot removed, showing Q->Delta transformation)
    ax5 = plt.subplot(3, 4, 5)
    # Show the original transformation for reference
    q_values = np.linspace(0, 2*beta, 1000)
    delta_from_q = np.clip(q_values / beta, 0, 1)
    ax5.plot(q_values, delta_from_q, linewidth=3, color='#A23B72')
    ax5.axvline(x=beta, color='red', linestyle='--', alpha=0.5, label=f'β={beta}')
    ax5.set_xlabel('|Q_i - Q_j| (Raw Q-difference)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Δ (Normalized difference)', fontsize=12, fontweight='bold')
    ax5.set_title('Normalization: Δ = clip(|Q_i - Q_j|/β, 0, 1)', 
                  fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-0.1, 1.1)
    
    # 6. Loss gradient w.r.t cosine similarity
    ax6 = plt.subplot(3, 4, 6)
    # Sample a few Delta values
    delta_samples = [0, 0.25, 0.5, 0.75, 1.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(delta_samples)))
    
    for delta_val, color in zip(delta_samples, colors):
        t_val = 1.0 - 2.0 * (delta_val ** gamma_shape)
        y_val = (1.0 - lam) * t_val
        
        error = cos_sim_range - y_val
        loss_1d = smooth_l1(error, huber_delta)
        ax6.plot(cos_sim_range, loss_1d, linewidth=2.5, 
                label=f'Δ={delta_val:.2f}, T={t_val:.2f}', color=color)
    
    ax6.set_xlabel('Current Cosine Similarity', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax6.set_title('Loss vs Cosine Similarity\nFor different normalized gaps Δ', 
                  fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9, loc='upper right')
    ax6.grid(True, alpha=0.3)
    
    # 7. 3D-style gradient field
    ax7 = plt.subplot(3, 4, 7)
    # Compute gradient of loss w.r.t. cosine similarity
    grad_cos = np.gradient(loss_terminal, axis=0)
    # Subsample for visualization
    skip = 10
    Delta_sub = Delta[::skip, ::skip]
    Cos_sub = Cos_sim[::skip, ::skip]
    grad_sub = grad_cos[::skip, ::skip]
    
    # Color by gradient magnitude
    grad_mag = np.abs(grad_sub)
    ax7.quiver(Delta_sub, Cos_sub, np.zeros_like(grad_sub), grad_sub,
              grad_mag, cmap='plasma', scale=20, width=0.003)
    ax7.set_xlabel('Δ (Normalized difference)', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Cosine Similarity', fontsize=11, fontweight='bold')
    ax7.set_title('Loss Gradient Direction\nHow embeddings are pushed', 
                  fontsize=12, fontweight='bold')
    
    # 8. Effect of gamma_shape parameter
    ax8 = plt.subplot(3, 4, 8)
    gamma_values = [0.5, 1.0, 2.0, 4.0]
    colors_gamma = plt.cm.cool(np.linspace(0, 1, len(gamma_values)))
    
    for gamma_val, color in zip(gamma_values, colors_gamma):
        t_gamma = 1.0 - 2.0 * (delta_1d ** gamma_val)
        ax8.plot(delta_1d, t_gamma, linewidth=2.5, 
                label=f'γ_shape={gamma_val}', color=color)
    
    ax8.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax8.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Δ (Normalized difference)', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Target Similarity T', fontsize=11, fontweight='bold')
    ax8.set_title('Effect of γ_shape Parameter\nControls transition sharpness', 
                  fontsize=12, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(-1.1, 1.1)
    
    # 9. Effect of lambda parameter
    ax9 = plt.subplot(3, 4, 9)
    lambda_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    colors_lam = plt.cm.autumn(np.linspace(0, 1, len(lambda_values)))
    
    # For a specific normalized difference
    delta_specific = 0.5
    t_specific = 1.0 - 2.0 * (delta_specific ** gamma_shape)
    
    for lam_val, color in zip(lambda_values, colors_lam):
        y_lam = (1.0 - lam_val) * t_specific + lam_val * discount * cos_sim_range
        ax9.plot(cos_sim_range, y_lam, linewidth=2.5,
                label=f'λ={lam_val}', color=color)
    
    ax9.axhline(y=t_specific, color='black', linestyle='--', alpha=0.3, 
                label=f'Pure T (λ=0)')
    ax9.set_xlabel('Next-state Similarity S_next', fontsize=11, fontweight='bold')
    ax9.set_ylabel('Target Y', fontsize=11, fontweight='bold')
    ax9.set_title(f'Effect of λ Parameter (Δ={delta_specific:.1f})\nBootstrapping weight', 
                  fontsize=12, fontweight='bold')
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)
    
    # 10. Distribution visualization
    ax10 = plt.subplot(3, 4, 10)
    # Simulate a batch of normalized differences
    np.random.seed(42)
    delta_sample = np.random.beta(2, 2, 1000)  # Beta distribution for normalized values
    
    ax10.hist(delta_sample, bins=50, alpha=0.7, color='skyblue', 
             edgecolor='black', density=True)
    ax10.axvline(x=np.quantile(delta_sample, 0.95), color='red', linestyle='--', linewidth=2,
                label=f'95th percentile')
    ax10.set_xlabel('Δ (Normalized difference) in batch', fontsize=11, fontweight='bold')
    ax10.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax10.set_title('Normalized Difference Distribution\nΔ ∈ [0,1] after normalization', 
                  fontsize=12, fontweight='bold')
    ax10.legend(fontsize=10)
    ax10.grid(True, alpha=0.3, axis='y')
    
    # 11. Conceptual diagram of the mechanism
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')
    
    # Create conceptual illustration
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    
    # Box 1: Q-values
    box1 = FancyBboxPatch((0.1, 0.7), 0.3, 0.15, boxstyle="round,pad=0.01",
                          edgecolor='#2E86AB', facecolor='#A8DADC', linewidth=2)
    ax11.add_patch(box1)
    ax11.text(0.25, 0.775, 'Q-values\n|Q_i - Q_j|', ha='center', va='center',
             fontsize=11, fontweight='bold')
    
    # Arrow to Delta
    arrow1 = FancyArrowPatch((0.4, 0.775), (0.55, 0.775),
                            arrowstyle='->', mutation_scale=20, linewidth=2,
                            color='black')
    ax11.add_patch(arrow1)
    ax11.text(0.475, 0.82, 'normalize\nby β', ha='center', fontsize=9)
    
    # Box 2: Delta
    box2 = FancyBboxPatch((0.55, 0.7), 0.25, 0.15, boxstyle="round,pad=0.01",
                          edgecolor='#A23B72', facecolor='#F18F01', linewidth=2)
    ax11.add_patch(box2)
    ax11.text(0.675, 0.775, 'Δ ∈ [0,1]\nΔ=|Q|/β',
             ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrow to T
    arrow2 = FancyArrowPatch((0.675, 0.7), (0.675, 0.55),
                            arrowstyle='->', mutation_scale=20, linewidth=2,
                            color='black')
    ax11.add_patch(arrow2)
    ax11.text(0.75, 0.625, 'T=1-2Δ^γ', ha='left', fontsize=9)
    
    # Box 3: T
    box3 = FancyBboxPatch((0.55, 0.35), 0.25, 0.15, boxstyle="round,pad=0.01",
                          edgecolor='#06A77D', facecolor='#90EE90', linewidth=2)
    ax11.add_patch(box3)
    ax11.text(0.675, 0.425, 'Target T\n∈ [-1,1]',
             ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Box 4: S_next (for non-terminal)
    box4 = FancyBboxPatch((0.1, 0.35), 0.3, 0.15, boxstyle="round,pad=0.01",
                          edgecolor='#457B9D', facecolor='#E0F4FF', linewidth=2)
    ax11.add_patch(box4)
    ax11.text(0.25, 0.425, 'S_next\nBootstrap',
             ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Combine
    arrow3 = FancyArrowPatch((0.4, 0.425), (0.475, 0.15),
                            arrowstyle='->', mutation_scale=20, linewidth=2,
                            color='black')
    ax11.add_patch(arrow3)
    ax11.text(0.4, 0.25, 'λ', ha='center', fontsize=11, fontweight='bold')
    
    arrow4 = FancyArrowPatch((0.55, 0.35), (0.525, 0.15),
                            arrowstyle='->', mutation_scale=20, linewidth=2,
                            color='black')
    ax11.add_patch(arrow4)
    ax11.text(0.58, 0.25, '1-λ', ha='center', fontsize=11, fontweight='bold')
    
    # Final target
    box5 = FancyBboxPatch((0.35, 0.02), 0.35, 0.13, boxstyle="round,pad=0.01",
                          edgecolor='red', facecolor='#FFE5E5', linewidth=3)
    ax11.add_patch(box5)
    ax11.text(0.525, 0.085, 'Target Y\n=(1-λ)T + λγS_next',
             ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax11.set_xlim(0, 1)
    ax11.set_ylim(0, 1)
    ax11.set_title('Representation Loss Flow\nHow targets are computed', 
                  fontsize=12, fontweight='bold')
    
    # 12. Summary comparison
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    summary_text = """
    KEY INSIGHTS:
    
    1. GOAL: Learn embeddings where cosine 
       similarity reflects Q-value similarity
    
    2. MECHANISM:
       • Similar Q-values (small gap) → T ≈ +1
         → Embeddings pulled together
       • Different Q-values (large gap) → T ≈ -1
         → Embeddings pushed apart
    
    3. ADAPTIVE SCALING:
       • β = EMA of 95th percentile gap
       • Automatically adjusts to value scale
    
    4. BOOTSTRAPPING:
       • Terminal: Y = (1-λ)T
       • Non-terminal: Y = (1-λ)T + λγS_next
       • Propagates similarity through time
    
    5. LOSS: Smooth L1 between current 
       similarity S and target Y
       • Trains encoder to match targets
    """
    
    ax12.text(0.05, 0.95, summary_text, ha='left', va='top',
             fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Recursive N-Step Cosine Loss: Complete Analysis',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    return fig


def plot_interactive_loss_surface():
    """
    Create a detailed 2D heatmap showing the loss surface
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    gamma_shape_vals = [0.5, 1.0, 2.0]
    lam_vals = [0.0, 0.5]
    
    delta_range = np.linspace(0, 1, 300)
    cos_sim_range = np.linspace(-1, 1, 300)
    Delta, Cos_sim = np.meshgrid(delta_range, cos_sim_range)
    
    def smooth_l1(x, delta=0.2):
        abs_x = np.abs(x)
        return np.where(abs_x < delta, 
                       0.5 * x**2 / delta,
                       abs_x - 0.5 * delta)
    
    for i, gamma_shape in enumerate(gamma_shape_vals):
        for j, lam in enumerate(lam_vals):
            ax = axes[j, i]
            
            T = 1.0 - 2.0 * (Delta ** gamma_shape)
            
            if lam == 0.0:
                Y = T
                title_suffix = "(Terminal)"
            else:
                Y = (1.0 - lam) * T + lam * 0.99 * Cos_sim
                title_suffix = "(Non-Terminal)"
            
            loss = smooth_l1(Cos_sim - Y, 0.2)
            
            # Use seaborn color palette
            im = ax.contourf(Delta, Cos_sim, loss, levels=40, 
                           cmap='RdYlGn_r', vmin=0, vmax=1.5)
            
            # Add contour lines
            contours = ax.contour(Delta, Cos_sim, loss, 
                                 levels=[0.1, 0.3, 0.5, 1.0],
                                 colors='black', linewidths=1.2, alpha=0.4,
                                 linestyles=['--', '-.', ':', '-'])
            ax.clabel(contours, inline=True, fontsize=8)
            
            # Mark the optimal line (where S = Y)
            if lam == 0.0:
                optimal_cos = T
            else:
                # Solve S = (1-λ)T + λγS for S
                optimal_cos = (1.0 - lam) * T / (1.0 - lam * 0.99)
            
            ax.contour(Delta, Cos_sim, Cos_sim - optimal_cos,
                      levels=[0], colors='cyan', linewidths=2.5,
                      linestyles='-', alpha=0.8)
            
            ax.set_xlabel('Δ (Normalized Q-difference)', fontsize=11)
            ax.set_ylabel('Cosine Similarity', fontsize=11)
            ax.set_title(f'γ={gamma_shape}, λ={lam} {title_suffix}',
                        fontsize=12, fontweight='bold')
            
            plt.colorbar(im, ax=ax, label='Loss')
            ax.grid(True, alpha=0.2)
    
    plt.suptitle('Loss Landscape: Effect of γ_shape and λ Parameters\n' +
                'Cyan line shows optimal similarity (zero loss)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_gradient_field_detailed():
    """
    Create detailed gradient field visualizations showing how embeddings are pushed
    """
    fig = plt.figure(figsize=(20, 10))
    
    gamma_shape_vals = [0.5, 1.0, 2.0]
    lam_vals = [0.0, 0.5]
    
    # Higher resolution for smoother gradients
    delta_range = np.linspace(0, 1, 100)
    cos_sim_range = np.linspace(-1, 1, 100)
    Delta, Cos_sim = np.meshgrid(delta_range, cos_sim_range)
    
    def smooth_l1(x, delta=0.2):
        abs_x = np.abs(x)
        return np.where(abs_x < delta, 
                       0.5 * x**2 / delta,
                       abs_x - 0.5 * delta)
    
    plot_idx = 1
    for j, lam in enumerate(lam_vals):
        for i, gamma_shape in enumerate(gamma_shape_vals):
            ax = plt.subplot(2, 3, plot_idx)
            plot_idx += 1
            
            T = 1.0 - 2.0 * (Delta ** gamma_shape)
            
            if lam == 0.0:
                Y = T
                title_suffix = "Terminal"
            else:
                Y = (1.0 - lam) * T + lam * 0.99 * Cos_sim
                title_suffix = "Non-Terminal"
            
            loss = smooth_l1(Cos_sim - Y, 0.2)
            
            # Plot loss landscape as background
            im = ax.contourf(Delta, Cos_sim, loss, levels=30, 
                           cmap='RdYlGn_r', alpha=0.6, vmin=0, vmax=1.2)
            
            # Compute gradients
            grad_delta = np.gradient(loss, axis=1)
            grad_cos = np.gradient(loss, axis=0)
            
            # Normalize gradients for visualization
            grad_magnitude = np.sqrt(grad_delta**2 + grad_cos**2)
            grad_magnitude = np.maximum(grad_magnitude, 1e-8)  # Avoid division by zero
            
            # Subsample for arrow visualization (more arrows, smaller)
            skip = 5
            Delta_sub = Delta[::skip, ::skip]
            Cos_sub = Cos_sim[::skip, ::skip]
            grad_delta_sub = grad_delta[::skip, ::skip]
            grad_cos_sub = grad_cos[::skip, ::skip]
            grad_mag_sub = grad_magnitude[::skip, ::skip]
            
            # Normalize arrow lengths
            arrow_delta = -grad_delta_sub / grad_mag_sub * 0.02
            arrow_cos = -grad_cos_sub / grad_mag_sub * 0.04
            
            # Plot arrows colored by gradient magnitude
            q = ax.quiver(Delta_sub, Cos_sub, arrow_delta, arrow_cos,
                         grad_mag_sub, cmap='plasma', 
                         scale=1, scale_units='xy', angles='xy',
                         width=0.003, headwidth=4, headlength=5,
                         alpha=0.8, edgecolors='k', linewidths=0.5)
            
            # Add contour lines for reference
            contours = ax.contour(Delta, Cos_sim, loss, 
                                 levels=[0.1, 0.3, 0.5, 0.8],
                                 colors='black', linewidths=1.5, alpha=0.3,
                                 linestyles=['-', '--', '-.', ':'])
            
            # Mark zero-loss optimal line
            if lam == 0.0:
                optimal_cos = T
            else:
                optimal_cos = (1.0 - lam) * T / (1.0 - lam * 0.99)
            
            ax.contour(Delta, Cos_sim, Cos_sim - optimal_cos,
                      levels=[0], colors='cyan', linewidths=3,
                      linestyles='-', alpha=0.9)
            
            ax.set_xlabel('Δ (Normalized Q-difference)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
            ax.set_title(f'{title_suffix}: γ={gamma_shape}, λ={lam}\nArrows show gradient direction',
                        fontsize=12, fontweight='bold')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, 1)
            ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
            
            # Add colorbar for gradient magnitude
            cbar = plt.colorbar(q, ax=ax, label='Gradient Magnitude', pad=0.02)
            cbar.ax.tick_params(labelsize=9)
    
    plt.suptitle('Gradient Field: How Embeddings Are Pushed by the Loss\n' +
                'Arrows point in the direction of steepest descent (negative gradient)\n' +
                'Cyan line shows zero-loss optimal path',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


def plot_gradient_cross_sections():
    """
    Create cross-section plots showing gradients at specific Delta values
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    gamma_shape = 1.0
    lam = 0.5
    discount = 0.99
    huber_delta = 0.2
    
    # Different Delta values to examine
    delta_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    cos_sim_range = np.linspace(-1, 1, 500)
    
    def smooth_l1(x, delta=0.2):
        abs_x = np.abs(x)
        return np.where(abs_x < delta, 
                       x / delta,  # Gradient
                       np.sign(x))
    
    for idx, delta_val in enumerate(delta_values):
        ax = axes[idx // 3, idx % 3]
        
        # Compute target
        T_val = 1.0 - 2.0 * (delta_val ** gamma_shape)
        Y_val = (1.0 - lam) * T_val + lam * discount * cos_sim_range
        
        # Compute loss and gradient
        error = cos_sim_range - Y_val
        loss = np.where(np.abs(error) < huber_delta,
                       0.5 * error**2 / huber_delta,
                       np.abs(error) - 0.5 * huber_delta)
        gradient = smooth_l1(error, huber_delta)
        
        # Plot loss
        ax2 = ax.twinx()
        line1 = ax.plot(cos_sim_range, loss, 'b-', linewidth=2.5, label='Loss')
        ax.fill_between(cos_sim_range, 0, loss, alpha=0.2, color='blue')
        
        # Plot gradient with arrows
        line2 = ax2.plot(cos_sim_range, gradient, 'r-', linewidth=2.5, label='Gradient')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        
        # Add arrow annotations at key points
        arrow_positions = np.linspace(-0.8, 0.8, 9)
        for pos in arrow_positions:
            idx_pos = np.argmin(np.abs(cos_sim_range - pos))
            grad_val = gradient[idx_pos]
            if np.abs(grad_val) > 0.1:  # Only show significant gradients
                arrow_length = -0.15 * np.sign(grad_val)
                ax.annotate('', xy=(pos + arrow_length, loss[idx_pos]),
                          xytext=(pos, loss[idx_pos]),
                          arrowprops=dict(arrowstyle='->', color='green', 
                                        lw=2, alpha=0.7))
        
        # Mark optimal point
        optimal_cos = Y_val
        optimal_idx = np.argmin(np.abs(cos_sim_range - np.mean(optimal_cos)))
        ax.plot(cos_sim_range[optimal_idx], loss[optimal_idx], 'go', 
               markersize=10, label='Optimal', zorder=5)
        
        ax.set_xlabel('Cosine Similarity', fontsize=11, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=11, fontweight='bold', color='b')
        ax2.set_ylabel('Gradient', fontsize=11, fontweight='bold', color='r')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax.set_title(f'Δ = {delta_val:.1f}, T = {T_val:.2f}\nGreen arrows show push direction',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', fontsize=9)
    
    plt.suptitle('Loss and Gradient Cross-Sections at Different Δ Values\n' +
                'Green arrows show how embeddings are pushed toward optimal similarity',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


if __name__ == "__main__":
    # Use non-interactive backend
    import matplotlib
    matplotlib.use('Agg')
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs('./plots', exist_ok=True)
    
    print("Generating comprehensive representation loss analysis...")
    fig1 = plot_representation_loss_analysis()
    fig1.savefig('./plots/representation_loss_complete_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("✓ Saved: ./plots/representation_loss_complete_analysis.png")
    
    print("\nGenerating detailed loss surface heatmaps...")
    fig2 = plot_interactive_loss_surface()
    fig2.savefig('./plots/representation_loss_surface_heatmaps.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("✓ Saved: ./plots/representation_loss_surface_heatmaps.png")
    
    print("\nGenerating gradient field visualization...")
    fig3 = plot_gradient_field_detailed()
    fig3.savefig('./plots/gradient_field_arrows.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("✓ Saved: ./plots/gradient_field_arrows.png")
    
    print("\nGenerating gradient cross-sections...")
    fig4 = plot_gradient_cross_sections()
    fig4.savefig('./plots/gradient_cross_sections.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print("✓ Saved: ./plots/gradient_cross_sections.png")
    
    print("\n✅ All visualizations created successfully!")
    print("\nGenerated plots:")
    print("1. representation_loss_complete_analysis.png - Complete mechanism (12 subplots)")
    print("2. representation_loss_surface_heatmaps.png - Loss surfaces as heatmaps")
    print("3. gradient_field_arrows.png - NEW! Gradient field with visible arrows")
    print("4. gradient_cross_sections.png - NEW! Cross-sections showing push directions")
    print("\nKey insights:")
    print("- Arrows show the direction embeddings are pushed by gradients")
    print("- Cyan lines mark zero-loss optimal paths")
    print("- X-axis shows normalized Δ ∈ [0,1] instead of raw Q-differences")
    print("- The loss encourages similar Q-values → similar embeddings")
