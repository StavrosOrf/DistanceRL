"""
Advanced Representation Geometry & Topology Visualizations for DistRL
=====================================================================

Implements comprehensive analysis of the learned representation space:

🧩 1. Representation Geometry & Topology
    (a) Latent Manifold Projection (UMAP/t-SNE)
    (b) Latent Distance vs. Reward Difference
    (c) Distance Heatmap (Pairwise Similarity Matrix)
    (d) Temporal Consistency Map

🎯 2. Linking Representation to Policy/Value Behavior
    (a) Action Clustering in Latent Space
    (b) Value Gradient Direction in Latent Space
    (c) Policy Coverage Ellipses
    (d) Representation Smoothness vs. TD Error

🔄 3. Training Evolution
    (a) Latent Drift Animation (if checkpoints available)
    (b) Representation Compression Curve
    (c) Mutual Information Evolution

🧬 4. Advanced Geometric Visualizations
    (a) Latent Curvature Field
    (b) Latent Transition Graph
    (c) Reward-Aware Cosine Similarity Map

🧠 5. Paper-Ready 3-Panel Figure

Usage:
    python results_analysis/visualize_representation_geometry.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gymnasium as gym
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.stats import spearmanr, pearsonr
from matplotlib.patches import Ellipse
import networkx as nx
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*threadpoolctl.*')

# Import DistRL components
import sys
sys.path.append('.')
from dist_rl.models import DistanceTrunk, GaussianActor, TwinQ
from dist_rl.utils import RunningMeanStd

# Configuration
DEVICE = 'cpu'
MODEL_PATH = './saved_models/best.pt'
ENV_NAME = 'Walker2d-v5'
NUM_SAMPLES = 1000      # For manifold visualizations
NUM_EPISODES = 1        # For trajectory analysis
MAX_STEPS = 1000         # Steps per episode
OUTPUT_DIR = Path('results_analysis/plots/representation_geometry')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def load_model_and_env():
    """Load trained DistRL model and environment."""
    print("Loading model and environment...")
    
    # Load environment
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Initialize models
    rep_trunk = DistanceTrunk(
        obs_dim=obs_dim,
        act_dim=action_dim,
        hidden=256,
        out_dim=256
    ).to(DEVICE)
    
    actor = GaussianActor(
        obs_dim=obs_dim,
        act_dim=action_dim,
        hidden=256
    ).to(DEVICE)
    
    qnet = TwinQ(
        obs_dim=obs_dim,
        act_dim=action_dim,
        hidden=256
    ).to(DEVICE)
    
    # Load weights
    actor.load_state_dict(checkpoint['actor'])
    rep_trunk.load_state_dict(checkpoint['rep_trunk'])
    qnet.load_state_dict(checkpoint['qnet'])
    
    # Load normalization
    normalization = RunningMeanStd(obs_dim, device=DEVICE)
    normalization.load_state_dict(checkpoint['normalization'])
    
    # Set to eval mode
    actor.eval()
    rep_trunk.eval()
    qnet.eval()
    
    print(f"✓ Model loaded from {MODEL_PATH}")
    print(f"✓ Environment: {ENV_NAME}")
    print(f"✓ Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    return env, actor, rep_trunk, qnet, normalization


def collect_trajectory_data(env, actor, rep_trunk, qnet, normalization, 
                            num_samples=2000, num_episodes=10):
    """Collect comprehensive trajectory data with latent representations."""
    print(f"\nCollecting {num_samples} samples from {num_episodes} episodes...")
    
    data = {
        'states': [],
        'states_next': [],
        'actions': [],
        'rewards': [],
        'latents': [],
        'latents_next': [],
        'q_values': [],
        'episode_ids': [],
        'step_ids': [],
        'gait_phases': []
    }
    
    obs, _ = env.reset()
    episode_id = 0
    step_id = 0
    total_steps = 0
    
    with torch.no_grad():
        while len(data['states']) < num_samples and episode_id < num_episodes:
            # Normalize observation
            obs_norm = normalization.normalize(torch.FloatTensor(obs).to(DEVICE))
            
            # Get action
            mu, _ = actor(obs_norm.unsqueeze(0))
            action = torch.tanh(mu).cpu().numpy()[0]
            
            # Get latent representation
            action_tensor = torch.FloatTensor(action).to(DEVICE).unsqueeze(0)
            latent = rep_trunk(obs_norm.unsqueeze(0), action_tensor)
            
            # Get Q-value
            q1, q2 = qnet(obs_norm.unsqueeze(0), action_tensor)
            q_value = torch.min(q1, q2).cpu().numpy()[0, 0]
            
            # Step environment
            action_env = ((action + 1) / 2) * (env.action_space.high - env.action_space.low) + env.action_space.low
            obs_next, reward, terminated, truncated, _ = env.step(action_env)
            
            # Get next state latent
            obs_next_norm = normalization.normalize(torch.FloatTensor(obs_next).to(DEVICE))
            mu_next, _ = actor(obs_next_norm.unsqueeze(0))
            action_next = torch.tanh(mu_next).cpu().numpy()[0]
            action_next_tensor = torch.FloatTensor(action_next).to(DEVICE).unsqueeze(0)
            latent_next = rep_trunk(obs_next_norm.unsqueeze(0), action_next_tensor)
            
            # Compute gait phase (for Walker2d, use forward velocity as proxy)
            forward_vel = obs[8] if len(obs) > 8 else 0
            gait_phase = (step_id % 100) / 100.0  # Approximate cycle
            
            # Store data
            data['states'].append(obs)
            data['states_next'].append(obs_next)
            data['actions'].append(action)
            data['rewards'].append(reward)
            data['latents'].append(latent.cpu().numpy()[0])
            data['latents_next'].append(latent_next.cpu().numpy()[0])
            data['q_values'].append(q_value)
            data['episode_ids'].append(episode_id)
            data['step_ids'].append(step_id)
            data['gait_phases'].append(gait_phase)
            
            obs = obs_next
            step_id += 1
            total_steps += 1
            
            if terminated or truncated:
                obs, _ = env.reset()
                episode_id += 1
                step_id = 0
                if total_steps % 1000 == 0:
                    print(f"  Collected {total_steps}/{num_samples} samples ({episode_id} episodes)")
                
    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    
    print(f"✓ Collected {len(data['states'])} samples from {episode_id} episodes")
    return data


# ========================================================================
# 🧩 1. REPRESENTATION GEOMETRY & TOPOLOGY
# ========================================================================

def viz_1a_latent_manifold_projection(data):
    """(a) Latent Manifold Projection using UMAP and t-SNE."""
    print("\n1(a) Creating Latent Manifold Projections...")
    
    latents = data['latents']
    rewards = data['rewards']
    q_values = data['q_values']
    gait_phases = data['gait_phases']
    episode_ids = data['episode_ids']
    
    # Subsample for speed
    n_viz = min(5000, len(latents))
    idx = np.random.choice(len(latents), n_viz, replace=False)
    latents_sub = latents[idx]
    rewards_sub = rewards[idx]
    q_values_sub = q_values[idx]
    gait_sub = gait_phases[idx]
    ep_sub = episode_ids[idx]
    
    # UMAP embedding
    print("  Running UMAP (2-3 minutes)...")
    umap_reducer = UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42, verbose=0)
    umap_emb = umap_reducer.fit_transform(latents_sub)
    print("  ✓ UMAP complete")
    
    # t-SNE embedding
    print("  Running t-SNE (2-3 minutes)...")
    tsne_reducer = TSNE(n_components=2, perplexity=30, random_state=42, verbose=0, n_jobs=1)
    tsne_emb = tsne_reducer.fit_transform(latents_sub)
    print("  ✓ t-SNE complete")
    
    # Create figure with 2 rows, 4 columns (UMAP and t-SNE)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    embeddings = [('UMAP', umap_emb), ('t-SNE', tsne_emb)]
    
    for row, (method_name, emb) in enumerate(embeddings):
        # (i) Color by reward
        sc1 = axes[row, 0].scatter(emb[:, 0], emb[:, 1], c=rewards_sub, 
                                   cmap='RdYlGn', s=10, alpha=0.6, edgecolors='none')
        axes[row, 0].set_title(f'{method_name}: Colored by Reward', fontweight='bold')
        axes[row, 0].set_xlabel(f'{method_name} Dim 1')
        axes[row, 0].set_ylabel(f'{method_name} Dim 2')
        plt.colorbar(sc1, ax=axes[row, 0], label='Reward')
        
        # (ii) Color by Q-value
        sc2 = axes[row, 1].scatter(emb[:, 0], emb[:, 1], c=q_values_sub, 
                                   cmap='viridis', s=10, alpha=0.6, edgecolors='none')
        axes[row, 1].set_title(f'{method_name}: Colored by Q-Value', fontweight='bold')
        axes[row, 1].set_xlabel(f'{method_name} Dim 1')
        axes[row, 1].set_ylabel(f'{method_name} Dim 2')
        plt.colorbar(sc2, ax=axes[row, 1], label='Q(s,a)')
        
        # (iii) Color by gait phase
        sc3 = axes[row, 2].scatter(emb[:, 0], emb[:, 1], c=gait_sub, 
                                   cmap='twilight', s=10, alpha=0.6, edgecolors='none')
        axes[row, 2].set_title(f'{method_name}: Colored by Gait Phase', fontweight='bold')
        axes[row, 2].set_xlabel(f'{method_name} Dim 1')
        axes[row, 2].set_ylabel(f'{method_name} Dim 2')
        plt.colorbar(sc3, ax=axes[row, 2], label='Gait Phase (0-1)')
        
        # (iv) Color by episode
        sc4 = axes[row, 3].scatter(emb[:, 0], emb[:, 1], c=ep_sub, 
                                   cmap='tab10', s=10, alpha=0.6, edgecolors='none')
        axes[row, 3].set_title(f'{method_name}: Colored by Episode', fontweight='bold')
        axes[row, 3].set_xlabel(f'{method_name} Dim 1')
        axes[row, 3].set_ylabel(f'{method_name} Dim 2')
        plt.colorbar(sc4, ax=axes[row, 3], label='Episode ID')
    
    plt.suptitle('Latent Manifold Projections: f_φ(s,a) Embeddings', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    filename = OUTPUT_DIR / '1a_latent_manifold_projection.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {filename}")
    plt.close()


def viz_1b_latent_distance_vs_reward_diff(data):
    """(b) Latent Distance vs. Reward Difference correlation."""
    print("\n1(b) Creating Latent Distance vs. Reward Difference plot...")
    
    latents = data['latents']
    rewards = data['rewards']
    
    # Sample pairs for efficiency
    n_pairs = 2000
    n_samples = len(latents)
    
    idx_i = np.random.choice(n_samples, n_pairs, replace=True)
    idx_j = np.random.choice(n_samples, n_pairs, replace=True)
    
    # Compute latent distances
    latent_dists = np.linalg.norm(latents[idx_i] - latents[idx_j], axis=1)
    
    # Compute reward differences
    reward_diffs = np.abs(rewards[idx_i] - rewards[idx_j])
    
    # Compute correlations
    pearson_corr, pearson_p = pearsonr(latent_dists, reward_diffs)
    spearman_corr, spearman_p = spearmanr(latent_dists, reward_diffs)
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot with hexbin overlay
    axes[0].hexbin(latent_dists, reward_diffs, gridsize=50, cmap='Blues', mincnt=1)
    axes[0].set_xlabel('Latent Distance ||f_φ(s_i,a_i) - f_φ(s_j,a_j)||₂', fontsize=11)
    axes[0].set_ylabel('Reward Difference |r_i - r_j|', fontsize=11)
    axes[0].set_title('Latent Distance vs. Reward Difference', fontweight='bold')
    
    # Add correlation text
    textstr = f'Pearson r = {pearson_corr:.3f} (p={pearson_p:.2e})\n'
    textstr += f'Spearman ρ = {spearman_corr:.3f} (p={spearman_p:.2e})'
    axes[0].text(0.05, 0.95, textstr, transform=axes[0].transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Binned analysis
    n_bins = 20
    bins = np.percentile(latent_dists, np.linspace(0, 100, n_bins+1))
    bin_indices = np.digitize(latent_dists, bins)
    
    bin_means = []
    bin_stds = []
    bin_centers = []
    
    for i in range(1, n_bins+1):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means.append(reward_diffs[mask].mean())
            bin_stds.append(reward_diffs[mask].std())
            bin_centers.append(latent_dists[mask].mean())
    
    axes[1].errorbar(bin_centers, bin_means, yerr=bin_stds, 
                    fmt='o-', capsize=5, markersize=6, linewidth=2)
    axes[1].set_xlabel('Latent Distance (binned)', fontsize=11)
    axes[1].set_ylabel('Mean Reward Difference', fontsize=11)
    axes[1].set_title('Binned Relationship (Mean ± Std)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Testing if Learned Metric Respects Reward Structure', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = OUTPUT_DIR / '1b_latent_distance_vs_reward_diff.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {filename}")
    print(f"     Pearson correlation: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}")
    plt.close()


def viz_1c_distance_heatmap(data):
    """(c) Pairwise Distance Heatmap with hierarchical clustering."""
    print("\n1(c) Creating Distance Heatmap...")
    
    latents = data['latents']
    rewards = data['rewards']
    
    # Sample for computational efficiency
    n_viz = 200
    idx = np.random.choice(len(latents), n_viz, replace=False)
    latents_sub = latents[idx]
    rewards_sub = rewards[idx]
    
    # Compute pairwise distances
    dist_matrix = squareform(pdist(latents_sub, metric='euclidean'))
    
    # Hierarchical clustering
    linkage_matrix = linkage(pdist(latents_sub), method='ward')
    order = leaves_list(linkage_matrix)
    
    # Reorder matrix
    dist_matrix_ordered = dist_matrix[order, :][:, order]
    rewards_ordered = rewards_sub[order]
    
    # Create figure
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.05], height_ratios=[0.3, 1],
                         hspace=0.05, wspace=0.05)
    
    # Dendrogram
    ax_dend = fig.add_subplot(gs[0, 0])
    dendrogram(linkage_matrix, ax=ax_dend, no_labels=True)
    ax_dend.set_title('Hierarchical Clustering Dendrogram', fontweight='bold')
    ax_dend.set_ylabel('Distance')
    ax_dend.set_xticks([])
    
    # Heatmap
    ax_heat = fig.add_subplot(gs[1, 0])
    im = ax_heat.imshow(dist_matrix_ordered, cmap='viridis', aspect='auto', interpolation='nearest')
    ax_heat.set_xlabel('Sample Index (clustered)', fontsize=11)
    ax_heat.set_ylabel('Sample Index (clustered)', fontsize=11)
    ax_heat.set_title('Latent Distance Matrix (Hierarchically Clustered)', fontweight='bold')
    
    # Colorbar
    ax_cbar = fig.add_subplot(gs[1, 1])
    plt.colorbar(im, cax=ax_cbar, label='Latent Distance')
    
    plt.suptitle('Pairwise Similarity Matrix in Latent Space', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    filename = OUTPUT_DIR / '1c_distance_heatmap.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {filename}")
    plt.close()


def viz_1d_temporal_consistency(data):
    """(d) Temporal Consistency Map - latent displacement over time."""
    print("\n1(d) Creating Temporal Consistency Map...")
    
    latents = data['latents']
    latents_next = data['latents_next']
    rewards = data['rewards']
    episode_ids = data['episode_ids']
    
    # Compute latent displacement
    latent_displacement = np.linalg.norm(latents_next - latents, axis=1)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (a) Displacement over time for first few episodes
    max_ep_viz = 5
    for ep in range(min(max_ep_viz, episode_ids.max() + 1)):
        mask = episode_ids == ep
        steps = np.arange(mask.sum())
        axes[0, 0].plot(steps, latent_displacement[mask], alpha=0.7, label=f'Episode {ep}')
    
    axes[0, 0].set_xlabel('Time Step', fontsize=11)
    axes[0, 0].set_ylabel('||f_φ(s_{t+1}) - f_φ(s_t)||', fontsize=11)
    axes[0, 0].set_title('Latent Displacement Over Time', fontweight='bold')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    # (b) Displacement vs. reward
    axes[0, 1].hexbin(rewards, latent_displacement, gridsize=40, cmap='YlOrRd', mincnt=1)
    axes[0, 1].set_xlabel('Reward', fontsize=11)
    axes[0, 1].set_ylabel('Latent Displacement', fontsize=11)
    axes[0, 1].set_title('Displacement vs. Reward', fontweight='bold')
    
    # (c) Distribution of displacements
    axes[1, 0].hist(latent_displacement, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(latent_displacement.mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean = {latent_displacement.mean():.3f}')
    axes[1, 0].axvline(np.median(latent_displacement), color='blue', linestyle='--', 
                      linewidth=2, label=f'Median = {np.median(latent_displacement):.3f}')
    axes[1, 0].set_xlabel('Latent Displacement', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Distribution of Temporal Displacements', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # (d) Smoothness per episode
    ep_smoothness = []
    ep_ids = []
    for ep in range(episode_ids.max() + 1):
        mask = episode_ids == ep
        if mask.sum() > 0:
            ep_smoothness.append(latent_displacement[mask].mean())
            ep_ids.append(ep)
    
    axes[1, 1].bar(ep_ids, ep_smoothness, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Episode ID', fontsize=11)
    axes[1, 1].set_ylabel('Mean Latent Displacement', fontsize=11)
    axes[1, 1].set_title('Temporal Smoothness per Episode', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Temporal Consistency in Latent Space', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = OUTPUT_DIR / '1d_temporal_consistency.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {filename}")
    plt.close()


# ========================================================================
# 🎯 2. LINKING REPRESENTATION TO POLICY/VALUE BEHAVIOR
# ========================================================================

def viz_2a_action_clustering(data):
    """(a) Action Clustering in Latent Space."""
    print("\n2(a) Creating Action Clustering visualization...")
    
    latents = data['latents']
    actions = data['actions']
    
    # Compute action properties
    action_magnitude = np.linalg.norm(actions, axis=1)
    action_angle = np.arctan2(actions[:, 1], actions[:, 0])  # Using first 2 dims
    
    # Subsample and project with PCA
    n_viz = min(5000, len(latents))
    idx = np.random.choice(len(latents), n_viz, replace=False)
    
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents[idx])
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color by action magnitude
    sc1 = axes[0].scatter(latents_2d[:, 0], latents_2d[:, 1], 
                         c=action_magnitude[idx], cmap='plasma', 
                         s=20, alpha=0.6, edgecolors='none')
    axes[0].set_xlabel('Latent PC1', fontsize=11)
    axes[0].set_ylabel('Latent PC2', fontsize=11)
    axes[0].set_title('Latent Space Colored by Action Magnitude', fontweight='bold')
    plt.colorbar(sc1, ax=axes[0], label='||a||')
    
    # Color by action angle
    sc2 = axes[1].scatter(latents_2d[:, 0], latents_2d[:, 1], 
                         c=action_angle[idx], cmap='hsv', 
                         s=20, alpha=0.6, edgecolors='none')
    axes[1].set_xlabel('Latent PC1', fontsize=11)
    axes[1].set_ylabel('Latent PC2', fontsize=11)
    axes[1].set_title('Latent Space Colored by Action Direction', fontweight='bold')
    plt.colorbar(sc2, ax=axes[1], label='Action Angle (rad)')
    
    # Add explained variance
    var_explained = pca.explained_variance_ratio_.sum()
    fig.text(0.5, 0.02, f'PCA explains {var_explained:.1%} of latent variance', 
            ha='center', fontsize=10, style='italic')
    
    plt.suptitle('Action Clustering in Latent Space', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = OUTPUT_DIR / '2a_action_clustering.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {filename}")
    plt.close()


def viz_2b_value_gradient_field(data, actor, rep_trunk, qnet, normalization, env):
    """(b) Value Gradient Direction in Latent Space."""
    print("\n2(b) Creating Value Gradient Field...")
    
    # Sample states and compute value gradients
    n_grid = 20
    states = data['states']
    
    # Get latent embeddings
    idx = np.random.choice(len(states), min(1000, len(states)), replace=False)
    states_sample = states[idx]
    
    # Project to 2D with PCA
    latents_all = data['latents'][idx]
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents_all)
    
    # Compute gradients in latent space (approximate with finite differences)
    print("  Computing value gradients...")
    gradients = []
    q_vals = []
    
    with torch.enable_grad():
        for i in range(0, min(200, len(idx)), 10):  # Sample subset for speed
            state = states_sample[i]
            state_norm = normalization.normalize(torch.FloatTensor(state).to(DEVICE))
            state_norm.requires_grad = True
            
            # Get action
            mu, _ = actor(state_norm.unsqueeze(0))
            action = torch.tanh(mu)
            
            # Compute Q-value
            q1, q2 = qnet(state_norm.unsqueeze(0), action)
            q_val = torch.min(q1, q2)
            q_vals.append(q_val.item())
            
            # Compute gradient w.r.t. state (approximate latent gradient)
            q_val.backward()
            grad = state_norm.grad.cpu().numpy()
            gradients.append(grad[:2])  # Use first 2 dims
    
    gradients = np.array(gradients)
    q_vals = np.array(q_vals)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Background scatter
    sc = ax.scatter(latents_2d[:, 0], latents_2d[:, 1], 
                   c=data['q_values'][idx], cmap='viridis', 
                   s=30, alpha=0.5, edgecolors='none')
    
    # Quiver plot (subsample for clarity)
    step = max(1, len(gradients) // 50)
    grad_norm = np.linalg.norm(gradients, axis=1, keepdims=True) + 1e-8
    gradients_normalized = gradients / grad_norm
    
    # Map gradients to latent 2D space (approximate)
    ax.quiver(latents_2d[::10, 0][:len(gradients)][::step], 
             latents_2d[::10, 1][:len(gradients)][::step],
             gradients_normalized[::step, 0], 
             gradients_normalized[::step, 1],
             q_vals[::step], cmap='RdYlGn', scale=20, width=0.003, alpha=0.8)
    
    ax.set_xlabel('Latent PC1', fontsize=12)
    ax.set_ylabel('Latent PC2', fontsize=12)
    ax.set_title('Value Gradient Direction in Latent Space\n(Arrows show ∇Q direction)', 
                fontsize=13, fontweight='bold')
    plt.colorbar(sc, ax=ax, label='Q-value')
    
    plt.tight_layout()
    filename = OUTPUT_DIR / '2b_value_gradient_field.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {filename}")
    plt.close()


def viz_2c_policy_coverage_ellipses(data):
    """(c) Policy Coverage Ellipses showing representation space exploration."""
    print("\n2(c) Creating Policy Coverage Ellipses...")
    
    latents = data['latents']
    episode_ids = data['episode_ids']
    
    # Project to 2D
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all points
    ax.scatter(latents_2d[:, 0], latents_2d[:, 1], 
              c=episode_ids, cmap='tab10', s=10, alpha=0.3, edgecolors='none')
    
    # Compute and plot ellipses for each episode
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for ep in range(min(10, episode_ids.max() + 1)):
        mask = episode_ids == ep
        if mask.sum() < 3:
            continue
        
        points = latents_2d[mask]
        mean = points.mean(axis=0)
        cov = np.cov(points.T)
        
        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * np.sqrt(eigenvalues) * 2  # 2-sigma ellipse
        
        ellipse = Ellipse(mean, width, height, angle=angle, 
                         facecolor=colors[ep], alpha=0.2, edgecolor=colors[ep], linewidth=2)
        ax.add_patch(ellipse)
        ax.plot(mean[0], mean[1], 'o', color=colors[ep], markersize=8, 
               markeredgecolor='black', markeredgewidth=1, label=f'Episode {ep}')
    
    ax.set_xlabel('Latent PC1', fontsize=12)
    ax.set_ylabel('Latent PC2', fontsize=12)
    ax.set_title('Policy Coverage in Latent Space\n(2σ covariance ellipses per episode)', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = OUTPUT_DIR / '2c_policy_coverage_ellipses.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {filename}")
    plt.close()


def viz_2d_smoothness_vs_td_error(data, actor, qnet, normalization):
    """(d) Representation Smoothness vs. TD Error."""
    print("\n2(d) Creating Smoothness vs. TD Error plot...")
    
    states = data['states']
    states_next = data['states_next']
    actions = data['actions']
    rewards = data['rewards']
    latents = data['latents']
    latents_next = data['latents_next']
    
    # Compute smoothness
    smoothness = np.linalg.norm(latents_next - latents, axis=1)
    
    # Compute TD errors
    print("  Computing TD errors...")
    td_errors = []
    
    with torch.no_grad():
        for i in range(0, len(states), 100):  # Batch processing
            batch_size = min(100, len(states) - i)
            
            # Current state-action
            s = torch.FloatTensor(states[i:i+batch_size]).to(DEVICE)
            a = torch.FloatTensor(actions[i:i+batch_size]).to(DEVICE)
            s_norm = torch.stack([normalization.normalize(s[j]) for j in range(len(s))])
            
            # Next state-action
            s_next = torch.FloatTensor(states_next[i:i+batch_size]).to(DEVICE)
            s_next_norm = torch.stack([normalization.normalize(s_next[j]) for j in range(len(s_next))])
            mu_next, _ = actor(s_next_norm)
            a_next = torch.tanh(mu_next)
            
            # Q-values
            q1, q2 = qnet(s_norm, a)
            q = torch.min(q1, q2).squeeze()
            
            q1_next, q2_next = qnet(s_next_norm, a_next)
            q_next = torch.min(q1_next, q2_next).squeeze()
            
            # TD error
            r = torch.FloatTensor(rewards[i:i+batch_size]).to(DEVICE)
            gamma = 0.99
            td = torch.abs(q - (r + gamma * q_next))
            
            td_errors.extend(td.cpu().numpy())
    
    td_errors = np.array(td_errors)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].hexbin(smoothness, td_errors, gridsize=50, cmap='YlOrRd', mincnt=1)
    axes[0].set_xlabel('Representation Smoothness ||f_φ(s\') - f_φ(s)||', fontsize=11)
    axes[0].set_ylabel('TD Error |Q(s,a) - (r + γQ(s\',a\'))|', fontsize=11)
    axes[0].set_title('Smoothness vs. TD Error', fontweight='bold')
    
    # Compute correlation
    corr, p_val = pearsonr(smoothness, td_errors)
    textstr = f'Correlation: {corr:.3f}\np-value: {p_val:.2e}'
    axes[0].text(0.05, 0.95, textstr, transform=axes[0].transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Binned analysis
    n_bins = 15
    bins = np.percentile(smoothness, np.linspace(0, 100, n_bins+1))
    bin_indices = np.digitize(smoothness, bins)
    
    bin_means = []
    bin_stds = []
    bin_centers = []
    
    for i in range(1, n_bins+1):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means.append(td_errors[mask].mean())
            bin_stds.append(td_errors[mask].std())
            bin_centers.append(smoothness[mask].mean())
    
    axes[1].errorbar(bin_centers, bin_means, yerr=bin_stds, 
                    fmt='o-', capsize=5, markersize=6, linewidth=2, color='darkred')
    axes[1].set_xlabel('Smoothness (binned)', fontsize=11)
    axes[1].set_ylabel('Mean TD Error', fontsize=11)
    axes[1].set_title('Binned Relationship', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Representation Smoothness vs. TD Error', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = OUTPUT_DIR / '2d_smoothness_vs_td_error.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {filename}")
    print(f"     Correlation: {corr:.3f} (p={p_val:.2e})")
    plt.close()


# ========================================================================
# 🧬 4. ADVANCED GEOMETRIC VISUALIZATIONS
# ========================================================================

def viz_4a_latent_curvature_field(data):
    """(a) Latent Curvature Field - sensitivity analysis."""
    print("\n4(a) Creating Latent Curvature Field...")
    
    latents = data['latents']
    states = data['states']
    
    # Project to 2D for visualization
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)
    
    # Compute local variance as proxy for curvature
    from sklearn.neighbors import NearestNeighbors
    
    nbrs = NearestNeighbors(n_neighbors=10).fit(latents)
    distances, indices = nbrs.kneighbors(latents)
    
    # Local variance (curvature proxy)
    curvature = distances[:, 1:].std(axis=1)  # Exclude self
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot colored by curvature
    sc = axes[0].scatter(latents_2d[:, 0], latents_2d[:, 1], 
                        c=curvature, cmap='hot', s=20, alpha=0.6, edgecolors='none')
    axes[0].set_xlabel('Latent PC1', fontsize=11)
    axes[0].set_ylabel('Latent PC2', fontsize=11)
    axes[0].set_title('Latent Curvature Field\n(Local variance of k-NN distances)', 
                     fontweight='bold')
    plt.colorbar(sc, ax=axes[0], label='Curvature (sensitivity)')
    
    # Histogram of curvature
    axes[1].hist(curvature, bins=50, alpha=0.7, edgecolor='black', color='coral')
    axes[1].axvline(curvature.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean = {curvature.mean():.3f}')
    axes[1].set_xlabel('Curvature', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Distribution of Curvature', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Latent Space Curvature Analysis', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = OUTPUT_DIR / '4a_latent_curvature_field.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {filename}")
    plt.close()


def viz_4b_latent_transition_graph(data):
    """(b) Latent Transition Graph - k-NN network."""
    print("\n4(b) Creating Latent Transition Graph...")
    
    latents = data['latents']
    rewards = data['rewards']
    
    # Subsample for visualization
    n_viz = 300
    idx = np.random.choice(len(latents), n_viz, replace=False)
    latents_sub = latents[idx]
    rewards_sub = rewards[idx]
    
    # Build k-NN graph
    from sklearn.neighbors import NearestNeighbors
    k = 5
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(latents_sub)
    distances, indices = nbrs.kneighbors(latents_sub)
    
    # Create networkx graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(n_viz):
        G.add_node(i, reward=rewards_sub[i])
    
    # Add edges
    reward_deltas = []
    for i in range(n_viz):
        for j in range(1, k+1):  # Skip self
            neighbor = indices[i, j]
            reward_delta = abs(rewards_sub[i] - rewards_sub[neighbor])
            G.add_edge(i, neighbor, weight=distances[i, j], reward_delta=reward_delta)
            reward_deltas.append(reward_delta)
    
    # Layout using spring layout
    print("  Computing graph layout...")
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw edges colored by reward delta
    edges = G.edges()
    edge_colors = [G[u][v]['reward_delta'] for u, v in edges]
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color=edge_colors, 
                          edge_cmap=plt.cm.RdYlGn_r, width=0.5, ax=ax)
    
    # Draw nodes colored by reward, sized by degree
    node_colors = [G.nodes[i]['reward'] for i in range(n_viz)]
    node_sizes = [G.degree(i) * 10 for i in range(n_viz)]
    
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                   node_size=node_sizes, cmap='RdYlGn', 
                                   alpha=0.8, edgecolors='black', linewidths=0.5, ax=ax)
    
    ax.set_title(f'Latent Transition Graph (k-NN, k={k})\nNode color=reward, size=degree, edge=reward delta', 
                fontsize=13, fontweight='bold')
    ax.axis('off')
    
    # Colorbars
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', 
                               norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Node Reward', shrink=0.6)
    
    plt.tight_layout()
    filename = OUTPUT_DIR / '4b_latent_transition_graph.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {filename}")
    plt.close()


def viz_4c_reward_aware_cosine_similarity(data):
    """(c) Reward-Aware Cosine Similarity Map."""
    print("\n4(c) Creating Reward-Aware Cosine Similarity Map...")
    
    latents = data['latents']
    rewards = data['rewards']
    
    # Sample for computational efficiency
    n_viz = 200
    idx = np.random.choice(len(latents), n_viz, replace=False)
    latents_sub = latents[idx]
    rewards_sub = rewards[idx]
    
    # Normalize latents for cosine similarity
    latents_norm = latents_sub / (np.linalg.norm(latents_sub, axis=1, keepdims=True) + 1e-8)
    
    # Compute cosine similarity matrix
    cosine_sim = latents_norm @ latents_norm.T
    
    # Compute reward-aware similarity: cos(z_i, z_j) * sign(r_i * r_j)
    reward_signs = np.sign(np.outer(rewards_sub, rewards_sub))
    reward_aware_sim = cosine_sim * reward_signs
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Standard cosine similarity
    im1 = axes[0].imshow(cosine_sim, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[0].set_title('Standard Cosine Similarity', fontweight='bold')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Sample Index')
    plt.colorbar(im1, ax=axes[0], label='cos(z_i, z_j)')
    
    # Reward-aware cosine similarity
    im2 = axes[1].imshow(reward_aware_sim, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[1].set_title('Reward-Aware Cosine Similarity', fontweight='bold')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Sample Index')
    plt.colorbar(im2, ax=axes[1], label='cos(z_i, z_j) × sign(r_i·r_j)')
    
    plt.suptitle('Reward-Aware Similarity Analysis', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = OUTPUT_DIR / '4c_reward_aware_cosine_similarity.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {filename}")
    plt.close()


# ========================================================================
# 🧠 5. PAPER-READY 3-PANEL FIGURE
# ========================================================================

def viz_5_paper_ready_trio(data):
    """Create publication-ready 3-panel figure."""
    print("\n5. Creating Paper-Ready 3-Panel Figure...")
    
    latents = data['latents']
    rewards = data['rewards']
    latents_next = data['latents_next']
    episode_ids = data['episode_ids']
    
    # Subsample
    n_viz = min(3000, len(latents))
    idx = np.random.choice(len(latents), n_viz, replace=False)
    
    # Panel 1: UMAP manifold colored by reward
    print("  Panel 1: UMAP manifold...")
    umap_reducer = UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42, verbose=0)
    umap_emb = umap_reducer.fit_transform(latents[idx])
    
    # Panel 2: Latent distance vs reward difference
    print("  Panel 2: Distance-reward correlation...")
    n_pairs = 1000
    idx_i = np.random.choice(len(latents), n_pairs, replace=True)
    idx_j = np.random.choice(len(latents), n_pairs, replace=True)
    latent_dists = np.linalg.norm(latents[idx_i] - latents[idx_j], axis=1)
    reward_diffs = np.abs(rewards[idx_i] - rewards[idx_j])
    
    # Panel 3: Temporal smoothness for one episode
    print("  Panel 3: Temporal smoothness...")
    ep_mask = episode_ids == 0
    ep_displacement = np.linalg.norm(latents_next[ep_mask] - latents[ep_mask], axis=1)
    ep_rewards = rewards[ep_mask]
    ep_steps = np.arange(len(ep_displacement))
    
    # Create figure
    fig = plt.figure(figsize=(18, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    
    # Panel A: Latent Manifold
    ax1 = fig.add_subplot(gs[0])
    sc1 = ax1.scatter(umap_emb[:, 0], umap_emb[:, 1], 
                     c=rewards[idx], cmap='RdYlGn', s=15, alpha=0.7, edgecolors='none')
    ax1.set_xlabel('UMAP Dimension 1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('UMAP Dimension 2', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Latent Manifold Organized by Reward', fontsize=13, fontweight='bold')
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label('Reward', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.2)
    
    # Panel B: Distance vs Reward
    ax2 = fig.add_subplot(gs[1])
    ax2.hexbin(latent_dists, reward_diffs, gridsize=40, cmap='Blues', mincnt=1)
    ax2.set_xlabel('Latent Distance ||f_φ(s_i,a_i) - f_φ(s_j,a_j)||', 
                  fontsize=11, fontweight='bold')
    ax2.set_ylabel('Reward Difference |r_i - r_j|', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Metric Respects Reward Structure', fontsize=13, fontweight='bold')
    
    # Add correlation
    corr, _ = spearmanr(latent_dists, reward_diffs)
    ax2.text(0.95, 0.95, f'ρ = {corr:.3f}', transform=ax2.transAxes, 
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
    ax2.grid(True, alpha=0.2)
    
    # Panel C: Temporal Smoothness
    ax3 = fig.add_subplot(gs[2])
    
    # Twin axis for displacement and reward
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(ep_steps, ep_displacement, 'b-', linewidth=2, alpha=0.8, label='Latent Displacement')
    ax3.fill_between(ep_steps, 0, ep_displacement, alpha=0.3, color='blue')
    ax3.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax3.set_ylabel('||f_φ(s_{t+1}) - f_φ(s_t)||', fontsize=11, fontweight='bold', color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')
    
    line2 = ax3_twin.plot(ep_steps, ep_rewards, 'g-', linewidth=1.5, alpha=0.6, label='Reward')
    ax3_twin.set_ylabel('Reward', fontsize=11, fontweight='bold', color='green')
    ax3_twin.tick_params(axis='y', labelcolor='green')
    
    ax3.set_title('(C) Temporal Smoothness Along Trajectory', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.2)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right', fontsize=9)
    
    plt.suptitle('DistRL Representation: Organizing State-Action Pairs by Utility', 
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    filename = OUTPUT_DIR / '5_paper_ready_trio.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {filename}")
    plt.close()


def main():
    """Main execution pipeline."""
    print("=" * 80)
    print("Advanced Representation Geometry & Topology Visualizations")
    print("=" * 80)
    
    # Load model
    env, actor, rep_trunk, qnet, normalization = load_model_and_env()
    
    # Collect comprehensive data
    data = collect_trajectory_data(env, actor, rep_trunk, qnet, normalization,
                                   num_samples=NUM_SAMPLES, num_episodes=NUM_EPISODES)
    
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)
    
    # 🧩 1. Representation Geometry & Topology
    print("\n" + "=" * 80)
    print("🧩 Section 1: Representation Geometry & Topology")
    print("=" * 80)
    viz_1a_latent_manifold_projection(data)
    viz_1b_latent_distance_vs_reward_diff(data)
    viz_1c_distance_heatmap(data)
    viz_1d_temporal_consistency(data)
    
    # 🎯 2. Linking Representation to Policy/Value
    print("\n" + "=" * 80)
    print("🎯 Section 2: Linking Representation to Policy/Value")
    print("=" * 80)
    viz_2a_action_clustering(data)
    viz_2b_value_gradient_field(data, actor, rep_trunk, qnet, normalization, env)
    viz_2c_policy_coverage_ellipses(data)
    viz_2d_smoothness_vs_td_error(data, actor, qnet, normalization)
    
    # 🧬 4. Advanced Geometric Visualizations
    print("\n" + "=" * 80)
    print("🧬 Section 4: Advanced Geometric Visualizations")
    print("=" * 80)
    viz_4a_latent_curvature_field(data)
    viz_4b_latent_transition_graph(data)
    viz_4c_reward_aware_cosine_similarity(data)
    
    # 🧠 5. Paper-Ready Figure
    print("\n" + "=" * 80)
    print("🧠 Section 5: Paper-Ready 3-Panel Figure")
    print("=" * 80)
    viz_5_paper_ready_trio(data)
    
    env.close()
    
    print("\n" + "=" * 80)
    print("✓ All visualizations complete!")
    print(f"✓ Saved to: {OUTPUT_DIR}")
    print("=" * 80)
    
    print("\n📊 Generated Visualizations Summary:")
    print("\n🧩 1. Representation Geometry & Topology:")
    print("  • 1a: Latent manifold projections (UMAP/t-SNE)")
    print("  • 1b: Latent distance vs. reward difference")
    print("  • 1c: Distance heatmap with hierarchical clustering")
    print("  • 1d: Temporal consistency maps")
    
    print("\n🎯 2. Linking Representation to Policy/Value:")
    print("  • 2a: Action clustering in latent space")
    print("  • 2b: Value gradient direction field")
    print("  • 2c: Policy coverage ellipses")
    print("  • 2d: Smoothness vs. TD error correlation")
    
    print("\n🧬 4. Advanced Geometric Visualizations:")
    print("  • 4a: Latent curvature field")
    print("  • 4b: Latent transition k-NN graph")
    print("  • 4c: Reward-aware cosine similarity")
    
    print("\n🧠 5. Paper-Ready Figure:")
    print("  • 3-panel: Manifold + Distance-Reward + Temporal Smoothness")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
