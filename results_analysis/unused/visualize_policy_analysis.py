"""
Advanced Policy Analysis Visualizations for DistRL Walker2d
============================================================

Implements five creative visualization techniques:
(a) Action Landscape over State Manifold - UMAP/t-SNE embedding colored by action properties
(b) Phase Portrait of Joint Torques - Joint angle vs. torque cyclic diagrams
(c) Policy Vector Field in Reduced Coordinates - 2D control field with arrows
(d) Latent Action Embedding Sphere - Action mean vectors on hypersphere
(e) Temporal Evolution Heatmap - Action amplitudes over time per joint

Usage:
    python results_analysis/visualize_policy_analysis.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gymnasium as gym
from sklearn.manifold import TSNE
from umap import UMAP
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*threadpoolctl.*')

# Import DistRL components
import sys
sys.path.append('.')
from dist_rl.models import DistanceTrunk, GaussianActor
from dist_rl.utils import RunningMeanStd

# Configuration
DEVICE = 'cpu'
MODEL_PATH = './saved_models/best.pt'
ENV_NAME = 'Walker2d-v5'
NUM_SAMPLES = 10000  # For manifold visualizations
NUM_EPISODES = 5      # For temporal analysis
MAX_STEPS = 200       # Steps per episode for temporal heatmap
OUTPUT_DIR = Path('results_analysis/plots/policy_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Walker2d joint names (6 actions)
JOINT_NAMES = ['thigh_joint', 'leg_joint', 'foot_joint', 
               'thigh_left_joint', 'leg_left_joint', 'foot_left_joint']

# State components (17 dimensions)
STATE_COMPONENTS = ['z', 'angle', 'thigh', 'leg', 'foot', 'thigh_left', 'leg_left', 'foot_left',
                    'velocity_x', 'velocity_z', 'ang_velocity', 'thigh_vel', 'leg_vel', 
                    'foot_vel', 'thigh_left_vel', 'leg_left_vel', 'foot_left_vel']


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
    
    # Load weights
    actor.load_state_dict(checkpoint['actor'])
    rep_trunk.load_state_dict(checkpoint['rep_trunk'])
    
    # Load normalization
    normalization = RunningMeanStd(obs_dim, device=DEVICE)
    normalization.load_state_dict(checkpoint['normalization'])
    
    # Set to eval mode
    actor.eval()
    rep_trunk.eval()
    
    print(f"✓ Model loaded from {MODEL_PATH}")
    print(f"✓ Environment: {ENV_NAME}")
    print(f"✓ Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    return env, actor, rep_trunk, normalization


def collect_samples(env, actor, normalization, num_samples=10000):
    """Collect random state-action samples from policy rollouts."""
    print(f"\nCollecting {num_samples} samples...")
    
    states = []
    actions_mean = []
    actions_std = []
    actions_sampled = []
    features = []
    rewards = []
    joint_angles = []
    joint_velocities = []
    
    obs, _ = env.reset()
    step_count = 0
    episode_count = 0
    
    with torch.no_grad():
        while len(states) < num_samples:
            # Normalize observation
            obs_norm = normalization.normalize(torch.FloatTensor(obs).to(DEVICE))
            
            # Get policy features (penultimate layer)
            obs_tensor = obs_norm.unsqueeze(0)
            mu, log_std = actor.forward_get_features(obs_tensor)
            feat = actor.last_features  # Saved by forward_get_features
            
            # Get action statistics
            action_mean = mu.cpu().numpy()[0]
            action_std = torch.exp(log_std).cpu().numpy()[0]
            action_sampled = torch.tanh(mu).cpu().numpy()[0]  # Deterministic
            
            # Store data
            states.append(obs)
            actions_mean.append(action_mean)
            actions_std.append(action_std)
            actions_sampled.append(action_sampled)
            features.append(feat.cpu().numpy()[0])
            
            # Extract joint angles and velocities (Walker2d specific)
            # Obs: [z, angle, 6 joint angles, velocity_x, velocity_z, ang_vel, 6 joint velocities]
            joint_angles.append(obs[2:8])  # 6 joint angles
            joint_velocities.append(obs[11:17])  # 6 joint velocities
            
            # Step environment
            action_env = ((action_sampled + 1) / 2) * (env.action_space.high - env.action_space.low) + env.action_space.low
            obs, reward, terminated, truncated, _ = env.step(action_env)
            rewards.append(reward)
            
            step_count += 1
            
            if terminated or truncated:
                obs, _ = env.reset()
                episode_count += 1
                step_count = 0
            
            if len(states) % 1000 == 0:
                print(f"  Collected {len(states)}/{num_samples} samples ({episode_count} episodes)")
    
    print(f"✓ Collected {len(states)} samples from {episode_count} episodes")
    
    return {
        'states': np.array(states),
        'actions_mean': np.array(actions_mean),
        'actions_std': np.array(actions_std),
        'actions_sampled': np.array(actions_sampled),
        'features': np.array(features),
        'rewards': np.array(rewards),
        'joint_angles': np.array(joint_angles),
        'joint_velocities': np.array(joint_velocities)
    }


def visualize_action_landscape_manifold(data, use_umap=True):
    """
    (a) Action Landscape over State Manifold
    Embed states using UMAP/t-SNE, color by action properties.
    """
    print("\n(a) Creating Action Landscape over State Manifold...")
    
    features = data['features']
    actions = data['actions_sampled']
    rewards = data['rewards']
    
    # Compute action magnitude
    action_magnitude = np.linalg.norm(actions, axis=1)
    
    # Compute dominant action direction (angle in action space)
    action_angle = np.arctan2(actions[:, 1], actions[:, 0])  # Using first 2 dims
    
    # Reduce dimensionality
    if use_umap:
        print("  Running UMAP (this may take 2-3 minutes)...")
        reducer = UMAP(n_neighbors=50, min_dist=0.1, n_components=2, random_state=42, verbose=0)
        embedding = reducer.fit_transform(features)
        method_name = "UMAP"
    else:
        print("  Running t-SNE (this may take 2-3 minutes)...")
        reducer = TSNE(n_components=2, perplexity=50, random_state=42, verbose=0, n_jobs=1)
        embedding = reducer.fit_transform(features)
        method_name = "t-SNE"
    
    print(f"  ✓ {method_name} complete!")
    
    # Create 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # (a1) Color by action magnitude
    sc1 = axes[0].scatter(embedding[:, 0], embedding[:, 1], 
                         c=action_magnitude, cmap='viridis', 
                         s=10, alpha=0.6, edgecolors='none')
    axes[0].set_title(f'Action Magnitude on {method_name} State Manifold', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(f'{method_name} Dimension 1')
    axes[0].set_ylabel(f'{method_name} Dimension 2')
    plt.colorbar(sc1, ax=axes[0], label='Action Magnitude')
    
    # (a2) Color by action angle (cyclic)
    sc2 = axes[1].scatter(embedding[:, 0], embedding[:, 1], 
                         c=action_angle, cmap='hsv', 
                         s=10, alpha=0.6, edgecolors='none')
    axes[1].set_title('Action Direction (Circular Hue)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel(f'{method_name} Dimension 1')
    axes[1].set_ylabel(f'{method_name} Dimension 2')
    plt.colorbar(sc2, ax=axes[1], label='Action Angle (rad)')
    
    # (a3) Color by reward
    sc3 = axes[2].scatter(embedding[:, 0], embedding[:, 1], 
                         c=rewards, cmap='RdYlGn', 
                         s=10, alpha=0.6, edgecolors='none')
    axes[2].set_title('Reward Distribution on State Manifold', fontsize=12, fontweight='bold')
    axes[2].set_xlabel(f'{method_name} Dimension 1')
    axes[2].set_ylabel(f'{method_name} Dimension 2')
    plt.colorbar(sc3, ax=axes[2], label='Reward')
    
    plt.tight_layout()
    filename = OUTPUT_DIR / f'a_action_landscape_{method_name.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {filename}")
    plt.close()


def visualize_phase_portraits(data):
    """
    (b) Phase Portrait of Joint Torques
    Plot θ (angle) vs. τ (torque) for each joint, colored by time/phase.
    """
    print("\n(b) Creating Phase Portrait of Joint Torques...")
    
    joint_angles = data['joint_angles']
    actions = data['actions_sampled']  # Actions correspond to joint torques
    
    # Create phase for coloring (0-1 based on sequence)
    n_samples = len(joint_angles)
    phase = np.linspace(0, 1, n_samples)
    
    # Plot 2x3 grid for 6 joints
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, joint_name in enumerate(JOINT_NAMES):
        ax = axes[i]
        
        # Plot angle vs. torque colored by phase
        scatter = ax.scatter(joint_angles[:, i], actions[:, i], 
                           c=phase, cmap='twilight', 
                           s=5, alpha=0.5, edgecolors='none')
        
        ax.set_xlabel(f'Joint Angle (rad)', fontsize=10)
        ax.set_ylabel(f'Joint Torque (normalized)', fontsize=10)
        ax.set_title(f'{joint_name}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for first subplot
        if i == 0:
            plt.colorbar(scatter, ax=ax, label='Gait Phase (0-1)')
    
    plt.suptitle('Phase Portraits: Joint Angle vs. Torque', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    filename = OUTPUT_DIR / 'b_phase_portraits.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {filename}")
    plt.close()


def visualize_policy_vector_field(env, actor, normalization):
    """
    (c) Policy Vector Field in Reduced Coordinates
    2D slice of state space with arrows showing action directions.
    """
    print("\n(c) Creating Policy Vector Field in Reduced Coordinates...")
    
    # Choose 2D slice: torso angle (state[1]) vs. forward velocity (state[8])
    angle_range = np.linspace(-0.5, 0.5, 20)
    vel_range = np.linspace(-1, 4, 20)
    
    # Create grid
    angle_grid, vel_grid = np.meshgrid(angle_range, vel_range)
    
    # Sample a reference state
    obs, _ = env.reset()
    reference_state = obs.copy()
    
    # Compute mean action for each grid point
    action_x = np.zeros_like(angle_grid)
    action_y = np.zeros_like(vel_grid)
    action_magnitude = np.zeros_like(angle_grid)
    
    with torch.no_grad():
        for i in range(len(angle_range)):
            for j in range(len(vel_range)):
                # Modify reference state
                state = reference_state.copy()
                state[1] = angle_grid[j, i]  # torso angle
                state[8] = vel_grid[j, i]    # forward velocity
                
                # Get policy action
                state_norm = normalization.normalize(torch.FloatTensor(state).to(DEVICE))
                mu, _ = actor(state_norm.unsqueeze(0))
                action = torch.tanh(mu).cpu().numpy()[0]
                
                # Store primary action components for arrows
                action_x[j, i] = action[0]  # First action dimension
                action_y[j, i] = action[1]  # Second action dimension
                action_magnitude[j, i] = np.linalg.norm(action)
    
    # Plot vector field
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Background colored by action magnitude
    im = ax.contourf(angle_grid, vel_grid, action_magnitude, 
                     levels=20, cmap='YlOrRd', alpha=0.6)
    plt.colorbar(im, ax=ax, label='Action Magnitude')
    
    # Quiver plot for action directions
    ax.quiver(angle_grid, vel_grid, action_x, action_y, 
             action_magnitude, cmap='viridis',
             scale=20, width=0.003, alpha=0.8)
    
    ax.set_xlabel('Torso Angle (rad)', fontsize=12)
    ax.set_ylabel('Forward Velocity (m/s)', fontsize=12)
    ax.set_title('Policy Vector Field: Control Actions in State Space', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = OUTPUT_DIR / 'c_policy_vector_field.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {filename}")
    plt.close()


def visualize_latent_action_sphere(data):
    """
    (d) Latent Action Embedding Sphere
    Project action mean vectors onto unit hypersphere, show clusters.
    """
    print("\n(d) Creating Latent Action Embedding Sphere...")
    
    actions_mean = data['actions_mean']
    rewards = data['rewards']
    
    # Normalize to unit sphere
    action_norms = np.linalg.norm(actions_mean, axis=1, keepdims=True)
    actions_normalized = actions_mean / (action_norms + 1e-8)
    
    # Reduce to 3D for visualization
    print("  Running PCA for 3D projection...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    actions_3d = pca.fit_transform(actions_normalized)
    
    # Normalize to sphere
    actions_3d_norm = np.linalg.norm(actions_3d, axis=1, keepdims=True)
    actions_3d_sphere = actions_3d / (actions_3d_norm + 1e-8)
    
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 5))
    
    # (d1) Colored by reward
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(actions_3d_sphere[:, 0], 
                          actions_3d_sphere[:, 1], 
                          actions_3d_sphere[:, 2],
                          c=rewards, cmap='RdYlGn', s=10, alpha=0.6)
    ax1.set_title('Action Embedding Sphere\n(Colored by Reward)', fontweight='bold')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    plt.colorbar(scatter1, ax=ax1, label='Reward', shrink=0.5)
    
    # (d2) Colored by action magnitude
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(actions_3d_sphere[:, 0], 
                          actions_3d_sphere[:, 1], 
                          actions_3d_sphere[:, 2],
                          c=action_norms.flatten(), cmap='viridis', s=10, alpha=0.6)
    ax2.set_title('Action Embedding Sphere\n(Colored by Magnitude)', fontweight='bold')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    plt.colorbar(scatter2, ax=ax2, label='Action Magnitude', shrink=0.5)
    
    # (d3) Density projection on 2D
    ax3 = fig.add_subplot(133)
    ax3.hexbin(actions_3d_sphere[:, 0], actions_3d_sphere[:, 1], 
               gridsize=30, cmap='Blues', mincnt=1)
    ax3.set_title('Action Density Projection\n(PC1 vs PC2)', fontweight='bold')
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.set_aspect('equal')
    circle = Circle((0, 0), 1, fill=False, edgecolor='red', linewidth=2, linestyle='--')
    ax3.add_patch(circle)
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)
    
    plt.suptitle('Latent Action Embeddings on Unit Hypersphere', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    filename = OUTPUT_DIR / 'd_action_embedding_sphere.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {filename}")
    plt.close()


def visualize_temporal_heatmap(env, actor, normalization, num_episodes=5, max_steps=200):
    """
    (e) Temporal Evolution Heatmap
    Show action amplitudes per joint over time during rollouts.
    """
    print("\n(e) Creating Temporal Evolution Heatmap...")
    
    all_actions = []
    all_rewards = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        actions_episode = []
        rewards_episode = []
        
        with torch.no_grad():
            for step in range(max_steps):
                # Get action
                obs_norm = normalization.normalize(torch.FloatTensor(obs).to(DEVICE))
                mu, _ = actor(obs_norm.unsqueeze(0))
                action = torch.tanh(mu).cpu().numpy()[0]
                
                # Step environment
                action_env = ((action + 1) / 2) * (env.action_space.high - env.action_space.low) + env.action_space.low
                obs, reward, terminated, truncated, _ = env.step(action_env)
                
                actions_episode.append(action)
                rewards_episode.append(reward)
                
                if terminated or truncated:
                    break
        
        all_actions.append(np.array(actions_episode))
        all_rewards.append(np.array(rewards_episode))
        print(f"  Episode {ep+1}/{num_episodes}: {len(actions_episode)} steps, total reward: {sum(rewards_episode):.2f}")
    
    # Create figure with multiple episodes
    fig, axes = plt.subplots(num_episodes, 1, figsize=(14, 3*num_episodes))
    if num_episodes == 1:
        axes = [axes]
    
    for ep, (actions_ep, rewards_ep) in enumerate(zip(all_actions, all_rewards)):
        # Transpose to (joints, time)
        actions_matrix = actions_ep.T
        
        # Plot heatmap
        im = axes[ep].imshow(actions_matrix, aspect='auto', cmap='RdBu_r', 
                            vmin=-1, vmax=1, interpolation='bilinear')
        
        axes[ep].set_yticks(range(6))
        axes[ep].set_yticklabels(JOINT_NAMES, fontsize=9)
        axes[ep].set_xlabel('Time Step', fontsize=10)
        axes[ep].set_ylabel('Joint', fontsize=10)
        axes[ep].set_title(f'Episode {ep+1}: Action Temporal Evolution (Reward: {sum(rewards_ep):.2f})', 
                          fontsize=11, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[ep])
        cbar.set_label('Normalized Torque', fontsize=9)
    
    plt.suptitle('Temporal Evolution of Joint Torques Across Episodes', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    filename = OUTPUT_DIR / 'e_temporal_heatmap.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to {filename}")
    plt.close()
    
    # Additional: Average temporal pattern
    # Pad to same length and average
    max_len = max(len(a) for a in all_actions)
    actions_padded = []
    for actions_ep in all_actions:
        padded = np.zeros((max_len, 6))
        padded[:len(actions_ep)] = actions_ep
        actions_padded.append(padded)
    
    actions_mean = np.mean(actions_padded, axis=0).T
    
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(actions_mean, aspect='auto', cmap='RdBu_r', 
                   vmin=-1, vmax=1, interpolation='bilinear')
    ax.set_yticks(range(6))
    ax.set_yticklabels(JOINT_NAMES)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Joint', fontsize=12)
    ax.set_title(f'Average Temporal Pattern (Across {num_episodes} Episodes)', 
                fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Mean Normalized Torque')
    plt.tight_layout()
    
    filename = OUTPUT_DIR / 'e_temporal_heatmap_average.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved average pattern to {filename}")
    plt.close()


def main():
    """Main execution pipeline."""
    print("=" * 70)
    print("Advanced Policy Analysis Visualizations for DistRL Walker2d")
    print("=" * 70)
    
    # Load model
    env, actor, rep_trunk, normalization = load_model_and_env()
    
    # Add forward_get_features method to actor if not present
    if not hasattr(actor, 'forward_get_features'):
        def forward_get_features(self, obs):
            # Forward through actor's MLP to get features
            h = self.net(obs)
            self.last_features = h  # Save for extraction
            # Continue through actor heads
            mu = self.mu(h)
            log_std = self.log_std(h)
            return mu, log_std
        
        import types
        actor.forward_get_features = types.MethodType(forward_get_features, actor)
        actor.last_features = None
    
    # Collect samples for manifold visualizations
    data = collect_samples(env, actor, normalization, num_samples=NUM_SAMPLES)
    
    # Generate all visualizations
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)
    
    # (a) Action Landscape over State Manifold
    visualize_action_landscape_manifold(data, use_umap=True)
    
    # (b) Phase Portrait of Joint Torques
    visualize_phase_portraits(data)
    
    # (c) Policy Vector Field in Reduced Coordinates
    visualize_policy_vector_field(env, actor, normalization)
    
    # (d) Latent Action Embedding Sphere
    visualize_latent_action_sphere(data)
    
    # (e) Temporal Evolution Heatmap
    visualize_temporal_heatmap(env, actor, normalization, 
                              num_episodes=NUM_EPISODES, 
                              max_steps=MAX_STEPS)
    
    env.close()
    
    print("\n" + "=" * 70)
    print("✓ All visualizations complete!")
    print(f"✓ Saved to: {OUTPUT_DIR}")
    print("=" * 70)
    
    # Summary
    print("\nGenerated visualizations:")
    print("  (a) Action Landscape over State Manifold - UMAP embedding")
    print("  (b) Phase Portrait of Joint Torques - 6 joint phase diagrams")
    print("  (c) Policy Vector Field - 2D control field with arrows")
    print("  (d) Latent Action Embedding Sphere - 3D hypersphere projection")
    print("  (e) Temporal Evolution Heatmap - Action patterns over time")
    print("\nThese visualizations reveal:")
    print("  • Policy clustering of locomotion phases")
    print("  • Rhythmic control patterns and limit cycles")
    print("  • State-space control field structure")
    print("  • Action space exploration patterns")
    print("  • Coordinated joint motion emergence")


if __name__ == '__main__':
    main()
