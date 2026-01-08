"""
3D Visualization script for DistRL representations.
Creates interactive and static 3D visualizations to demonstrate how the 
representation learning works in DistRL.
"""

import sys
from pathlib import Path

# Add parent directory to path to import dist_rl module
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import gymnasium as gym
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import models
from dist_rl.models import DistanceTrunk, GaussianActor
from dist_rl.utils import RunningMeanStd

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DistRL3DVisualizer:
    """3D Visualizer for DistRL trained representations."""
    
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_path = model_path
        
        # Load the model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Determine environment based on observation dimension
        sample_weight = list(checkpoint['actor'].values())[0]
        obs_dim = sample_weight.shape[1]
        
        if obs_dim == 17:
            self.env_name = "Walker2d-v5"
            self.obs_dim = 17
            self.act_dim = 6
        elif obs_dim == 376:
            self.env_name = "Humanoid-v5"
            self.obs_dim = 376
            self.act_dim = 17
        else:
            raise ValueError(f"Unknown observation dimension: {obs_dim}")
        
        print(f"Detected environment: {self.env_name} (obs_dim={self.obs_dim}, act_dim={self.act_dim})")
        
        # Initialize environment
        self.env = gym.make(self.env_name)
        self.env.reset(seed=42)
        
        # Store action space bounds as tensors
        self.action_low = torch.as_tensor(self.env.action_space.low, device=device, dtype=torch.float32)
        self.action_high = torch.as_tensor(self.env.action_space.high, device=device, dtype=torch.float32)
        
        # Initialize models
        self.actor = GaussianActor(self.obs_dim, self.act_dim, hidden=256).to(device)
        self.rep_trunk = DistanceTrunk(self.obs_dim, self.act_dim, hidden=256, out_dim=256).to(device)
        
        # Load weights
        self.actor.load_state_dict(checkpoint['actor'])
        self.rep_trunk.load_state_dict(checkpoint['rep_trunk'])
        
        # Load normalization stats if available
        self.obs_mean = None
        self.obs_std = None
        
        norm_state = checkpoint['normalization']
        print("Loading observation normalization statistics...")
        self.obs_rms = RunningMeanStd(self.obs_dim, device=device)

        self.obs_rms.load_state_dict(norm_state)

        self.actor.eval()
        self.rep_trunk.eval()
        
        print(f"Model loaded successfully! Training steps: {checkpoint.get('steps', 'unknown')}")
        
        # Create output directory
        self.output_dir = Path("results_analysis/plots/distrl_embeddings_3d")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_rollout_data(self, num_episodes=10, max_steps=1000):
        """Collect trajectory data from the environment."""
        print(f"\nCollecting rollout data from {num_episodes} episodes...")
        
        states = []
        actions = []
        rewards = []
        embeddings = []
        next_states = []
        next_embeddings = []
        episode_ids = []
        timesteps = []
        cumulative_rewards = []
        
        for ep in range(num_episodes):
            print(f" Episode {ep+1}/{num_episodes}")
            obs, _ = self.env.reset()
            ep_reward = 0
            cum_reward = 0
            
            for t in range(max_steps):
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                obs_t_norm = self.obs_rms.normalize(obs_t)
                
                # Get action from policy (deterministic evaluation mode)
                with torch.no_grad():
                    mu, _ = self.actor.forward(obs_t_norm)
                    action = torch.tanh(mu) #.clamp(-1, 1)
                    
                    # Get embedding z(s,a) - rep_trunk expects actions in [-1, 1] range
                    embedding = self.rep_trunk(obs_t_norm, action)
                    embedding = F.normalize(embedding, p=2, dim=1)
                    
                    # Convert action to environment space for stepping
                    action_env = ((action + 1) / 2) * (self.action_high - self.action_low) + self.action_low
                    action_env_np = action_env.cpu().numpy()[0]
                
                # Step environment with properly scaled action
                next_obs, reward, done, trunc, _ = self.env.step(action_env_np)
                cum_reward += reward
                
                # Get next state embedding
                next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
                next_obs_t_norm = self.obs_rms.normalize(next_obs_t)
                with torch.no_grad():
                    next_mu, _ = self.actor.forward(next_obs_t_norm)
                    next_action = torch.tanh(next_mu)
                    # rep_trunk expects actions in [-1, 1] range
                    next_embedding = self.rep_trunk(next_obs_t_norm, next_action)
                    next_embedding = F.normalize(next_embedding, p=2, dim=1)
                
                # Store data
                states.append(obs)
                actions.append(action.cpu().numpy()[0])
                rewards.append(reward)
                embeddings.append(embedding.cpu().numpy()[0])
                next_states.append(next_obs)
                next_embeddings.append(next_embedding.cpu().numpy()[0])
                episode_ids.append(ep)
                timesteps.append(t)
                cumulative_rewards.append(cum_reward)
                
                ep_reward += reward
                obs = next_obs
                
                if done or trunc:
                    break
            
            print(f"  Episode {ep+1}/{num_episodes}: reward={ep_reward:.2f}, steps={t+1}")
        
        data = {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'embeddings': np.array(embeddings),
            'next_embeddings': np.array(next_embeddings),
            'episode_ids': np.array(episode_ids),
            'timesteps': np.array(timesteps),
            'cumulative_rewards': np.array(cumulative_rewards),
        }
        
        print(f"Collected {len(states)} state-action pairs")
        return data
    
    def visualize_3d_embedding_space(self, data, method='pca'):
        """Create 3D visualization of embedding space."""
        print(f"\nCreating 3D embedding space visualization using {method.upper()}...")
        
        embeddings = data['embeddings']
        rewards = data['rewards']
        episode_ids = data['episode_ids']
        
        # Dimensionality reduction to 3D
        if method == 'tsne':
            reducer = TSNE(n_components=3, random_state=42, perplexity=30)
            coords_3d = reducer.fit_transform(embeddings)
        else:  # PCA
            reducer = PCA(n_components=3, random_state=42)
            coords_3d = reducer.fit_transform(embeddings)
            explained_var = reducer.explained_variance_ratio_
            print(f"  Explained variance: {explained_var[0]:.3f}, {explained_var[1]:.3f}, {explained_var[2]:.3f}")
        
        # Create 4 different views
        fig = plt.figure(figsize=(20, 15))
        
        # View 1: Colored by reward
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        scatter1 = ax1.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
                              c=rewards, cmap='RdYlGn', s=30, alpha=0.6,
                              edgecolors='black', linewidth=0.3)
        ax1.set_title(f'Embeddings colored by Reward ({method.upper()})', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('Component 1', fontsize=10, labelpad=10)
        ax1.set_ylabel('Component 2', fontsize=10, labelpad=10)
        ax1.set_zlabel('Component 3', fontsize=10, labelpad=10)
        plt.colorbar(scatter1, ax=ax1, label='Reward', pad=0.1, shrink=0.8)
        ax1.view_init(elev=20, azim=45)
        
        # View 2: Colored by episode
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        scatter2 = ax2.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
                              c=episode_ids, cmap='tab10', s=30, alpha=0.6,
                              edgecolors='black', linewidth=0.3)
        ax2.set_title(f'Embeddings colored by Episode ({method.upper()})', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('Component 1', fontsize=10, labelpad=10)
        ax2.set_ylabel('Component 2', fontsize=10, labelpad=10)
        ax2.set_zlabel('Component 3', fontsize=10, labelpad=10)
        plt.colorbar(scatter2, ax=ax2, label='Episode ID', pad=0.1, shrink=0.8)
        ax2.view_init(elev=20, azim=135)
        
        # View 3: Colored by timestep
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        scatter3 = ax3.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
                              c=data['timesteps'], cmap='viridis', s=30, alpha=0.6,
                              edgecolors='black', linewidth=0.3)
        ax3.set_title(f'Embeddings colored by Timestep ({method.upper()})', 
                     fontsize=14, fontweight='bold', pad=20)
        ax3.set_xlabel('Component 1', fontsize=10, labelpad=10)
        ax3.set_ylabel('Component 2', fontsize=10, labelpad=10)
        ax3.set_zlabel('Component 3', fontsize=10, labelpad=10)
        plt.colorbar(scatter3, ax=ax3, label='Timestep', pad=0.1, shrink=0.8)
        ax3.view_init(elev=20, azim=225)
        
        # View 4: Colored by cumulative reward
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        scatter4 = ax4.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
                              c=data['cumulative_rewards'], cmap='plasma', s=30, alpha=0.6,
                              edgecolors='black', linewidth=0.3)
        ax4.set_title(f'Embeddings colored by Cumulative Reward ({method.upper()})', 
                     fontsize=14, fontweight='bold', pad=20)
        ax4.set_xlabel('Component 1', fontsize=10, labelpad=10)
        ax4.set_ylabel('Component 2', fontsize=10, labelpad=10)
        ax4.set_zlabel('Component 3', fontsize=10, labelpad=10)
        plt.colorbar(scatter4, ax=ax4, label='Cumulative Reward', pad=0.1, shrink=0.8)
        ax4.view_init(elev=20, azim=315)
        
        plt.suptitle(f'3D Embedding Space - {self.env_name}', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        save_path = self.output_dir / f'3d_embedding_space_{method}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
        
        return coords_3d
    
    def visualize_3d_trajectories(self, data, num_episodes=5):
        """Visualize trajectories in 3D embedding space."""
        print(f"\nVisualizing 3D trajectories for {num_episodes} episodes...")
        
        # Use PCA for consistent projection
        pca = PCA(n_components=3, random_state=42)
        coords_3d = pca.fit_transform(data['embeddings'])
        
        fig = plt.figure(figsize=(20, 10))
        
        # View 1: Trajectories with gradient coloring
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        
        colors = plt.cm.tab10(np.linspace(0, 1, num_episodes))
        
        for ep_id in range(num_episodes):
            mask = data['episode_ids'] == ep_id
            ep_coords = coords_3d[mask]
            ep_rewards = data['rewards'][mask]
            cumulative = np.cumsum(ep_rewards)
            
            # Plot trajectory as connected line segments with color gradient
            for i in range(len(ep_coords) - 1):
                ax1.plot([ep_coords[i, 0], ep_coords[i+1, 0]],
                        [ep_coords[i, 1], ep_coords[i+1, 1]],
                        [ep_coords[i, 2], ep_coords[i+1, 2]],
                        color=colors[ep_id], alpha=0.6, linewidth=2)
            
            # Plot points colored by cumulative reward
            scatter = ax1.scatter(ep_coords[:, 0], ep_coords[:, 1], ep_coords[:, 2],
                                c=cumulative, cmap='RdYlGn', s=50, alpha=0.8,
                                edgecolors='black', linewidth=0.5)
            
            # Mark start (star) and end (X)
            ax1.scatter(ep_coords[0, 0], ep_coords[0, 1], ep_coords[0, 2],
                       c='blue', s=300, marker='*', edgecolors='black',
                       linewidth=2, zorder=10)
            ax1.scatter(ep_coords[-1, 0], ep_coords[-1, 1], ep_coords[-1, 2],
                       c='red', s=300, marker='X', edgecolors='black',
                       linewidth=2, zorder=10)
        
        ax1.set_title('Episode Trajectories in 3D Embedding Space\n(★ = Start, ✖ = End)',
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('PC 1', fontsize=11, labelpad=10)
        ax1.set_ylabel('PC 2', fontsize=11, labelpad=10)
        ax1.set_zlabel('PC 3', fontsize=11, labelpad=10)
        ax1.view_init(elev=20, azim=45)
        
        # View 2: Same trajectories from different angle
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        
        for ep_id in range(num_episodes):
            mask = data['episode_ids'] == ep_id
            ep_coords = coords_3d[mask]
            ep_rewards = data['rewards'][mask]
            cumulative = np.cumsum(ep_rewards)
            
            # Plot trajectory
            for i in range(len(ep_coords) - 1):
                ax2.plot([ep_coords[i, 0], ep_coords[i+1, 0]],
                        [ep_coords[i, 1], ep_coords[i+1, 1]],
                        [ep_coords[i, 2], ep_coords[i+1, 2]],
                        color=colors[ep_id], alpha=0.6, linewidth=2,
                        label=f'Episode {ep_id+1}' if i == 0 else '')
            
            # Plot points
            scatter = ax2.scatter(ep_coords[:, 0], ep_coords[:, 1], ep_coords[:, 2],
                                c=cumulative, cmap='RdYlGn', s=50, alpha=0.8,
                                edgecolors='black', linewidth=0.5)
            
            # Mark start and end
            ax2.scatter(ep_coords[0, 0], ep_coords[0, 1], ep_coords[0, 2],
                       c='blue', s=300, marker='*', edgecolors='black',
                       linewidth=2, zorder=10)
            ax2.scatter(ep_coords[-1, 0], ep_coords[-1, 1], ep_coords[-1, 2],
                       c='red', s=300, marker='X', edgecolors='black',
                       linewidth=2, zorder=10)
        
        ax2.set_title('Episode Trajectories (Different View)\n(★ = Start, ✖ = End)',
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('PC 1', fontsize=11, labelpad=10)
        ax2.set_ylabel('PC 2', fontsize=11, labelpad=10)
        ax2.set_zlabel('PC 3', fontsize=11, labelpad=10)
        ax2.legend(fontsize=10, loc='upper left')
        ax2.view_init(elev=30, azim=225)
        
        plt.suptitle(f'3D Trajectory Evolution - {self.env_name}',
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        save_path = self.output_dir / '3d_trajectories.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def visualize_temporal_transitions_3d(self, data, sample_size=500):
        """Visualize temporal transitions z(s,a) -> z(s',a') in 3D."""
        print(f"\nVisualizing temporal transitions in 3D...")
        
        # Use PCA for dimensionality reduction
        all_embeddings = np.vstack([data['embeddings'], data['next_embeddings']])
        pca = PCA(n_components=3, random_state=42)
        all_coords = pca.fit_transform(all_embeddings)
        
        n_samples = len(data['embeddings'])
        coords_3d = all_coords[:n_samples]
        next_coords_3d = all_coords[n_samples:]
        
        # Sample for visualization
        if len(data['embeddings']) > sample_size:
            indices = np.random.choice(len(data['embeddings']), sample_size, replace=False)
        else:
            indices = np.arange(len(data['embeddings']))
        
        fig = plt.figure(figsize=(20, 10))
        
        # View 1: Arrows colored by reward
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        
        for idx in indices:
            reward = data['rewards'][idx]
            color = plt.cm.RdYlGn((reward - data['rewards'].min()) / 
                                  (data['rewards'].max() - data['rewards'].min() + 1e-8))
            
            # Draw arrow from z(s,a) to z(s',a')
            ax1.plot([coords_3d[idx, 0], next_coords_3d[idx, 0]],
                    [coords_3d[idx, 1], next_coords_3d[idx, 1]],
                    [coords_3d[idx, 2], next_coords_3d[idx, 2]],
                    color=color, alpha=0.3, linewidth=1)
        
        # Plot current states
        scatter1 = ax1.scatter(coords_3d[indices, 0], coords_3d[indices, 1], coords_3d[indices, 2],
                              c=data['rewards'][indices], cmap='RdYlGn', s=30, alpha=0.6,
                              edgecolors='black', linewidth=0.5, label='z(s,a)')
        
        ax1.set_title('Temporal Transitions: z(s,a) → z(s\',a\')\nColored by Reward',
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('PC 1', fontsize=11, labelpad=10)
        ax1.set_ylabel('PC 2', fontsize=11, labelpad=10)
        ax1.set_zlabel('PC 3', fontsize=11, labelpad=10)
        plt.colorbar(scatter1, ax=ax1, label='Reward', pad=0.1, shrink=0.8)
        ax1.view_init(elev=20, azim=45)
        
        # View 2: Transition magnitude visualization
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        
        transition_dists = np.linalg.norm(coords_3d - next_coords_3d, axis=1)
        
        for idx in indices:
            dist = transition_dists[idx]
            color = plt.cm.viridis(dist / (transition_dists.max() + 1e-8))
            
            ax2.plot([coords_3d[idx, 0], next_coords_3d[idx, 0]],
                    [coords_3d[idx, 1], next_coords_3d[idx, 1]],
                    [coords_3d[idx, 2], next_coords_3d[idx, 2]],
                    color=color, alpha=0.3, linewidth=1)
        
        scatter2 = ax2.scatter(coords_3d[indices, 0], coords_3d[indices, 1], coords_3d[indices, 2],
                              c=transition_dists[indices], cmap='viridis', s=30, alpha=0.6,
                              edgecolors='black', linewidth=0.5)
        
        ax2.set_title('Temporal Transitions\nColored by Transition Distance',
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('PC 1', fontsize=11, labelpad=10)
        ax2.set_ylabel('PC 2', fontsize=11, labelpad=10)
        ax2.set_zlabel('PC 3', fontsize=11, labelpad=10)
        plt.colorbar(scatter2, ax=ax2, label='Transition Distance', pad=0.1, shrink=0.8)
        ax2.view_init(elev=20, azim=135)
        
        plt.suptitle(f'Temporal Dynamics in Embedding Space - {self.env_name}',
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        save_path = self.output_dir / '3d_temporal_transitions.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def visualize_reward_landscape_3d(self, data):
        """Create 3D reward landscape visualization."""
        print(f"\nCreating 3D reward landscape...")
        
        # Use PCA for dimensionality reduction
        pca = PCA(n_components=3, random_state=42)
        coords_3d = pca.fit_transform(data['embeddings'])
        
        fig = plt.figure(figsize=(20, 15))
        
        # View 1: Surface plot approximation
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        
        # Create a surface by binning the data
        from scipy.interpolate import griddata
        
        # Create grid
        xi = np.linspace(coords_3d[:, 0].min(), coords_3d[:, 0].max(), 50)
        yi = np.linspace(coords_3d[:, 1].min(), coords_3d[:, 1].max(), 50)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpolate Z values (PC3) and rewards
        Zi = griddata((coords_3d[:, 0], coords_3d[:, 1]), coords_3d[:, 2],
                     (Xi, Yi), method='cubic', fill_value=0)
        Ri = griddata((coords_3d[:, 0], coords_3d[:, 1]), data['rewards'],
                     (Xi, Yi), method='cubic', fill_value=data['rewards'].mean())
        
        surf = ax1.plot_surface(Xi, Yi, Zi, facecolors=plt.cm.RdYlGn((Ri - Ri.min()) / (Ri.max() - Ri.min() + 1e-8)),
                               alpha=0.7, linewidth=0, antialiased=True)
        
        ax1.set_title('Reward Landscape (Surface Approximation)',
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('PC 1', fontsize=11, labelpad=10)
        ax1.set_ylabel('PC 2', fontsize=11, labelpad=10)
        ax1.set_zlabel('PC 3', fontsize=11, labelpad=10)
        ax1.view_init(elev=25, azim=45)
        
        # View 2: Points with size proportional to reward
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        
        # Normalize rewards to get point sizes
        normalized_rewards = (data['rewards'] - data['rewards'].min()) / (data['rewards'].max() - data['rewards'].min() + 1e-8)
        sizes = 10 + normalized_rewards * 100  # Size between 10 and 110
        
        scatter = ax2.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
                            c=data['rewards'], cmap='RdYlGn', s=sizes, alpha=0.6,
                            edgecolors='black', linewidth=0.3)
        
        ax2.set_title('Embeddings (size ∝ reward)',
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('PC 1', fontsize=11, labelpad=10)
        ax2.set_ylabel('PC 2', fontsize=11, labelpad=10)
        ax2.set_zlabel('PC 3', fontsize=11, labelpad=10)
        plt.colorbar(scatter, ax=ax2, label='Reward', pad=0.1, shrink=0.8)
        ax2.view_init(elev=25, azim=135)
        
        # View 3: Reward clusters in 3D
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        
        # Create reward bins
        n_bins = 5
        reward_percentiles = np.percentile(data['rewards'], np.linspace(0, 100, n_bins + 1))
        reward_bins = np.digitize(data['rewards'], reward_percentiles[1:-1])
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_bins))
        for i in range(n_bins):
            mask = reward_bins == i
            if mask.sum() > 0:
                ax3.scatter(coords_3d[mask, 0], coords_3d[mask, 1], coords_3d[mask, 2],
                          c=[colors[i]], s=40, alpha=0.6, edgecolors='black', linewidth=0.3,
                          label=f'R ∈ [{reward_percentiles[i]:.1f}, {reward_percentiles[i+1]:.1f}]')
        
        ax3.set_title('Reward-based Clustering',
                     fontsize=14, fontweight='bold', pad=20)
        ax3.set_xlabel('PC 1', fontsize=11, labelpad=10)
        ax3.set_ylabel('PC 2', fontsize=11, labelpad=10)
        ax3.set_zlabel('PC 3', fontsize=11, labelpad=10)
        ax3.legend(fontsize=9, loc='upper left')
        ax3.view_init(elev=25, azim=225)
        
        # View 4: Density visualization with contours
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        
        scatter4 = ax4.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
                              c=data['cumulative_rewards'], cmap='plasma', s=40, alpha=0.6,
                              edgecolors='black', linewidth=0.3)
        
        ax4.set_title('Embeddings by Cumulative Reward',
                     fontsize=14, fontweight='bold', pad=20)
        ax4.set_xlabel('PC 1', fontsize=11, labelpad=10)
        ax4.set_ylabel('PC 2', fontsize=11, labelpad=10)
        ax4.set_zlabel('PC 3', fontsize=11, labelpad=10)
        plt.colorbar(scatter4, ax=ax4, label='Cumulative Reward', pad=0.1, shrink=0.8)
        ax4.view_init(elev=25, azim=315)
        
        plt.suptitle(f'3D Reward Landscape - {self.env_name}',
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        save_path = self.output_dir / '3d_reward_landscape.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def visualize_action_effect_3d(self, data, num_samples=1000):
        """Visualize how different actions affect embeddings in 3D."""
        print(f"\nVisualizing action effects in 3D...")
        
        # Sample states for visualization
        if len(data['states']) > num_samples:
            indices = np.random.choice(len(data['states']), num_samples, replace=False)
        else:
            indices = np.arange(len(data['states']))
        
        # For each sampled state, generate embeddings with different actions
        print("  Generating action variations...")
        state_embeddings = []
        action_norms = []
        action_indices = []
        original_rewards = []
        
        for idx in indices[:100]:  # Limit to 100 states for computational efficiency
            state = data['states'][idx]
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            state_t_norm = self.obs_rms.normalize(state_t)
            
            # Generate K different actions
            K = 20
            with torch.no_grad():
                # Sample multiple actions
                for k in range(K):
                    action, _, _ = self.actor.sample(state_t_norm)
                    
                    # Add some noise for variation
                    if k > 0:
                        action = action + 0.2 * torch.randn_like(action)
                        action = action.clamp(-1, 1)
                    
                    # Get embedding
                    embedding = self.rep_trunk(state_t_norm, action)
                    embedding = F.normalize(embedding, p=2, dim=1)
                    
                    state_embeddings.append(embedding.cpu().numpy()[0])
                    action_norms.append(np.linalg.norm(action.cpu().numpy()[0]))
                    action_indices.append(idx)
                    original_rewards.append(data['rewards'][idx])
        
        state_embeddings = np.array(state_embeddings)
        action_norms = np.array(action_norms)
        
        print("  Performing dimensionality reduction...")
        # Reduce to 3D
        pca = PCA(n_components=3, random_state=42)
        coords_3d = pca.fit_transform(state_embeddings)
        
        fig = plt.figure(figsize=(20, 10))
        
        # View 1: Colored by action norm
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        
        scatter1 = ax1.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
                              c=action_norms, cmap='plasma', s=40, alpha=0.6,
                              edgecolors='black', linewidth=0.3)
        
        ax1.set_title('Action Variations\nColored by Action Magnitude',
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('PC 1', fontsize=11, labelpad=10)
        ax1.set_ylabel('PC 2', fontsize=11, labelpad=10)
        ax1.set_zlabel('PC 3', fontsize=11, labelpad=10)
        plt.colorbar(scatter1, ax=ax1, label='‖Action‖', pad=0.1, shrink=0.8)
        ax1.view_init(elev=20, azim=45)
        
        # View 2: Group by original state, show action variations
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        
        # Plot each state's action variations as connected group
        unique_states = np.unique(action_indices)
        colors_state = plt.cm.tab20(np.linspace(0, 1, len(unique_states)))
        
        for i, state_idx in enumerate(unique_states[:15]):  # Show first 15 states
            mask = np.array(action_indices) == state_idx
            state_coords = coords_3d[mask]
            
            # Plot the cluster of actions for this state
            ax2.scatter(state_coords[:, 0], state_coords[:, 1], state_coords[:, 2],
                       c=[colors_state[i]], s=60, alpha=0.7, edgecolors='black',
                       linewidth=0.5, label=f'State {i+1}')
            
            # Draw lines connecting action variations from same state
            center = state_coords.mean(axis=0)
            for coord in state_coords:
                ax2.plot([center[0], coord[0]], [center[1], coord[1]], [center[2], coord[2]],
                        color=colors_state[i], alpha=0.2, linewidth=1)
        
        ax2.set_title('Action Variations per State\n(lines connect variations from same state)',
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('PC 1', fontsize=11, labelpad=10)
        ax2.set_ylabel('PC 2', fontsize=11, labelpad=10)
        ax2.set_zlabel('PC 3', fontsize=11, labelpad=10)
        ax2.legend(fontsize=8, loc='upper left', ncol=2)
        ax2.view_init(elev=20, azim=135)
        
        plt.suptitle(f'Action Effects on Embeddings - {self.env_name}',
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        save_path = self.output_dir / '3d_action_effects.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def create_rotating_views(self, data, num_angles=8):
        """Create multiple views at different angles for a rotating effect."""
        print(f"\nCreating rotating view visualization with {num_angles} angles...")
        
        # Use PCA for dimensionality reduction
        pca = PCA(n_components=3, random_state=42)
        coords_3d = pca.fit_transform(data['embeddings'])
        
        fig = plt.figure(figsize=(24, 16))
        
        angles = np.linspace(0, 360, num_angles, endpoint=False)
        
        for i, azim in enumerate(angles):
            ax = fig.add_subplot(2, 4, i+1, projection='3d')
            
            scatter = ax.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
                               c=data['rewards'], cmap='RdYlGn', s=20, alpha=0.6,
                               edgecolors='black', linewidth=0.2)
            
            ax.set_title(f'View: {azim:.0f}°', fontsize=12, fontweight='bold')
            ax.set_xlabel('PC 1', fontsize=9)
            ax.set_ylabel('PC 2', fontsize=9)
            ax.set_zlabel('PC 3', fontsize=9)
            ax.view_init(elev=20, azim=azim)
            
            if i == num_angles - 1:
                plt.colorbar(scatter, ax=ax, label='Reward', pad=0.1, shrink=0.7)
        
        plt.suptitle(f'360° Rotating View of Embedding Space - {self.env_name}',
                    fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        save_path = self.output_dir / '3d_rotating_views.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()


def main():
    """Main execution function."""
    print("="*80)
    print("DistRL 3D Embedding Visualization")
    print("="*80)
    
    # Initialize visualizer
    model_path = "./saved_models/best.pt"
    visualizer = DistRL3DVisualizer(model_path)
    
    # Collect rollout data
    data = visualizer.collect_rollout_data(num_episodes=5, max_steps=1000)
    
    # Create all 3D visualizations
    print("\n" + "="*80)
    print("Creating 3D Visualizations")
    print("="*80)
    
    visualizer.visualize_3d_embedding_space(data, method='pca')
    visualizer.visualize_3d_embedding_space(data, method='tsne')
    visualizer.visualize_3d_trajectories(data, num_episodes=5)
    visualizer.visualize_temporal_transitions_3d(data, sample_size=500)
    visualizer.visualize_reward_landscape_3d(data)
    visualizer.visualize_action_effect_3d(data, num_samples=1000)
    visualizer.create_rotating_views(data, num_angles=8)
    
    print("\n" + "="*80)
    print("All 3D visualizations complete!")
    print(f"Saved to: {visualizer.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
