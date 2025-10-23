"""
Visualization script for DistRL representations.
Loads a trained actor and distance network and creates multiple visualizations
to explain how the representation learning works and demonstrate its effectiveness.
"""

import sys
from pathlib import Path
# Add parent directory to path to import dist_rl module
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
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

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DistRLVisualizer:
    """Visualizer for DistRL trained representations."""
    
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
            self.env_name = "Walker2d-v4"
            self.obs_dim = 17
            self.act_dim = 6
        elif obs_dim == 376:
            self.env_name = "Humanoid-v4"
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
        self.obs_rms = None
        
        norm_state = checkpoint['normalization']
        print("Loading observation normalization statistics...")
        self.obs_rms = RunningMeanStd(self.obs_dim, device=device)
        self.obs_rms.load_state_dict(norm_state)
        
        self.actor.eval()
        self.rep_trunk.eval()
        
        print(f"Model loaded successfully! Training steps: {checkpoint.get('steps', 'unknown')}")
        
        # Create output directory
        self.output_dir = Path("results_analysis/plots/distrl_embeddings")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_rollout_data(self, num_episodes=4, max_steps=1000):
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
        
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            ep_reward = 0
            
            for t in range(max_steps):
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                obs_t_norm = self.obs_rms.normalize(obs_t)
                
                # Get action from policy (deterministic evaluation mode)
                with torch.no_grad():
                    mu, _ = self.actor.forward(obs_t_norm)
                    action = torch.tanh(mu).clamp(-1, 1)
                    
                    # Get embedding z(s,a) - rep_trunk expects actions in [-1, 1] range
                    embedding = self.rep_trunk(obs_t_norm, action)
                    embedding = F.normalize(embedding, p=2, dim=1)  # Normalized as in training
                    
                    # Convert action to environment space for stepping
                    action_env = ((action + 1) / 2) * (self.action_high - self.action_low) + self.action_low
                    action_env_np = action_env.cpu().numpy()[0]
                
                # Step environment with properly scaled action
                next_obs, reward, done, trunc, _ = self.env.step(action_env_np)
                
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
        }
        
        print(f"Collected {len(states)} state-action pairs")
        return data
    
    def visualize_embedding_space_2d(self, data, method='tsne'):
        """Visualize embedding space in 2D using t-SNE or PCA."""
        print(f"\nCreating 2D embedding visualization using {method.upper()}...")
        
        embeddings = data['embeddings']
        rewards = data['rewards']
        episode_ids = data['episode_ids']
        
        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            coords_2d = reducer.fit_transform(embeddings)
        else:  # PCA
            reducer = PCA(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(embeddings)
            explained_var = reducer.explained_variance_ratio_
            print(f"  Explained variance: {explained_var[0]:.3f}, {explained_var[1]:.3f}")
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        # Plot 1: Colored by reward
        sc1 = axes[0].scatter(coords_2d[:, 0], coords_2d[:, 1], 
                             c=rewards, cmap='RdYlGn', s=20, alpha=0.6)
        axes[0].set_title(f'Embeddings colored by Reward ({method.upper()})', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Component 1', fontsize=12)
        axes[0].set_ylabel('Component 2', fontsize=12)
        plt.colorbar(sc1, ax=axes[0], label='Reward')
        
        # Plot 2: Colored by episode
        sc2 = axes[1].scatter(coords_2d[:, 0], coords_2d[:, 1], 
                             c=episode_ids, cmap='tab10', s=20, alpha=0.6)
        axes[1].set_title(f'Embeddings colored by Episode ({method.upper()})', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Component 1', fontsize=12)
        axes[1].set_ylabel('Component 2', fontsize=12)
        plt.colorbar(sc2, ax=axes[1], label='Episode ID')
        
        # Plot 3: Colored by timestep
        sc3 = axes[2].scatter(coords_2d[:, 0], coords_2d[:, 1], 
                             c=data['timesteps'], cmap='viridis', s=20, alpha=0.6)
        axes[2].set_title(f'Embeddings colored by Timestep ({method.upper()})', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Component 1', fontsize=12)
        axes[2].set_ylabel('Component 2', fontsize=12)
        plt.colorbar(sc3, ax=axes[2], label='Timestep')
        
        plt.tight_layout()
        save_path = self.output_dir / f'embedding_space_2d_{method}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def visualize_trajectory_evolution(self, data, num_episodes=3):
        """Visualize how embeddings evolve along trajectories."""
        print(f"\nVisualizing trajectory evolution for {num_episodes} episodes...")
        
        # Use PCA for consistent 2D projection
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(data['embeddings'])
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, num_episodes))
        
        for ep_id in range(num_episodes):
            mask = data['episode_ids'] == ep_id
            ep_coords = coords_2d[mask]
            ep_rewards = data['rewards'][mask]
            ep_timesteps = data['timesteps'][mask]
            
            # Plot trajectory with arrows
            for i in range(len(ep_coords) - 1):
                ax.annotate('', xy=ep_coords[i+1], xytext=ep_coords[i],
                           arrowprops=dict(arrowstyle='->', color=colors[ep_id], 
                                         alpha=0.3, lw=1.5))
            
            # Plot points colored by cumulative reward
            cumulative_reward = np.cumsum(ep_rewards)
            scatter = ax.scatter(ep_coords[:, 0], ep_coords[:, 1], 
                               c=cumulative_reward, cmap='RdYlGn',
                               s=50, alpha=0.7, edgecolors='black', linewidth=0.5,
                               label=f'Episode {ep_id+1}')
            
            # Mark start and end
            ax.scatter(ep_coords[0, 0], ep_coords[0, 1], 
                      c='blue', s=200, marker='*', edgecolors='black', 
                      linewidth=2, zorder=10)
            ax.scatter(ep_coords[-1, 0], ep_coords[-1, 1], 
                      c='red', s=200, marker='X', edgecolors='black', 
                      linewidth=2, zorder=10)
        
        ax.set_title('Trajectory Evolution in Embedding Space\n(★ = Start, ✖ = End)', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('PCA Component 1', fontsize=12)
        ax.set_ylabel('PCA Component 2', fontsize=12)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'trajectory_evolution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def visualize_similarity_matrices(self, data, sample_size=200):
        """Visualize similarity matrices between embeddings."""
        print(f"\nCreating similarity matrix visualizations...")
        
        # Sample for computational efficiency
        indices = np.random.choice(len(data['embeddings']), 
                                  min(sample_size, len(data['embeddings'])), 
                                  replace=False)
        embeddings = data['embeddings'][indices]
        rewards = data['rewards'][indices]
        
        # Compute cosine similarity matrix
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarity_matrix = embeddings_norm @ embeddings_norm.T
        
        # Sort by rewards for better visualization
        sort_idx = np.argsort(rewards)
        similarity_sorted = similarity_matrix[sort_idx][:, sort_idx]
        rewards_sorted = rewards[sort_idx]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Similarity matrix
        im1 = axes[0].imshow(similarity_sorted, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        axes[0].set_title('Cosine Similarity Matrix\n(sorted by reward)', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('State-Action Pair Index', fontsize=12)
        axes[0].set_ylabel('State-Action Pair Index', fontsize=12)
        plt.colorbar(im1, ax=axes[0], label='Cosine Similarity')
        
        # Plot 2: Similarity vs reward difference
        # Sample pairs for scatter plot
        n_samples = min(5000, len(embeddings) * len(embeddings) // 2)
        pair_indices = np.random.choice(len(embeddings) * (len(embeddings) - 1) // 2, 
                                       n_samples, replace=False)
        
        similarities = []
        reward_diffs = []
        for idx in pair_indices:
            # Convert linear index to 2D
            i = int(np.sqrt(2 * idx))
            j = idx - i * (i + 1) // 2
            if i < len(embeddings) and j < len(embeddings):
                similarities.append(similarity_matrix[i, j])
                reward_diffs.append(abs(rewards[i] - rewards[j]))
        
        axes[1].hexbin(reward_diffs, similarities, gridsize=50, cmap='YlOrRd', mincnt=1)
        axes[1].set_title('Similarity vs Reward Difference', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('|Reward Difference|', fontsize=12)
        axes[1].set_ylabel('Cosine Similarity', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'similarity_matrices.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def visualize_action_space_coverage(self, data):
        """Visualize how different actions lead to different embeddings."""
        print(f"\nVisualizing action space coverage...")
        
        # Use PCA for 2D projection
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(data['embeddings'])
        
        # Compute action norms and angles (for first 2 action dims)
        action_norms = np.linalg.norm(data['actions'], axis=1)
        action_angles = np.arctan2(data['actions'][:, 1] if data['actions'].shape[1] > 1 else 0, 
                                   data['actions'][:, 0])
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Colored by action norm
        sc1 = axes[0].scatter(coords_2d[:, 0], coords_2d[:, 1], 
                             c=action_norms, cmap='plasma', s=20, alpha=0.6)
        axes[0].set_title('Embeddings colored by Action Magnitude', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('PCA Component 1', fontsize=12)
        axes[0].set_ylabel('PCA Component 2', fontsize=12)
        plt.colorbar(sc1, ax=axes[0], label='‖Action‖')
        
        # Plot 2: Action space visualization (first 2 dims)
        sc2 = axes[1].scatter(data['actions'][:, 0], 
                             data['actions'][:, 1] if data['actions'].shape[1] > 1 else np.zeros_like(data['actions'][:, 0]),
                             c=data['rewards'], cmap='RdYlGn', s=20, alpha=0.6)
        axes[1].set_title('Action Space (first 2 dims) colored by Reward', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Action Dim 0', fontsize=12)
        axes[1].set_ylabel('Action Dim 1', fontsize=12)
        plt.colorbar(sc2, ax=axes[1], label='Reward')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'action_space_coverage.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def visualize_temporal_consistency(self, data):
        """Visualize temporal consistency: similarity between consecutive embeddings."""
        print(f"\nAnalyzing temporal consistency...")
        
        embeddings = data['embeddings']
        next_embeddings = data['next_embeddings']
        rewards = data['rewards']
        
        # Compute cosine similarity between z(s,a) and z(s',a')
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        next_embeddings_norm = next_embeddings / (np.linalg.norm(next_embeddings, axis=1, keepdims=True) + 1e-8)
        
        temporal_sims = np.sum(embeddings_norm * next_embeddings_norm, axis=1)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Temporal similarity distribution
        axes[0, 0].hist(temporal_sims, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0, 0].axvline(temporal_sims.mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {temporal_sims.mean():.3f}')
        axes[0, 0].set_title('Distribution of Temporal Similarities\ncos(z(s,a), z(s\',a\'))', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Cosine Similarity', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Temporal similarity vs reward
        axes[0, 1].hexbin(rewards, temporal_sims, gridsize=40, cmap='YlOrRd', mincnt=1)
        axes[0, 1].set_title('Temporal Similarity vs Reward', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Reward', fontsize=12)
        axes[0, 1].set_ylabel('Temporal Similarity', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Temporal similarity over episodes
        for ep_id in range(min(5, data['episode_ids'].max() + 1)):
            mask = data['episode_ids'] == ep_id
            ep_timesteps = data['timesteps'][mask]
            ep_sims = temporal_sims[mask]
            axes[1, 0].plot(ep_timesteps, ep_sims, alpha=0.7, label=f'Episode {ep_id+1}')
        
        axes[1, 0].set_title('Temporal Similarity Along Episodes', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Timestep', fontsize=12)
        axes[1, 0].set_ylabel('Temporal Similarity', fontsize=12)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Embedding distance vs reward
        embedding_dists = np.linalg.norm(embeddings - next_embeddings, axis=1)
        axes[1, 1].hexbin(rewards, embedding_dists, gridsize=40, cmap='YlOrRd', mincnt=1)
        axes[1, 1].set_title('Embedding Distance vs Reward', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Reward', fontsize=12)
        axes[1, 1].set_ylabel('‖z(s,a) - z(s\',a\')‖', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'temporal_consistency.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def visualize_reward_clustering(self, data, num_clusters=5):
        """Visualize how embeddings cluster by reward ranges."""
        print(f"\nVisualizing reward-based clustering...")
        
        # Use t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        coords_2d = tsne.fit_transform(data['embeddings'])
        
        # Create reward bins
        reward_percentiles = np.percentile(data['rewards'], 
                                          np.linspace(0, 100, num_clusters + 1))
        reward_bins = np.digitize(data['rewards'], reward_percentiles[1:-1])
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Plot 1: Scatter by reward bins
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, num_clusters))
        for i in range(num_clusters):
            mask = reward_bins == i
            if mask.sum() > 0:
                reward_range = f'[{reward_percentiles[i]:.2f}, {reward_percentiles[i+1]:.2f}]'
                axes[0].scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                              c=[colors[i]], s=30, alpha=0.6, 
                              label=f'Cluster {i+1}: {reward_range}',
                              edgecolors='black', linewidth=0.3)
        
        axes[0].set_title('Embeddings Clustered by Reward Ranges', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('t-SNE Component 1', fontsize=12)
        axes[0].set_ylabel('t-SNE Component 2', fontsize=12)
        axes[0].legend(fontsize=9, loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Continuous reward coloring with contours
        scatter = axes[1].scatter(coords_2d[:, 0], coords_2d[:, 1], 
                                 c=data['rewards'], cmap='RdYlGn', 
                                 s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
        axes[1].set_title('Embedding Space with Reward Gradient', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('t-SNE Component 1', fontsize=12)
        axes[1].set_ylabel('t-SNE Component 2', fontsize=12)
        plt.colorbar(scatter, ax=axes[1], label='Reward')
        
        plt.tight_layout()
        save_path = self.output_dir / 'reward_clustering.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def analyze_embedding_statistics(self, data):
        """Analyze and visualize embedding statistics."""
        print(f"\nAnalyzing embedding statistics...")
        
        embeddings = data['embeddings']
        
        # Compute statistics
        embedding_norms = np.linalg.norm(embeddings, axis=1)
        embedding_mean = embeddings.mean(axis=0)
        embedding_std = embeddings.std(axis=0)
        
        # Pairwise cosine similarities
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        sample_size = min(1000, len(embeddings))
        sample_idx = np.random.choice(len(embeddings), sample_size, replace=False)
        sim_matrix = embeddings_norm[sample_idx] @ embeddings_norm[sample_idx].T
        pairwise_sims = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Embedding norms
        axes[0, 0].hist(embedding_norms, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0, 0].axvline(embedding_norms.mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {embedding_norms.mean():.3f}')
        axes[0, 0].set_title('Distribution of Embedding Norms', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('‖z(s,a)‖', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Pairwise similarities
        axes[0, 1].hist(pairwise_sims, bins=50, alpha=0.7, color='coral', edgecolor='black')
        axes[0, 1].axvline(pairwise_sims.mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {pairwise_sims.mean():.3f}')
        axes[0, 1].set_title('Distribution of Pairwise Cosine Similarities', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Cosine Similarity', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Dimension-wise statistics
        dim_indices = np.arange(embeddings.shape[1])
        axes[1, 0].fill_between(dim_indices, 
                                embedding_mean - embedding_std,
                                embedding_mean + embedding_std,
                                alpha=0.3, label='±1 std')
        axes[1, 0].plot(dim_indices, embedding_mean, 'b-', linewidth=2, label='Mean')
        axes[1, 0].set_title('Embedding Dimensions: Mean ± Std', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Dimension Index', fontsize=12)
        axes[1, 0].set_ylabel('Value', fontsize=12)
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Dimension variance
        dim_variance = embeddings.var(axis=0)
        sorted_idx = np.argsort(dim_variance)[::-1]
        axes[1, 1].bar(range(min(50, len(dim_variance))), 
                      dim_variance[sorted_idx][:50],
                      alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_title('Top 50 Dimensions by Variance', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Dimension Rank', fontsize=12)
        axes[1, 1].set_ylabel('Variance', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.output_dir / 'embedding_statistics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
        
        # Print statistics
        print(f"\n  Embedding Statistics:")
        print(f"    Mean norm: {embedding_norms.mean():.4f} ± {embedding_norms.std():.4f}")
        print(f"    Mean pairwise similarity: {pairwise_sims.mean():.4f} ± {pairwise_sims.std():.4f}")
        print(f"    Effective dimensionality (ratio of sum to max variance): {dim_variance.sum() / dim_variance.max():.2f}")
    
    def create_summary_figure(self, data):
        """Create a comprehensive summary figure."""
        print(f"\nCreating summary visualization...")
        
        # Use t-SNE for main visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        coords_2d = tsne.fit_transform(data['embeddings'])
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main plot: Large embedding space visualization
        ax_main = fig.add_subplot(gs[0:2, 0:2])
        scatter = ax_main.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                                 c=data['rewards'], cmap='RdYlGn', 
                                 s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
        ax_main.set_title(f'DistRL Embedding Space - {self.env_name}\n(colored by reward)', 
                         fontsize=16, fontweight='bold')
        ax_main.set_xlabel('t-SNE Component 1', fontsize=12)
        ax_main.set_ylabel('t-SNE Component 2', fontsize=12)
        plt.colorbar(scatter, ax=ax_main, label='Reward')
        
        # Top right: Reward distribution
        ax_reward = fig.add_subplot(gs[0, 2])
        ax_reward.hist(data['rewards'], bins=40, alpha=0.7, color='steelblue', edgecolor='black')
        ax_reward.set_title('Reward Distribution', fontsize=12, fontweight='bold')
        ax_reward.set_xlabel('Reward', fontsize=10)
        ax_reward.set_ylabel('Count', fontsize=10)
        ax_reward.grid(True, alpha=0.3)
        
        # Middle right: Temporal similarity
        embeddings_norm = data['embeddings'] / (np.linalg.norm(data['embeddings'], axis=1, keepdims=True) + 1e-8)
        next_embeddings_norm = data['next_embeddings'] / (np.linalg.norm(data['next_embeddings'], axis=1, keepdims=True) + 1e-8)
        temporal_sims = np.sum(embeddings_norm * next_embeddings_norm, axis=1)
        
        ax_temporal = fig.add_subplot(gs[1, 2])
        ax_temporal.hist(temporal_sims, bins=40, alpha=0.7, color='coral', edgecolor='black')
        ax_temporal.axvline(temporal_sims.mean(), color='red', linestyle='--', 
                           linewidth=2, label=f'Mean: {temporal_sims.mean():.3f}')
        ax_temporal.set_title('Temporal Consistency', fontsize=12, fontweight='bold')
        ax_temporal.set_xlabel('cos(z(s,a), z(s\',a\'))', fontsize=10)
        ax_temporal.set_ylabel('Count', fontsize=10)
        ax_temporal.legend(fontsize=9)
        ax_temporal.grid(True, alpha=0.3)
        
        # Bottom left: Episode trajectories
        ax_traj = fig.add_subplot(gs[2, 0])
        pca = PCA(n_components=2, random_state=42)
        coords_pca = pca.fit_transform(data['embeddings'])
        
        for ep_id in range(min(5, data['episode_ids'].max() + 1)):
            mask = data['episode_ids'] == ep_id
            ep_coords = coords_pca[mask]
            ax_traj.plot(ep_coords[:, 0], ep_coords[:, 1], alpha=0.6, linewidth=2, 
                        label=f'Ep {ep_id+1}')
            ax_traj.scatter(ep_coords[0, 0], ep_coords[0, 1], 
                          c='blue', s=100, marker='o', zorder=10)
        
        ax_traj.set_title('Episode Trajectories (PCA)', fontsize=12, fontweight='bold')
        ax_traj.set_xlabel('PC 1', fontsize=10)
        ax_traj.set_ylabel('PC 2', fontsize=10)
        ax_traj.legend(fontsize=8)
        ax_traj.grid(True, alpha=0.3)
        
        # Bottom middle: Action magnitude
        ax_action = fig.add_subplot(gs[2, 1])
        action_norms = np.linalg.norm(data['actions'], axis=1)
        ax_action.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                         c=action_norms, cmap='plasma', s=20, alpha=0.6)
        ax_action.set_title('Action Magnitude', fontsize=12, fontweight='bold')
        ax_action.set_xlabel('t-SNE Comp 1', fontsize=10)
        ax_action.set_ylabel('t-SNE Comp 2', fontsize=10)
        
        # Bottom right: Statistics text
        ax_stats = fig.add_subplot(gs[2, 2])
        ax_stats.axis('off')
        
        stats_text = f"""
        STATISTICS SUMMARY
        ─────────────────────
        Environment: {self.env_name}
        Total samples: {len(data['states'])}
        Episodes: {data['episode_ids'].max() + 1}
        
        Rewards:
          Mean: {data['rewards'].mean():.3f}
          Std: {data['rewards'].std():.3f}
          Min/Max: {data['rewards'].min():.3f}/{data['rewards'].max():.3f}
        
        Embeddings:
          Dimension: {data['embeddings'].shape[1]}
          Mean norm: {np.linalg.norm(data['embeddings'], axis=1).mean():.3f}
        
        Temporal Similarity:
          Mean: {temporal_sims.mean():.3f}
          Std: {temporal_sims.std():.3f}
        """
        
        ax_stats.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                     verticalalignment='center', transform=ax_stats.transAxes)
        
        plt.suptitle('DistRL Representation Learning - Comprehensive Summary', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        save_path = self.output_dir / 'summary_visualization.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()


def main():
    """Main execution function."""
    print("="*80)
    print("DistRL Embedding Visualization")
    print("="*80)
    
    # Initialize visualizer
    model_path = "./saved_models/best.pt"
    visualizer = DistRLVisualizer(model_path)
    
    # Collect rollout data
    data = visualizer.collect_rollout_data(num_episodes=15, max_steps=1000)
    
    # Create all visualizations
    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)
    
    visualizer.visualize_embedding_space_2d(data, method='tsne')
    visualizer.visualize_embedding_space_2d(data, method='pca')
    visualizer.visualize_trajectory_evolution(data, num_episodes=5)
    visualizer.visualize_similarity_matrices(data, sample_size=300)
    visualizer.visualize_action_space_coverage(data)
    visualizer.visualize_temporal_consistency(data)
    visualizer.visualize_reward_clustering(data, num_clusters=5)
    visualizer.analyze_embedding_statistics(data)
    visualizer.create_summary_figure(data)
    
    print("\n" + "="*80)
    print("All visualizations complete!")
    print(f"Saved to: {visualizer.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
