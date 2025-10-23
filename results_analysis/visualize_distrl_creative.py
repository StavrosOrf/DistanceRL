"""
Creative visualizations for DistRL representation trunk.
Explores the learned embedding space through systematic state-action probing.
"""

import sys
from pathlib import Path
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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import os

# Suppress all warnings including threadpoolctl
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

from dist_rl.models import DistanceTrunk, GaussianActor
from dist_rl.utils import RunningMeanStd

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class CreativeDistRLVisualizer:
    """Creative visualizer for DistRL representations."""
    
    # def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model_path = model_path
        
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Determine environment
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
        
        print(f"Detected environment: {self.env_name}")
        
        # Initialize environment
        self.env = gym.make(self.env_name)
        self.env.reset(seed=42)
        
        # Initialize models
        self.actor = GaussianActor(self.obs_dim, self.act_dim, hidden=256).to(device)
        self.rep_trunk = DistanceTrunk(self.obs_dim, self.act_dim, hidden=256, out_dim=256).to(device)
        
        # Load weights
        self.actor.load_state_dict(checkpoint['actor'])
        self.rep_trunk.load_state_dict(checkpoint['rep_trunk'])
        
        # Load normalization
        self.obs_rms = RunningMeanStd(self.obs_dim, device=device)
        self.obs_rms.load_state_dict(checkpoint['normalization'])
        
        self.actor.eval()
        self.rep_trunk.eval()
        
        print(f"Model loaded! Training steps: {checkpoint.get('steps', 'unknown')}")
        
        # Create output directory
        self.output_dir = Path("results_analysis/plots/distrl_creative")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_reference_states(self, num_episodes=10, max_steps=1000):
        """Collect diverse reference states from rollouts."""
        print(f"\nCollecting {num_episodes} episodes of reference states...")
        
        all_states = []
        all_actions = []
        all_rewards = []
        
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            ep_reward = 0
            
            for t in range(max_steps):
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                obs_t_norm = self.obs_rms.normalize(obs_t)
                
                with torch.no_grad():
                    mu, _ = self.actor.forward(obs_t_norm)
                    action = torch.tanh(mu).clamp(-1, 1)
                    action_np = action.cpu().numpy()[0]
                
                # Step with scaled action
                action_low = torch.as_tensor(self.env.action_space.low, device=self.device, dtype=torch.float32)
                action_high = torch.as_tensor(self.env.action_space.high, device=self.device, dtype=torch.float32)
                action_env = ((action + 1) / 2) * (action_high - action_low) + action_low
                
                next_obs, reward, done, trunc, _ = self.env.step(action_env.cpu().numpy()[0])
                
                all_states.append(obs)
                all_actions.append(action_np)
                all_rewards.append(reward)
                
                ep_reward += reward
                obs = next_obs
                
                if done or trunc:
                    break
            
            print(f"  Episode {ep+1}: reward={ep_reward:.2f}, steps={t+1}")
        
        return {
            'states': np.array(all_states),
            'actions': np.array(all_actions),
            'rewards': np.array(all_rewards)
        }
    
    def visualize_state_action_grid(self, reference_data, n_states=50, n_actions=50):
        """
        Visualize embedding space by sampling a grid of states and actions.
        This shows how the representation space is structured.
        """
        print(f"\nCreating state-action grid visualization...")
        
        # Sample diverse states from reference data
        state_indices = np.random.choice(len(reference_data['states']), 
                                        min(n_states, len(reference_data['states'])), 
                                        replace=False)
        sampled_states = reference_data['states'][state_indices]
        
        # Create action grid spanning the action space
        action_grid = np.linspace(-1, 1, n_actions)
        
        embeddings_list = []
        state_ids = []
        action_vals = []
        
        print(f"  Computing embeddings for {n_states} states × {n_actions} actions...")
        
        for i, state in enumerate(sampled_states):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            state_t_norm = self.obs_rms.normalize(state_t)
            
            for action_val in action_grid:
                # Create action: vary first dimension, keep others at 0
                action = np.zeros(self.act_dim)
                action[0] = action_val
                action_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    embedding = self.rep_trunk(state_t_norm, action_t)
                    embedding = F.normalize(embedding, p=2, dim=1)
                
                embeddings_list.append(embedding.cpu().numpy()[0])
                state_ids.append(i)
                action_vals.append(action_val)
        
        embeddings = np.array(embeddings_list)
        state_ids = np.array(state_ids)
        action_vals = np.array(action_vals)
        
        # Apply t-SNE (with progress indicator)
        print(f"  Running t-SNE on {len(embeddings)} embeddings (this may take 1-2 minutes)...")
        print("  Note: You may see threadpoolctl warnings - these are harmless and can be ignored")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1), verbose=0, n_jobs=1)
        coords_2d = tsne.fit_transform(embeddings)
        print("  ✓ t-SNE complete!")
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        
        # Plot 1: Colored by state
        scatter1 = axes[0].scatter(coords_2d[:, 0], coords_2d[:, 1], 
                                  c=state_ids, cmap='tab20', s=30, alpha=0.6)
        axes[0].set_title('Embedding Space Colored by State Identity\n(Fixed action dim varied)', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('t-SNE Component 1', fontsize=12)
        axes[0].set_ylabel('t-SNE Component 2', fontsize=12)
        plt.colorbar(scatter1, ax=axes[0], label='State ID')
        
        # Plot 2: Colored by action value
        scatter2 = axes[1].scatter(coords_2d[:, 0], coords_2d[:, 1], 
                                  c=action_vals, cmap='coolwarm', s=30, alpha=0.6,
                                  vmin=-1, vmax=1)
        axes[1].set_title('Embedding Space Colored by Action Value\n(action[0] ∈ [-1, 1])', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('t-SNE Component 1', fontsize=12)
        axes[1].set_ylabel('t-SNE Component 2', fontsize=12)
        plt.colorbar(scatter2, ax=axes[1], label='Action Value')
        
        # Plot 3: Show structure with grid lines connecting same states
        axes[2].scatter(coords_2d[:, 0], coords_2d[:, 1], 
                       c=action_vals, cmap='coolwarm', s=30, alpha=0.6,
                       vmin=-1, vmax=1)
        
        # Draw lines connecting embeddings of same state
        for state_id in range(min(10, n_states)):  # Only show first 10 for clarity
            mask = state_ids == state_id
            state_coords = coords_2d[mask]
            if len(state_coords) > 1:
                # Sort by action value for smooth lines
                sort_idx = np.argsort(action_vals[mask])
                state_coords = state_coords[sort_idx]
                axes[2].plot(state_coords[:, 0], state_coords[:, 1], 
                           alpha=0.3, linewidth=1)
        
        axes[2].set_title('Embedding Space with State Trajectories\n(lines connect same state, different actions)', 
                         fontsize=14, fontweight='bold')
        axes[2].set_xlabel('t-SNE Component 1', fontsize=12)
        axes[2].set_ylabel('t-SNE Component 2', fontsize=12)
        
        plt.tight_layout()
        save_path = self.output_dir / 'state_action_grid.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def visualize_action_manifolds(self, reference_data, n_states=20):
        """
        Visualize how varying actions creates manifolds in embedding space.
        Shows 2D and 3D action variations.
        """
        print(f"\nCreating action manifold visualization...")
        
        # Sample states
        state_indices = np.random.choice(len(reference_data['states']), 
                                        min(n_states, len(reference_data['states'])), 
                                        replace=False)
        sampled_states = reference_data['states'][state_indices]
        
        # Create 2D action grid (vary first two action dimensions)
        n_grid = 15
        action_vals = np.linspace(-1, 1, n_grid)
        action_grid = np.array(np.meshgrid(action_vals, action_vals)).T.reshape(-1, 2)
        
        all_embeddings = []
        all_state_ids = []
        all_action_coords = []
        
        print(f"  Computing embeddings for {n_states} states × {len(action_grid)} actions...")
        
        for i, state in enumerate(sampled_states):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            state_t_norm = self.obs_rms.normalize(state_t)
            
            for action_2d in action_grid:
                action = np.zeros(self.act_dim)
                action[0] = action_2d[0]
                action[1] = action_2d[1]
                action_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    embedding = self.rep_trunk(state_t_norm, action_t)
                    embedding = F.normalize(embedding, p=2, dim=1)
                
                all_embeddings.append(embedding.cpu().numpy()[0])
                all_state_ids.append(i)
                all_action_coords.append(action_2d)
        
        embeddings = np.array(all_embeddings)
        state_ids = np.array(all_state_ids)
        action_coords = np.array(all_action_coords)
        
        # Apply t-SNE (with progress indicator)
        print(f"  Running t-SNE on {len(embeddings)} embeddings (this may take 1-2 minutes)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1), verbose=0, n_jobs=1)
        coords_2d = tsne.fit_transform(embeddings)
        print("  ✓ t-SNE complete!")
        
        # Create visualization
        fig = plt.figure(figsize=(24, 8))
        
        # Plot 1: Show manifolds for different states
        ax1 = plt.subplot(131)
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_states, 20)))
        
        for i in range(min(n_states, 10)):  # Show first 10
            mask = state_ids == i
            state_coords = coords_2d[mask]
            ax1.scatter(state_coords[:, 0], state_coords[:, 1], 
                       c=[colors[i]], s=40, alpha=0.7, label=f'State {i}')
            
            # Draw hull or contour for this state's manifold
            if len(state_coords) > 3:
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(state_coords)
                    for simplex in hull.simplices:
                        ax1.plot(state_coords[simplex, 0], state_coords[simplex, 1], 
                               color=colors[i], alpha=0.3, linewidth=1)
                except:
                    pass
        
        ax1.set_title('Action Manifolds for Different States\n(Each color = fixed state, varied actions)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE Component 1', fontsize=12)
        ax1.set_ylabel('t-SNE Component 2', fontsize=12)
        ax1.legend(fontsize=8, ncol=2)
        
        # Plot 2: Colored by action[0]
        ax2 = plt.subplot(132)
        scatter2 = ax2.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                              c=action_coords[:, 0], cmap='RdBu_r', 
                              s=30, alpha=0.6, vmin=-1, vmax=1)
        ax2.set_title('Embeddings Colored by Action Dimension 0', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('t-SNE Component 1', fontsize=12)
        ax2.set_ylabel('t-SNE Component 2', fontsize=12)
        plt.colorbar(scatter2, ax=ax2, label='Action[0]')
        
        # Plot 3: Colored by action[1]
        ax3 = plt.subplot(133)
        scatter3 = ax3.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                              c=action_coords[:, 1], cmap='RdBu_r', 
                              s=30, alpha=0.6, vmin=-1, vmax=1)
        ax3.set_title('Embeddings Colored by Action Dimension 1', 
                     fontsize=14, fontweight='bold')
        ax3.set_xlabel('t-SNE Component 1', fontsize=12)
        ax3.set_ylabel('t-SNE Component 2', fontsize=12)
        plt.colorbar(scatter3, ax=ax3, label='Action[1]')
        
        plt.tight_layout()
        save_path = self.output_dir / 'action_manifolds.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def visualize_embedding_interpolation(self, reference_data):
        """
        Visualize interpolation in embedding space vs state/action space.
        Shows if the representation is smooth and continuous.
        """
        print(f"\nCreating embedding interpolation visualization...")
        
        # Pick two random states
        idx1, idx2 = np.random.choice(len(reference_data['states']), 2, replace=False)
        state1 = reference_data['states'][idx1]
        state2 = reference_data['states'][idx2]
        action1 = reference_data['actions'][idx1]
        action2 = reference_data['actions'][idx2]
        
        # Create interpolations
        alphas = np.linspace(0, 1, 20)
        
        state_interp_embeddings = []
        action_interp_embeddings = []
        both_interp_embeddings = []
        
        print("  Computing interpolated embeddings...")
        
        for alpha in alphas:
            # Interpolate in state space
            state_interp = (1 - alpha) * state1 + alpha * state2
            state_t = torch.FloatTensor(state_interp).unsqueeze(0).to(self.device)
            state_t_norm = self.obs_rms.normalize(state_t)
            action_t = torch.FloatTensor(action1).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                emb = self.rep_trunk(state_t_norm, action_t)
                emb = F.normalize(emb, p=2, dim=1)
                state_interp_embeddings.append(emb.cpu().numpy()[0])
            
            # Interpolate in action space
            action_interp = (1 - alpha) * action1 + alpha * action2
            state_t = torch.FloatTensor(state1).unsqueeze(0).to(self.device)
            state_t_norm = self.obs_rms.normalize(state_t)
            action_t = torch.FloatTensor(action_interp).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                emb = self.rep_trunk(state_t_norm, action_t)
                emb = F.normalize(emb, p=2, dim=1)
                action_interp_embeddings.append(emb.cpu().numpy()[0])
            
            # Interpolate in both
            state_t = torch.FloatTensor(state_interp).unsqueeze(0).to(self.device)
            state_t_norm = self.obs_rms.normalize(state_t)
            action_t = torch.FloatTensor(action_interp).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                emb = self.rep_trunk(state_t_norm, action_t)
                emb = F.normalize(emb, p=2, dim=1)
                both_interp_embeddings.append(emb.cpu().numpy()[0])
        
        state_interp_embeddings = np.array(state_interp_embeddings)
        action_interp_embeddings = np.array(action_interp_embeddings)
        both_interp_embeddings = np.array(both_interp_embeddings)
        
        # Compute distances between consecutive embeddings
        state_dists = np.linalg.norm(np.diff(state_interp_embeddings, axis=0), axis=1)
        action_dists = np.linalg.norm(np.diff(action_interp_embeddings, axis=0), axis=1)
        both_dists = np.linalg.norm(np.diff(both_interp_embeddings, axis=0), axis=1)
        
        # Create visualization
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Row 1: Distance plots
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(alphas[1:], state_dists, 'o-', linewidth=2, markersize=6, label='State interp')
        ax1.set_title('State Interpolation Smoothness', fontsize=12, fontweight='bold')
        ax1.set_xlabel('α', fontsize=10)
        ax1.set_ylabel('‖Δz‖', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(alphas[1:], action_dists, 'o-', linewidth=2, markersize=6, 
                color='orange', label='Action interp')
        ax2.set_title('Action Interpolation Smoothness', fontsize=12, fontweight='bold')
        ax2.set_xlabel('α', fontsize=10)
        ax2.set_ylabel('‖Δz‖', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(alphas[1:], both_dists, 'o-', linewidth=2, markersize=6, 
                color='green', label='Both interp')
        ax3.set_title('Joint Interpolation Smoothness', fontsize=12, fontweight='bold')
        ax3.set_xlabel('α', fontsize=10)
        ax3.set_ylabel('‖Δz‖', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Row 2: PCA visualizations
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        all_embeddings = np.vstack([state_interp_embeddings, 
                                    action_interp_embeddings, 
                                    both_interp_embeddings])
        pca.fit(all_embeddings)
        
        state_pca = pca.transform(state_interp_embeddings)
        action_pca = pca.transform(action_interp_embeddings)
        both_pca = pca.transform(both_interp_embeddings)
        
        ax4 = fig.add_subplot(gs[1, :])
        ax4.plot(state_pca[:, 0], state_pca[:, 1], 'o-', linewidth=2, markersize=8, 
                label='State interpolation', alpha=0.7)
        ax4.plot(action_pca[:, 0], action_pca[:, 1], 's-', linewidth=2, markersize=8, 
                label='Action interpolation', alpha=0.7)
        ax4.plot(both_pca[:, 0], both_pca[:, 1], '^-', linewidth=2, markersize=8, 
                label='Joint interpolation', alpha=0.7)
        
        # Mark start and end
        ax4.scatter([state_pca[0, 0]], [state_pca[0, 1]], s=200, c='blue', 
                   marker='*', edgecolors='black', linewidth=2, zorder=10, label='Start')
        ax4.scatter([state_pca[-1, 0]], [state_pca[-1, 1]], s=200, c='red', 
                   marker='X', edgecolors='black', linewidth=2, zorder=10, label='End')
        
        ax4.set_title('Interpolation Paths in PCA Space', fontsize=14, fontweight='bold')
        ax4.set_xlabel('PC 1', fontsize=12)
        ax4.set_ylabel('PC 2', fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # Row 3: 3D PCA visualization
        ax5 = fig.add_subplot(gs[2, :], projection='3d')
        ax5.plot(state_pca[:, 0], state_pca[:, 1], state_pca[:, 2], 
                'o-', linewidth=2, markersize=6, label='State interp', alpha=0.7)
        ax5.plot(action_pca[:, 0], action_pca[:, 1], action_pca[:, 2], 
                's-', linewidth=2, markersize=6, label='Action interp', alpha=0.7)
        ax5.plot(both_pca[:, 0], both_pca[:, 1], both_pca[:, 2], 
                '^-', linewidth=2, markersize=6, label='Joint interp', alpha=0.7)
        
        ax5.scatter([state_pca[0, 0]], [state_pca[0, 1]], [state_pca[0, 2]], 
                   s=200, c='blue', marker='*', edgecolors='black', linewidth=2, zorder=10)
        ax5.scatter([state_pca[-1, 0]], [state_pca[-1, 1]], [state_pca[-1, 2]], 
                   s=200, c='red', marker='X', edgecolors='black', linewidth=2, zorder=10)
        
        ax5.set_title('Interpolation Paths in 3D PCA Space', fontsize=14, fontweight='bold')
        ax5.set_xlabel('PC 1', fontsize=10)
        ax5.set_ylabel('PC 2', fontsize=10)
        ax5.set_zlabel('PC 3', fontsize=10)
        ax5.legend(fontsize=9)
        
        plt.suptitle('Embedding Space Interpolation Analysis', fontsize=16, fontweight='bold')
        
        save_path = self.output_dir / 'embedding_interpolation.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def visualize_embedding_neighborhoods(self, reference_data, n_centers=9):
        """
        Visualize local neighborhoods in embedding space.
        Shows how similar states/actions cluster together.
        """
        print(f"\nCreating embedding neighborhood visualization...")
        
        # Collect embeddings from reference data
        print("  Computing embeddings for reference data...")
        embeddings = []
        states = []
        actions = []
        rewards = []
        
        for i in range(len(reference_data['states'])):
            state = reference_data['states'][i]
            action = reference_data['actions'][i]
            
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            state_t_norm = self.obs_rms.normalize(state_t)
            action_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                emb = self.rep_trunk(state_t_norm, action_t)
                emb = F.normalize(emb, p=2, dim=1)
            
            embeddings.append(emb.cpu().numpy()[0])
            states.append(state)
            actions.append(action)
            rewards.append(reference_data['rewards'][i])
        
        embeddings = np.array(embeddings)
        rewards = np.array(rewards)
        
        # Apply t-SNE (with progress indicator)
        print(f"  Running t-SNE on {len(embeddings)} embeddings (this may take 1-2 minutes)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1), verbose=0, n_jobs=1)
        coords_2d = tsne.fit_transform(embeddings)
        print("  ✓ t-SNE complete!")
        
        # Sample center points across the space
        n_sqrt = int(np.sqrt(n_centers))
        x_range = coords_2d[:, 0].max() - coords_2d[:, 0].min()
        y_range = coords_2d[:, 1].max() - coords_2d[:, 1].min()
        
        x_centers = np.linspace(coords_2d[:, 0].min() + 0.1*x_range, 
                               coords_2d[:, 0].max() - 0.1*x_range, n_sqrt)
        y_centers = np.linspace(coords_2d[:, 1].min() + 0.1*y_range, 
                               coords_2d[:, 1].max() - 0.1*y_range, n_sqrt)
        
        centers = np.array(np.meshgrid(x_centers, y_centers)).T.reshape(-1, 2)
        
        # Create visualization
        fig = plt.figure(figsize=(20, 20))
        gs = GridSpec(n_sqrt, n_sqrt, figure=fig, hspace=0.05, wspace=0.05)
        
        for idx, center in enumerate(centers[:n_centers]):
            # Find k nearest neighbors to this center
            dists = np.linalg.norm(coords_2d - center, axis=1)
            k = 50
            nearest_idx = np.argsort(dists)[:k]
            
            row = idx // n_sqrt
            col = idx % n_sqrt
            ax = fig.add_subplot(gs[row, col])
            
            # Plot all points in gray
            ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c='lightgray', s=5, alpha=0.3)
            
            # Highlight neighborhood
            neighbor_coords = coords_2d[nearest_idx]
            neighbor_rewards = rewards[nearest_idx]
            
            scatter = ax.scatter(neighbor_coords[:, 0], neighbor_coords[:, 1], 
                               c=neighbor_rewards, cmap='RdYlGn', s=50, alpha=0.8,
                               edgecolors='black', linewidth=0.5)
            
            # Mark center
            ax.scatter([center[0]], [center[1]], c='red', s=200, marker='X',
                      edgecolors='black', linewidth=2, zorder=10)
            
            # Draw circle around neighborhood
            circle_radius = dists[nearest_idx[-1]]
            circle = plt.Circle(center, circle_radius, fill=False, 
                              edgecolor='red', linewidth=2, linestyle='--')
            ax.add_patch(circle)
            
            ax.set_xlim(coords_2d[:, 0].min(), coords_2d[:, 0].max())
            ax.set_ylim(coords_2d[:, 1].min(), coords_2d[:, 1].max())
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add reward statistics
            mean_reward = neighbor_rewards.mean()
            ax.text(0.05, 0.95, f'μ={mean_reward:.1f}', 
                   transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.suptitle('Local Neighborhoods in Embedding Space\n(Each panel shows k=50 nearest neighbors to a center point)', 
                    fontsize=16, fontweight='bold')
        
        save_path = self.output_dir / 'embedding_neighborhoods.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def visualize_distance_heatmaps(self, reference_data):
        """
        Create heatmaps showing distances in embedding space vs input space.
        """
        print(f"\nCreating distance heatmap visualization...")
        
        # Sample subset for efficiency
        n_samples = min(100, len(reference_data['states']))
        indices = np.random.choice(len(reference_data['states']), n_samples, replace=False)
        
        sampled_states = reference_data['states'][indices]
        sampled_actions = reference_data['actions'][indices]
        sampled_rewards = reference_data['rewards'][indices]
        
        # Compute embeddings
        print(f"  Computing embeddings for {n_samples} samples...")
        embeddings = []
        
        for i in range(n_samples):
            state_t = torch.FloatTensor(sampled_states[i]).unsqueeze(0).to(self.device)
            state_t_norm = self.obs_rms.normalize(state_t)
            action_t = torch.FloatTensor(sampled_actions[i]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                emb = self.rep_trunk(state_t_norm, action_t)
                emb = F.normalize(emb, p=2, dim=1)
            
            embeddings.append(emb.cpu().numpy()[0])
        
        embeddings = np.array(embeddings)
        
        # Compute distance matrices
        from scipy.spatial.distance import pdist, squareform
        
        state_dists = squareform(pdist(sampled_states, metric='euclidean'))
        action_dists = squareform(pdist(sampled_actions, metric='euclidean'))
        embedding_dists = squareform(pdist(embeddings, metric='cosine'))
        
        # Sort by rewards for better visualization
        sort_idx = np.argsort(sampled_rewards)
        state_dists_sorted = state_dists[sort_idx][:, sort_idx]
        action_dists_sorted = action_dists[sort_idx][:, sort_idx]
        embedding_dists_sorted = embedding_dists[sort_idx][:, sort_idx]
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        
        # Row 1: Distance matrices
        im1 = axes[0, 0].imshow(state_dists_sorted, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('State Distance Matrix\n(Euclidean)', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(action_dists_sorted, cmap='viridis', aspect='auto')
        axes[0, 1].set_title('Action Distance Matrix\n(Euclidean)', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=axes[0, 1])
        
        im3 = axes[0, 2].imshow(embedding_dists_sorted, cmap='plasma', aspect='auto')
        axes[0, 2].set_title('Embedding Distance Matrix\n(Cosine)', fontsize=12, fontweight='bold')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Row 2: Correlation plots
        # State dist vs embedding dist
        axes[1, 0].hexbin(state_dists.flatten(), embedding_dists.flatten(), 
                         gridsize=40, cmap='YlOrRd', mincnt=1)
        axes[1, 0].set_title('State Distance vs Embedding Distance', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('State Distance', fontsize=10)
        axes[1, 0].set_ylabel('Embedding Distance', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Action dist vs embedding dist
        axes[1, 1].hexbin(action_dists.flatten(), embedding_dists.flatten(), 
                         gridsize=40, cmap='YlOrRd', mincnt=1)
        axes[1, 1].set_title('Action Distance vs Embedding Distance', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Action Distance', fontsize=10)
        axes[1, 1].set_ylabel('Embedding Distance', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Combined (state + action) dist vs embedding dist
        combined_dists = np.sqrt(state_dists**2 + action_dists**2)
        axes[1, 2].hexbin(combined_dists.flatten(), embedding_dists.flatten(), 
                         gridsize=40, cmap='YlOrRd', mincnt=1)
        
        # Compute correlation
        corr = np.corrcoef(combined_dists.flatten(), embedding_dists.flatten())[0, 1]
        axes[1, 2].set_title(f'Combined Distance vs Embedding\n(correlation: {corr:.3f})', 
                           fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('√(State² + Action²)', fontsize=10)
        axes[1, 2].set_ylabel('Embedding Distance', fontsize=10)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Distance Analysis: Input Space vs Embedding Space', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / 'distance_heatmaps.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def visualize_principal_directions(self, reference_data):
        """
        Visualize principal directions in embedding space and what they correspond to.
        """
        print(f"\nAnalyzing principal directions in embedding space...")
        
        # Compute embeddings for reference data
        print("  Computing embeddings...")
        embeddings = []
        states = []
        actions = []
        rewards = []
        
        for i in range(len(reference_data['states'])):
            state = reference_data['states'][i]
            action = reference_data['actions'][i]
            
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            state_t_norm = self.obs_rms.normalize(state_t)
            action_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                emb = self.rep_trunk(state_t_norm, action_t)
                emb = F.normalize(emb, p=2, dim=1)
            
            embeddings.append(emb.cpu().numpy()[0])
            states.append(state)
            actions.append(action)
            rewards.append(reference_data['rewards'][i])
        
        embeddings = np.array(embeddings)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        
        # PCA analysis
        print("  Running PCA...")
        pca = PCA(n_components=10)
        embeddings_pca = pca.fit_transform(embeddings)
        
        # Create visualization
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot 1: Variance explained
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.bar(range(10), pca.explained_variance_ratio_[:10], alpha=0.7, color='steelblue')
        ax1.plot(range(10), np.cumsum(pca.explained_variance_ratio_[:10]), 
                'ro-', linewidth=2, markersize=8, label='Cumulative')
        ax1.set_title('Variance Explained by Principal Components', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Component', fontsize=12)
        ax1.set_ylabel('Variance Explained', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: First 3 PCs colored by reward
        ax2 = fig.add_subplot(gs[0, 2:], projection='3d')
        scatter = ax2.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], embeddings_pca[:, 2],
                             c=rewards, cmap='RdYlGn', s=20, alpha=0.6)
        ax2.set_title('First 3 Principal Components', fontsize=14, fontweight='bold')
        ax2.set_xlabel('PC 1', fontsize=10)
        ax2.set_ylabel('PC 2', fontsize=10)
        ax2.set_zlabel('PC 3', fontsize=10)
        plt.colorbar(scatter, ax=ax2, label='Reward', shrink=0.5)
        
        # Plots 3-6: PC pairs colored by reward
        pc_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
        for idx, (pc1, pc2) in enumerate(pc_pairs):
            row = 1 + idx // 2
            col = (idx % 2) * 2
            ax = fig.add_subplot(gs[row, col:col+2])
            
            scatter = ax.scatter(embeddings_pca[:, pc1], embeddings_pca[:, pc2],
                               c=rewards, cmap='RdYlGn', s=20, alpha=0.6)
            ax.set_title(f'PC {pc1+1} vs PC {pc2+1}', fontsize=12, fontweight='bold')
            ax.set_xlabel(f'PC {pc1+1} ({pca.explained_variance_ratio_[pc1]:.1%})', fontsize=10)
            ax.set_ylabel(f'PC {pc2+1} ({pca.explained_variance_ratio_[pc2]:.1%})', fontsize=10)
            plt.colorbar(scatter, ax=ax, label='Reward')
            ax.grid(True, alpha=0.3)
        
        # Plot: Correlation with state/action dims
        ax_corr = fig.add_subplot(gs[3, :2])
        
        # Correlate first PC with state dimensions
        state_corrs = []
        for dim in range(min(10, states.shape[1])):
            corr = np.corrcoef(embeddings_pca[:, 0], states[:, dim])[0, 1]
            state_corrs.append(abs(corr))
        
        x_pos = np.arange(len(state_corrs))
        ax_corr.bar(x_pos, state_corrs, alpha=0.7, color='steelblue')
        ax_corr.set_title('PC1 Correlation with State Dimensions', fontsize=12, fontweight='bold')
        ax_corr.set_xlabel('State Dimension', fontsize=10)
        ax_corr.set_ylabel('|Correlation|', fontsize=10)
        ax_corr.grid(True, alpha=0.3, axis='y')
        
        # Plot: Correlation with action dims
        ax_corr2 = fig.add_subplot(gs[3, 2:])
        
        action_corrs = []
        for dim in range(actions.shape[1]):
            corr = np.corrcoef(embeddings_pca[:, 0], actions[:, dim])[0, 1]
            action_corrs.append(abs(corr))
        
        x_pos = np.arange(len(action_corrs))
        ax_corr2.bar(x_pos, action_corrs, alpha=0.7, color='coral')
        ax_corr2.set_title('PC1 Correlation with Action Dimensions', fontsize=12, fontweight='bold')
        ax_corr2.set_xlabel('Action Dimension', fontsize=10)
        ax_corr2.set_ylabel('|Correlation|', fontsize=10)
        ax_corr2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Principal Component Analysis of Embedding Space', 
                    fontsize=16, fontweight='bold')
        
        save_path = self.output_dir / 'principal_directions.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()


def main():
    """Main execution."""
    print("="*80)
    print("Creative DistRL Representation Visualizations")
    print("="*80)
    
    # Initialize
    model_path = "./saved_models/best.pt"
    visualizer = CreativeDistRLVisualizer(model_path)
    
    # Collect reference data (using fewer episodes for speed)
    reference_data = visualizer.collect_reference_states(num_episodes=5, max_steps=1000)
    
    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)
    
    # Create all visualizations (with reduced sizes for speed)
    visualizer.visualize_state_action_grid(reference_data, n_states=30, n_actions=30)
    visualizer.visualize_action_manifolds(reference_data, n_states=15)
    visualizer.visualize_embedding_interpolation(reference_data)
    visualizer.visualize_embedding_neighborhoods(reference_data, n_centers=9)
    visualizer.visualize_distance_heatmaps(reference_data)
    visualizer.visualize_principal_directions(reference_data)
    
    print("\n" + "="*80)
    print("All creative visualizations complete!")
    print(f"Saved to: {visualizer.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
