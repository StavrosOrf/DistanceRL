import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gymnasium as gym
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*threadpoolctl.*')

# Import DistRL components
import sys
sys.path.append('.')
from dist_rl.models import DistanceTrunk, GaussianActor, TwinQ
from dist_rl.utils import RunningMeanStd

# Configuration
DEVICE = 'cuda'
MODEL_PATH = './saved_models/best.pt'
ENV_NAME = 'Walker2d-v5'
NUM_SAMPLES = 1000      # For manifold visualiza    tions
NUM_EPISODES = 1        # For trajectory analysis
MAX_STEPS = 1000         # Steps per episode
OUTPUT_DIR = Path('results_analysis/plots/representation_geometry')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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




def visualize_representation_space():
    """Visualize the representation space using various techniques."""
    env, actor, rep_trunk, qnet, normalization = load_model_and_env()
    print("Sampling observations...")
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"Action dim: {env.action_space}")
    print(f"Observation dim: {env.observation_space}")
    
    # for a fixed state, vary each action dimension and see how the representation changes
    state = env.reset()[0]
    state = torch.tensor(state, dtype=torch.float32).to(DEVICE)
    state = normalization.normalize(state)
    # state = state.unsqueeze(0).repeat(100, 1)  # Repeat for batch processing                
    
    df_rep = []
    action = torch.zeros((act_dim)).to(DEVICE)
    
    samples_per_dim = 10

    for a0 in range(samples_per_dim):
        for a1 in range(samples_per_dim):
            for a2 in range(samples_per_dim):
                for a3 in range(samples_per_dim):
                    for a4 in range(samples_per_dim):
                        for a5 in range(samples_per_dim):
                            print(f"Processing action indices: {a0}, {a1}, {a2}, {a3}, {a4}, {a5}")
                            action[0] = -1.0 + 2.0 * a0 / (samples_per_dim - 1)
                            action[1] = -1.0 + 2.0 * a1 / (samples_per_dim - 1)
                            action[2] = -1.0 + 2.0 * a2 / (samples_per_dim - 1)
                            action[3] = -1.0 + 2.0 * a3 / (samples_per_dim - 1)
                            action[4] = -1.0 + 2.0 * a4 / (samples_per_dim - 1)
                            action[5] = -1.0 + 2.0 * a5 / (samples_per_dim - 1)

                            with torch.no_grad():
                                rep = rep_trunk(state.unsqueeze(0),
                                                action.unsqueeze(0)).cpu().numpy()
                                #normalize rep
                                rep = rep / np.linalg.norm(rep, axis=1, keepdims=True)
                            
                            action_t = action.cpu().numpy()
                            df_rep.append({
                                'a_0': action_t[0],
                                'a_1': action_t[1],
                                'a_2': action_t[2],
                                'a_3': action_t[3],
                                'a_4': action_t[4],
                                'a_5': action_t[5],
                                'left_leg_sum': action_t[0] + action_t[1] + action_t[2],
                                'right_leg_sum': action_t[3] + action_t[4] + action_t[5],
                                'rep_mean': np.mean(rep),
                                'rep_sum': np.sum(rep),                                
                            })
                                                        
    import pandas as pd

    df = pd.DataFrame(df_rep)
    df.to_csv("representation_space.csv", index=False)
    print("Visualizing representation space...")
    
    sns.pairplot(df, vars=['a_0', 'a_1', 'a_2', 'a_3', 'a_4', 'a_5', 'rep_mean'])
    plt.suptitle("Representation Space Pairplot", y=1.02)
    plt.savefig(OUTPUT_DIR / "representation_space_pairplot.png")
    plt.close()
        
    
    #use seaborn to plot heatmaps of rep_mean vs left_leg_sum and right_leg_sum
    plt.figure(figsize=(10, 8))
    pivot_table = df.pivot_table(values='rep_mean',
                                 index='left_leg_sum',
                                 columns='right_leg_sum')
    sns.heatmap(pivot_table, cmap='viridis')
    plt.title("Representation Mean vs Left Leg Sum and Right Leg Sum")
    plt.xlabel("Right Leg Sum")
    plt.ylabel("Left Leg Sum")
    plt.savefig(OUTPUT_DIR / "representation_mean_heatmap.png")
    plt.close()
    
    plt.figure(figsize=(10, 8))
    pivot_table = df.pivot_table(values='rep_sum',
                                 index='left_leg_sum',
                                 columns='right_leg_sum')
    sns.heatmap(pivot_table, cmap='viridis')
    plt.title("Representation Sum vs Left Leg Sum and Right Leg Sum")
    plt.xlabel("Right Leg Sum")
    plt.ylabel("Left Leg Sum")
    plt.savefig(OUTPUT_DIR / "representation_sum_heatmap.png")
    plt.close()

if __name__ == "__main__":
    visualize_representation_space()
    
    