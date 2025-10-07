
import numpy as np
import minari
from dist_rl.utils import RTGRolloutBuffer


def load_minari_dataset_into_buffer(dataset_name: str, buffer: RTGRolloutBuffer, device: str = "cpu", max_episodes: int = None):
    """
    Load a Minari dataset into an RTGRolloutBuffer.
    
    Args:
        dataset_name: Name of the Minari dataset (e.g., "halfcheetah-medium-v0")
        buffer: RTGRolloutBuffer instance to populate
        device: Device to use for tensors
        max_episodes: Maximum number of episodes to load (None for all)
    
    Returns:
        Dictionary with dataset statistics
    """
    print(f"\n{'='*70}")
    print(f"Loading Minari dataset: {dataset_name}")
    print(f"{'='*70}")
    
    # Download and load dataset
    try:
        dataset = minari.load_dataset(dataset_name, download=True)
        split_datasets = minari.split_dataset(dataset, sizes=[2], seed=42)
        
        dataset = split_datasets[0]  # Use the first split (10% of data)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Available datasets: {minari.list_local_datasets()}")
        raise
    
    total_episodes = dataset.total_episodes
    episodes_to_load = min(total_episodes, max_episodes) if max_episodes else total_episodes
    
    print(f"Total episodes in dataset: {total_episodes}")
    print(f"Episodes to load: {episodes_to_load}")
    
    # Statistics
    total_steps = 0
    episode_returns = []
    episode_lengths = []
    
    # Load episodes into buffer
    for ep_idx, episode in enumerate(dataset.iterate_episodes()):
        if max_episodes and ep_idx >= max_episodes:
            break
            
        observations = episode.observations
        actions = episode.actions
        rewards = episode.rewards
        terminations = episode.terminations
        truncations = episode.truncations
        
        episode_length = len(rewards)
        episode_return = np.sum(rewards)
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        
        # Add transitions to buffer
        for t in range(episode_length):
            obs = observations[t]
            next_obs = observations[t + 1] if t < episode_length - 1 else observations[t]
            action = actions[t]
            reward = rewards[t]
            done = terminations[t] or truncations[t]
            
            buffer.add(obs, next_obs, action, reward, done)
            total_steps += 1      
              
        if (ep_idx + 1) % 100 == 0:
            print(f"Loaded {ep_idx + 1}/{episodes_to_load} episodes...")
    
    stats = {
        'total_episodes': episodes_to_load,
        'total_steps': total_steps,
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'min_return': np.min(episode_returns),
        'max_return': np.max(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
    }
    
    print(f"\n{'='*70}")
    print("Dataset Statistics:")
    print(f"{'='*70}")
    print(f"Total Episodes:    {stats['total_episodes']}")
    print(f"Total Steps:       {stats['total_steps']}")
    print(f"Mean Return:       {stats['mean_return']:.2f} ± {stats['std_return']:.2f}")
    print(f"Return Range:      [{stats['min_return']:.2f}, {stats['max_return']:.2f}]")
    print(f"Mean Length:       {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
    print(f"{'='*70}\n")
    
    return stats
