import gymnasium as gym
import numpy as np
import torch
from src.ppo import PPO
from typing import List


def train_ppo_pendulum(num_episodes: int = 1000, save_path: str = None) -> List[float]:
    """Train PPO on Pendulum environment."""
    
    # Create environment
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create PPO agent
    agent = PPO(state_dim, action_dim)
    
    # Training loop
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action
            action, log_prob, value = agent.select_action(state)
            
            # Environment step
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, value, log_prob, done or truncated)
            
            state = next_state
            episode_reward += reward
        
        # Update policy
        if agent.buffer.size >= agent.buffer.capacity:
            losses = agent.update()
            if episode % 100 == 0 and losses:
                print(f"Episode {episode}: Reward = {episode_reward:.2f}, "
                      f"Policy Loss = {losses['policy_loss']:.4f}, "
                      f"Value Loss = {losses['value_loss']:.4f}")
        
        episode_rewards.append(episode_reward)
        
        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}: Average Reward (last 100) = {avg_reward:.2f}")
    
    # Save model
    if save_path:
        agent.save(save_path)
        print(f"Model saved to {save_path}")
    
    env.close()
    return episode_rewards


def evaluate_ppo_pendulum(model_path: str, num_episodes: int = 10) -> List[float]:
    """Evaluate trained PPO on Pendulum environment."""
    
    # Create environment
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create and load agent
    agent = PPO(state_dim, action_dim)
    agent.load(model_path)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action (deterministic for evaluation)
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action, _ = agent.policy(state_tensor)
                action = action.numpy()[0]
            
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    env.close()
    avg_reward = np.mean(episode_rewards)
    print(f"Average evaluation reward: {avg_reward:.2f}")
    
    return episode_rewards


if __name__ == "__main__":
    print("Training PPO on Pendulum-v1...")
    rewards = train_ppo_pendulum(num_episodes=500, save_path="ppo_pendulum.pth")
    
    print("\nEvaluating trained model...")
    eval_rewards = evaluate_ppo_pendulum("ppo_pendulum.pth", num_episodes=5)