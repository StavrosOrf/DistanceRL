import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List
import gymnasium as gym


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action and value."""
        shared_features = self.shared(state)
        action = self.actor(shared_features)
        value = self.critic(shared_features)
        return action, value
    
    def get_action(self, state: torch.Tensor, action_std: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action with noise for exploration."""
        action_mean, value = self.forward(state)
        action_std_tensor = torch.full_like(action_mean, action_std)
        dist = torch.distributions.Normal(action_mean, action_std_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value


class PPOBuffer:
    """Experience buffer for PPO."""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.states = torch.zeros((capacity, state_dim))
        self.actions = torch.zeros((capacity, action_dim))
        self.rewards = torch.zeros(capacity)
        self.values = torch.zeros(capacity)
        self.log_probs = torch.zeros(capacity)
        self.dones = torch.zeros(capacity, dtype=torch.bool)
        self.ptr = 0
        self.size = 0
    
    def store(self, state, action, reward, value, log_prob, done):
        """Store experience."""
        self.states[self.ptr] = torch.tensor(state, dtype=torch.float32)
        self.actions[self.ptr] = torch.tensor(action, dtype=torch.float32)
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def get_batch(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> dict:
        """Get batch with computed advantages."""
        # Compute advantages using GAE
        advantages = torch.zeros_like(self.rewards)
        returns = torch.zeros_like(self.rewards)
        
        last_gae = 0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = 0
                next_non_terminal = 0
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1 - self.dones[t + 1].float()
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            returns[t] = advantages[t] + self.values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            'states': self.states[:self.size],
            'actions': self.actions[:self.size],
            'old_log_probs': self.log_probs[:self.size],
            'returns': returns[:self.size],
            'advantages': advantages[:self.size]
        }
    
    def clear(self):
        """Clear buffer."""
        self.ptr = 0
        self.size = 0


class PPO:
    """Proximal Policy Optimization algorithm."""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4, 
                 clip_ratio: float = 0.2, gamma: float = 0.99, gae_lambda: float = 0.95,
                 ppo_epochs: int = 10, buffer_size: int = 2048):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.buffer = PPOBuffer(buffer_size, state_dim, action_dim)
        
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        
    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Select action using current policy."""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state_tensor)
        
        return action.cpu().numpy()[0], log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition in buffer."""
        self.buffer.store(state, action, reward, value, log_prob, done)
    
    def update(self) -> dict:
        """Update policy using PPO."""
        if self.buffer.size < self.buffer.capacity:
            return {}
        
        batch = self.buffer.get_batch(self.gamma, self.gae_lambda)
        
        total_policy_loss = 0
        total_value_loss = 0
        
        for _ in range(self.ppo_epochs):
            # Get current policy predictions
            actions_pred, values_pred = self.policy(batch['states'].to(self.device))
            
            # Compute new log probabilities
            action_std = torch.full_like(actions_pred, 0.1)
            dist = torch.distributions.Normal(actions_pred, action_std)
            new_log_probs = dist.log_prob(batch['actions'].to(self.device)).sum(-1)
            
            # Compute ratio for PPO
            ratio = torch.exp(new_log_probs - batch['old_log_probs'].to(self.device))
            
            # Compute surrogate losses
            advantages = batch['advantages'].to(self.device)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            
            # Policy loss
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            returns = batch['returns'].to(self.device)
            value_loss = 0.5 * (values_pred.squeeze() - returns).pow(2).mean()
            
            # Total loss
            loss = policy_loss + value_loss
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        self.buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / self.ppo_epochs,
            'value_loss': total_value_loss / self.ppo_epochs
        }
    
    def save(self, filepath: str):
        """Save model."""
        torch.save(self.policy.state_dict(), filepath)
    
    def load(self, filepath: str):
        """Load model."""
        self.policy.load_state_dict(torch.load(filepath, map_location=self.device))