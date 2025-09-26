import pytest
import torch
import numpy as np
import gymnasium as gym
from src.ppo import PPO, ActorCritic, PPOBuffer


class TestActorCritic:
    """Test cases for ActorCritic network."""
    
    def test_init(self):
        """Test ActorCritic initialization."""
        model = ActorCritic(state_dim=3, action_dim=1, hidden_dim=32)
        assert isinstance(model, ActorCritic)
        
        # Check network structure
        assert len(model.shared) == 4  # 2 Linear + 2 ReLU
        assert model.shared[0].in_features == 3
        assert model.shared[0].out_features == 32
        assert model.actor[0].out_features == 1  # Linear layer before Tanh
        assert model.critic.out_features == 1
    
    def test_forward_pass(self):
        """Test forward pass through network."""
        model = ActorCritic(state_dim=3, action_dim=1)
        state = torch.randn(5, 3)  # Batch of 5 states
        
        action, value = model(state)
        
        assert action.shape == (5, 1)
        assert value.shape == (5, 1)
        assert torch.all(torch.abs(action) <= 1.0)  # Tanh output
    
    def test_get_action(self):
        """Test action sampling with exploration."""
        model = ActorCritic(state_dim=3, action_dim=1)
        state = torch.randn(1, 3)
        
        action, log_prob, value = model.get_action(state, action_std=0.1)
        
        assert action.shape == (1, 1)
        assert log_prob.shape == (1,)
        assert value.shape == (1, 1)
        assert isinstance(log_prob.item(), float)


class TestPPOBuffer:
    """Test cases for PPO experience buffer."""
    
    def test_init(self):
        """Test buffer initialization."""
        buffer = PPOBuffer(capacity=100, state_dim=3, action_dim=1)
        
        assert buffer.capacity == 100
        assert buffer.states.shape == (100, 3)
        assert buffer.actions.shape == (100, 1)
        assert buffer.rewards.shape == (100,)
        assert buffer.ptr == 0
        assert buffer.size == 0
    
    def test_store_single(self):
        """Test storing single experience."""
        buffer = PPOBuffer(capacity=100, state_dim=3, action_dim=1)
        
        state = np.array([1.0, 2.0, 3.0])
        action = np.array([0.5])
        reward = 1.0
        value = 0.8
        log_prob = -0.5
        done = False
        
        buffer.store(state, action, reward, value, log_prob, done)
        
        assert buffer.size == 1
        assert buffer.ptr == 1
        assert torch.allclose(buffer.states[0], torch.tensor(state, dtype=torch.float32))
        assert torch.allclose(buffer.actions[0], torch.tensor(action, dtype=torch.float32))
        assert buffer.rewards[0] == reward
        assert buffer.values[0] == value
        assert buffer.log_probs[0] == log_prob
        assert buffer.dones[0] == done
    
    def test_store_multiple(self):
        """Test storing multiple experiences."""
        buffer = PPOBuffer(capacity=3, state_dim=2, action_dim=1)
        
        # Store 5 experiences (more than capacity)
        for i in range(5):
            state = np.array([i, i+1])
            action = np.array([i * 0.1])
            buffer.store(state, action, i, i*0.5, -i*0.1, i % 2 == 1)
        
        assert buffer.size == 3  # Limited by capacity
        assert buffer.ptr == 2  # Wrapped around
    
    def test_get_batch(self):
        """Test getting batch with advantage computation."""
        buffer = PPOBuffer(capacity=10, state_dim=2, action_dim=1)
        
        # Fill buffer with some experiences
        for i in range(10):
            state = np.array([i, i+1])
            action = np.array([0.1])
            reward = 1.0 if i % 2 == 0 else -1.0
            value = 0.5
            log_prob = -0.1
            done = i == 9
            buffer.store(state, action, reward, value, log_prob, done)
        
        batch = buffer.get_batch(gamma=0.99, gae_lambda=0.95)
        
        assert 'states' in batch
        assert 'actions' in batch
        assert 'old_log_probs' in batch
        assert 'returns' in batch
        assert 'advantages' in batch
        
        assert batch['states'].shape == (10, 2)
        assert batch['actions'].shape == (10, 1)
        assert batch['advantages'].shape == (10,)
        
        # Advantages should be normalized (mean ~0, std ~1)
        assert abs(batch['advantages'].mean().item()) < 0.1
        assert abs(batch['advantages'].std().item() - 1.0) < 0.1
    
    def test_clear(self):
        """Test buffer clearing."""
        buffer = PPOBuffer(capacity=10, state_dim=2, action_dim=1)
        
        # Add some data
        buffer.store(np.array([1, 2]), np.array([0.1]), 1.0, 0.5, -0.1, False)
        assert buffer.size == 1
        
        # Clear buffer
        buffer.clear()
        assert buffer.size == 0
        assert buffer.ptr == 0


class TestPPO:
    """Test cases for PPO algorithm."""
    
    def test_init(self):
        """Test PPO initialization."""
        ppo = PPO(state_dim=3, action_dim=1)
        
        assert isinstance(ppo.policy, ActorCritic)
        assert isinstance(ppo.buffer, PPOBuffer)
        assert ppo.clip_ratio == 0.2
        assert ppo.gamma == 0.99
        assert ppo.gae_lambda == 0.95
    
    def test_select_action(self):
        """Test action selection."""
        ppo = PPO(state_dim=3, action_dim=1)
        state = np.array([1.0, 2.0, 3.0])
        
        action, log_prob, value = ppo.select_action(state)
        
        assert action.shape == (1,)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
    
    def test_store_transition(self):
        """Test transition storage."""
        ppo = PPO(state_dim=3, action_dim=1, buffer_size=10)
        
        state = np.array([1.0, 2.0, 3.0])
        action = np.array([0.5])
        reward = 1.0
        value = 0.8
        log_prob = -0.5
        done = False
        
        ppo.store_transition(state, action, reward, value, log_prob, done)
        
        assert ppo.buffer.size == 1
    
    def test_update_empty_buffer(self):
        """Test update with insufficient data."""
        ppo = PPO(state_dim=3, action_dim=1, buffer_size=10)
        
        # Update with empty buffer
        losses = ppo.update()
        assert losses == {}
    
    def test_save_load(self):
        """Test model saving and loading."""
        import tempfile
        import os
        
        ppo = PPO(state_dim=3, action_dim=1)
        
        # Get initial weights
        initial_weights = ppo.policy.state_dict()
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name
        
        try:
            ppo.save(temp_path)
            
            # Create new PPO and load
            ppo2 = PPO(state_dim=3, action_dim=1)
            ppo2.load(temp_path)
            
            # Check weights match
            loaded_weights = ppo2.policy.state_dict()
            for key in initial_weights:
                assert torch.allclose(initial_weights[key], loaded_weights[key])
        
        finally:
            os.unlink(temp_path)


class TestPendulumIntegration:
    """Integration tests with Pendulum environment."""
    
    def test_environment_interaction(self):
        """Test PPO can interact with Pendulum environment."""
        env = gym.make('Pendulum-v1')
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        ppo = PPO(state_dim, action_dim)
        
        state, _ = env.reset()
        assert state.shape == (3,)
        
        action, log_prob, value = ppo.select_action(state)
        assert action.shape == (1,)
        
        next_state, reward, done, truncated, _ = env.step(action)
        assert isinstance(reward, (int, float, np.number))
        assert next_state.shape == (3,)
        
        env.close()
    
    def test_short_training_run(self):
        """Test a short training run doesn't crash."""
        env = gym.make('Pendulum-v1')
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        ppo = PPO(state_dim, action_dim, buffer_size=64)  # Small buffer for quick test
        
        # Run a few episodes
        total_steps = 0
        for episode in range(3):
            state, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            
            while not (done or truncated) and total_steps < 100:
                action, log_prob, value = ppo.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                
                ppo.store_transition(state, action, reward, value, log_prob, done or truncated)
                
                state = next_state
                episode_reward += reward
                total_steps += 1
                
                # Try to update if buffer is full
                if ppo.buffer.size >= ppo.buffer.capacity:
                    losses = ppo.update()
                    assert isinstance(losses, dict)
        
        env.close()
        assert total_steps > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])