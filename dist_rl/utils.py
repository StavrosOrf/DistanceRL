import numpy as np
import torch
from typing import Tuple
import random

class RolloutBuffer:
    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int, device):
        self.obs = torch.zeros((buffer_size, obs_dim),
                               dtype=torch.float32, device=device)
        self.next_obs = torch.zeros(
            (buffer_size, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(
            (buffer_size, act_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(
            (buffer_size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros(
            (buffer_size,), dtype=torch.float32, device=device)
        self.ptr = -1
        self.entry_count = 0
        self.max_size = buffer_size
        self.device = device

    def add(self, obs, next_obs, action, reward, done):
        self.ptr += 1
        self.entry_count += 1
        if self.ptr >= self.max_size:
            self.ptr = 0  # Overwrite when buffer is full

        self.obs[self.ptr] = torch.tensor(
            obs, dtype=torch.float32, device=self.device)
        self.next_obs[self.ptr] = torch.tensor(
            next_obs, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.tensor(
            action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = torch.tensor(
            reward, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.tensor(
            done, dtype=torch.float32, device=self.device)

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:        
        if self.entry_count >= self.max_size:
            idxs = np.random.choice(
                self.max_size, size=batch_size, replace=False)
        else:
            idxs = np.random.choice(self.ptr, size=batch_size, replace=False)

        return (
            self.obs[idxs].detach(),
            self.next_obs[idxs].detach(),
            self.actions[idxs].detach(),
            self.rewards[idxs].detach(),
            self.dones[idxs].detach(),
        )


class Trajectory_ReplayBuffer(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_episode_length,
                 device=None,
                 max_size=int(1e4)):

        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.max_length = max_episode_length

        self.state = torch.zeros((max_size, max_episode_length, state_dim))
        self.action = torch.zeros((max_size, max_episode_length, action_dim))
        self.rewards = torch.zeros((max_size, max_episode_length))
        self.dones = torch.zeros((max_size, max_episode_length))
        self.traj_lengths = np.zeros((max_size,), dtype=int)

        self.device = device

    def add(self, state, action, reward, done, env_step):
        self.state[self.ptr, :, :] = state
        self.action[self.ptr, :, :] = action
        self.rewards[self.ptr, :] = reward.squeeze()
        self.dones[self.ptr, :] = done.squeeze()
        self.traj_lengths[self.ptr] = env_step

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # Example of the sample method in utils.py
    def sample(self, batch_size, sequence_length):
        ind = np.random.randint(0, self.size, size=batch_size)
        start = np.random.randint(
            0, self.max_length - sequence_length, size=batch_size)
        end = start + sequence_length

        # Ensure ind, start, and end are integers
        ind = ind.astype(int)
        start = start.astype(int)
        end = end.astype(int)

        # Sample states and actions
        states = torch.FloatTensor(self.state[ind, :, :]).to(self.device)
        actions = torch.FloatTensor(self.action[ind, :, :]).to(self.device)
        next_states = torch.FloatTensor(self.state[ind, :, :]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[ind, :]).to(self.device)
        dones = torch.FloatTensor(self.dones[ind, :]).to(self.device)

        states = [states[i, start[i]:end[i], :]
                  for i in range(batch_size)]
        next_states = [next_states[i, start[i]:end[i], :]
                       for i in range(batch_size)]
        actions = [actions[i, start[i]:end[i], :]
                   for i in range(batch_size)]
        rewards = [rewards[i, start[i]:end[i]]
                   for i in range(batch_size)]
        dones = [dones[i, start[i]:end[i]]
                 for i in range(batch_size)]

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)

        return states, actions, next_states, rewards, dones

    def get_batch(self, batch_size, seq_length):
        
        assert seq_length <= self.max_length, f"seq_length {seq_length} exceeds max_length {self.max_length}"
                
        ind = np.random.randint(0, self.size-1, size=batch_size)

        # Ensure ind, start, and end are integers
        ind = ind.astype(int)

        # Sample states and actions
        states = torch.FloatTensor(self.state[ind, :, :]).to(self.device)
        actions = torch.FloatTensor(self.action[ind, :, :]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[ind, :]).to(self.device)

        dones = torch.FloatTensor(self.dones[ind, :]).to(self.device)
        traj_lengths = self.traj_lengths[ind]
        # print(f"traj_lengths: {traj_lengths}")

        if np.any(traj_lengths <= seq_length):
            traj_lengths = np.maximum(traj_lengths, seq_length)
            # print(f'Changed traj_lengths: {traj_lengths}')            
            print(f'!!! It is recommended to use seq_length smaller than the minimum traj_length in the buffer.')
            start = np.random.randint(
                0, traj_lengths - seq_length + 1, size=batch_size)                
        else:
            start = np.random.randint(
                0, traj_lengths - seq_length, size=batch_size)                
        
        start = start.astype(int)

        states_new = torch.zeros(
            (batch_size, seq_length, states.shape[-1]), device=self.device)
        actions_new = torch.zeros(
            (batch_size, seq_length, actions.shape[-1]), device=self.device)
        rewards_new = torch.zeros((batch_size, seq_length), device=self.device)
        dones_new = torch.ones((batch_size, seq_length), device=self.device)
        
        
        for i in range(batch_size):
            s = start[i]                        
            states_new[i, :, :] = states[i, s:s+seq_length, :]
            actions_new[i, :, :] = actions[i, s:s+seq_length, :]
            rewards_new[i, :] = rewards[i, s:s+seq_length]
            dones_new[i, :] = dones[i, s:s+seq_length]

        return states_new.detach(), actions_new.detach(), rewards_new.detach(), dones_new.detach()

class RTGRolloutBuffer:
    """
    Replay buffer with:
      - full Return-To-Go (RTG) back-filled at episode end,
      - n-step returns (no bootstrap) for quick, lower-variance targets.

    get_batch(...) returns: obs, next_obs, actions, rewards, dones, rtg, nstep
    """
    def __init__(self,
                 buffer_size: int,
                 obs_dim: int,
                 act_dim: int,
                 device,
                 gamma: float = 0.99,
                 n_step: int = 20):
        self.max_size = int(buffer_size)
        self.device   = torch.device(device)
        self.gamma    = float(gamma)
        self.n_step   = int(n_step)

        self.obs      = torch.zeros((self.max_size, obs_dim),  dtype=torch.float32, device=self.device)
        self.next_obs = torch.zeros((self.max_size, obs_dim),  dtype=torch.float32, device=self.device)
        self.actions  = torch.zeros((self.max_size, act_dim),  dtype=torch.float32, device=self.device)
        self.rewards  = torch.zeros((self.max_size,),          dtype=torch.float32, device=self.device)
        self.dones    = torch.zeros((self.max_size,),          dtype=torch.float32, device=self.device)

        # targets
        self.rtg      = torch.zeros((self.max_size,),          dtype=torch.float32, device=self.device)  # full MC
        self.nreturn  = torch.zeros((self.max_size,),          dtype=torch.float32, device=self.device)  # n-step sum

        self.ptr = -1               # last written index
        self.entry_count = 0        # number of written entries (<= max_size)
        self.active_entries = 0      # number of entries with valid RTG & n-step (<= entry_count)

        # track current episode indices to back-fill RTG & n-step at 'done'
        self._ep_idx = []           # python list of integer indices (cyclic indices inside buffer)

    def _advance_ptr(self) -> int:
        self.ptr += 1
        if self.ptr >= self.max_size:
            self.ptr = 0
        if self.entry_count < self.max_size:
            self.entry_count += 1
        return self.ptr

    def add(self, obs, next_obs, action, reward, done: bool):
        i = self._advance_ptr()

        self.obs[i]      = torch.as_tensor(obs,      dtype=torch.float32, device=self.device)
        self.next_obs[i] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        self.actions[i]  = torch.as_tensor(action,   dtype=torch.float32, device=self.device)
        self.rewards[i]  = torch.as_tensor(reward,   dtype=torch.float32, device=self.device)
        self.dones[i]    = torch.as_tensor(float(done), dtype=torch.float32, device=self.device)

        # track episode indices
        self._ep_idx.append(i)

        # If episode ended, back-fill full RTG & n-step returns for that episode window
        if done:
            self._backfill_episode(self._ep_idx)
            self._ep_idx.clear()
            self.active_entries = self.entry_count  # all entries now have valid targets

    @torch.no_grad()
    def _backfill_episode(self, ep_indices):
        """Compute MC RTG and n-step returns for the finished episode."""
        # ep_indices are in chronological order for this episode
        # 1) Full MC RTG (backwards)
        G = 0.0
        for t_rev, idx in enumerate(reversed(ep_indices)):
            r = float(self.rewards[idx].item())
            G = r + self.gamma * G
            self.rtg[idx] = G

        # 2) n-step truncated return (forward window sums)
        T = len(ep_indices)
        r_cpu = self.rewards[ep_indices].detach().cpu().numpy()  # faster loop on CPU
        gammas = np.power(self.gamma, np.arange(self.n_step, dtype=np.float32))
        for t in range(T):
            i = ep_indices[t]
            horizon = min(self.n_step, T - t)  # truncate at episode end
            if horizon <= 0:
                self.nreturn[i] = 0.0
                continue
            self.nreturn[i] = float((r_cpu[t:t+horizon] * gammas[:horizon]).sum())

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        effective = min(self.max_size, self.active_entries)
        replace = effective < batch_size
        idxs = np.random.choice(effective, size=batch_size, replace=replace)
        idxs = torch.as_tensor(idxs, device=self.device, dtype=torch.long)

        return (
            self.obs[idxs],
            self.next_obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.dones[idxs],
            self.rtg[idxs],
            self.nreturn[idxs],
        )
        
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise."""
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = None
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.size) * self.mu
    
    def sample(self):
        dx = self.theta * (self.mu - self.state) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.randn(self.size)
        self.state += dx
        return self.state
    
    def set_sigma(self, sigma):
        self.sigma = sigma

def load_hyperparameters(env_id: str, agent_type: str = "distrl"):
    """
    Load hyperparameters from YAML file for a specific environment.
    
    Args:
        env_id: Environment ID (e.g., "HalfCheetah-v4")
        agent_type: Type of agent ("distrl" or "stoch_distrl")
    
    Returns:
        Dictionary of hyperparameters
    
    Example:
        >>> params = load_hyperparameters("HalfCheetah-v4", "distrl")
        >>> agent = DistanceAgent(env_id="HalfCheetah-v4", **params)
    """
    import yaml
    from pathlib import Path
    
    # Determine the path to the hyperparameters file
    current_dir = Path(__file__).parent
    hyperparam_file = current_dir / "hyperparams" / f"{agent_type}.yaml"
    
    if not hyperparam_file.exists():
        print(f"Warning: Hyperparameter file {hyperparam_file} not found.")
        print("Using default parameters.")
        return {}
    
    # Load the YAML file
    with open(hyperparam_file, 'r') as f:
        all_hyperparams = yaml.safe_load(f)
    
    # Get parameters for the specific environment, or use defaults
    if env_id in all_hyperparams:
        params = all_hyperparams[env_id]
        print(f"Loaded hyperparameters for {env_id} from {agent_type}.yaml")
    else:
        params = all_hyperparams.get('default', {})
        print(f"Environment {env_id} not found in {agent_type}.yaml, using defaults")
    
    return params

class BetaEMA:
    """Keeps an EMA of the 95th percentile gap to stabilize scaling across batches."""
    def __init__(self, decay: float = 0.99, eps: float = 1e-6):
        self.decay = decay
        self.eps = eps
        self.val = None
    def update(self, beta_batch: torch.Tensor):
        b = float(beta_batch.item())
        if self.val is None:
            self.val = b
        else:
            self.val = self.decay * self.val + (1 - self.decay) * b
        return self.val