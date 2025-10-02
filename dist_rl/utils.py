import numpy as np
import torch
from typing import Tuple


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
        # print(f"Buffer ptr: {self.ptr}, max size: {self.max_size}, batch size: {batch_size}, entry count: {self.entry_count}")
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

