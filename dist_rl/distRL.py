from collections import deque
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

import random
import wandb
import time
from tqdm import tqdm

from dist_rl.models import Actor, Distance
from dist_rl.loss import recursive_nstep_cosine_loss
from dist_rl.utils import (RTGRolloutBuffer,
                           OrnsteinUhlenbeckNoise,
                           set_seed)

class DistanceAgent:
    def __init__(
        self,
        env_id,
        seed,
        K=5,
        total_steps=20_000,
        comp_samples=256,
        buffer_size=10**5,
        update_epochs_policy=1,
        update_epochs_val=1,
        batch_size=64,
        policy_training_start=500,
        val_training_start=500,
        lr=3e-4,
        expl_sigma=0.3,
        hidden_size=64,
        top_k=32,
        dynamic_topk=True,
        device="cpu",
        noise_type="OU",  # "OU" or "scheduled Gaussian"
        wandb_run=None,
        rtg_enabled=True,
        model_save_path='./saved_models/',
        eval_episodes=5,
        eval_freq=1000,
        v_gamma=1.0,
        **kwargs,
    ):
        set_seed(seed)
        self.device = torch.device(device)

        # Env
        self.env = gym.make(env_id)
        self.eval_env = gym.make(env_id)
        assert isinstance(self.env.action_space, gym.spaces.Box)

        # how to get maximum step per episode
        if hasattr(self.env, '_max_episode_steps'):
            self.max_episode_steps = self.env._max_episode_steps
        else:
            raise Warning(
                "Max episode steps not found!!!!, using default: 1000")

        self.obs_dim = int(np.prod(self.env.observation_space.shape))
        self.act_dim = int(np.prod(self.env.action_space.shape))
        min_action = self.env.action_space.low[0]
        max_action = self.env.action_space.high[0]

        # Print algorithm configuration in three columns
        print("\n" + "="*70)
        print(" "*10 + f"TRAINING CONFIGURATION for {env_id} with DistRL")
        print("="*70)

        # Prepare parameters in three columns
        params = [
            ("Seed", seed),
            ("K (n-step)", K),
            ("Total Steps", f"{total_steps:,}"),
            ("Comp Samples", comp_samples),
            ("Buffer Size", f"{buffer_size:,}"),
            ("Update Epochs Policy", update_epochs_policy),
            ("Update Epochs Val", update_epochs_val),
            ("Batch Size", batch_size),
            ("Policy Train Start", policy_training_start),
            ("Val Train Start", val_training_start),
            ("Learning Rate", lr),
            ("Exploration Sigma", expl_sigma),
            ("Hidden Size", hidden_size),
            ("Device", device),
            ("RTG Enabled", rtg_enabled),
            ("Eval Episodes", eval_episodes),
            ("Eval Frequency", eval_freq),
            ("V Gamma", v_gamma),
            ("Noise Type", noise_type),
            ("Top K", top_k),
            ("Dynamic TopK", dynamic_topk),
            ("Model Save Path", model_save_path),
        ]

        # Print in three columns
        for i in range(0, len(params), 3):
            row = params[i:i+3]
            line = ""
            for param_name, param_value in row:
                line += f"{param_name:.<15s} {str(param_value):<8s}   "
            print(line)

        print("="*70)
        print(f"Observation Space: {self.env.observation_space}")
        print(f"Action Space: {self.env.action_space}")
        print(f"Max Episode Steps: {self.max_episode_steps}")
        print("="*70 + "\n")

        # Model
        self.actor = Actor(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_size=hidden_size,
            max_action=max_action,
            min_action=min_action
        ).to(self.device)

        self.distance = Distance(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_size=hidden_size,
        ).to(self.device)
        

        self.actor_target = copy.deepcopy(self.actor)
        self.distance_target = copy.deepcopy(self.distance)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),    lr=lr)
        self.distance_optimizer = optim.Adam(
            self.distance.parameters(), lr=lr)

        self.buffer = RTGRolloutBuffer(
            buffer_size=buffer_size,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            n_step=K,
            device=self.device,
        )

        self.model_save_path = model_save_path

        self.seed = seed
        self.K = K
        self.top_k = top_k
        self.steps_collected = 0
        self.distance_training_start = val_training_start
        self.total_steps = total_steps
        self.update_epochs_policy = update_epochs_policy
        self.update_epochs_val = update_epochs_val
        self.batch_size = batch_size
        self.policy_training_start = policy_training_start
        self.val_training_start = val_training_start
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.comp_samples = comp_samples
        self.rtg_enabled = rtg_enabled
        self.noise_type = noise_type
        self.tau = 0.005
        self.discount = 0.99

        self.expl_sigma = expl_sigma
        self.expl_sigma_start = expl_sigma
        self.expl_sigma_final = 0.05
        self.expl_decay_steps = self.total_steps * 0.8
        
        # Top-k scheduler: starts at 5, linearly increases to top_k over half training
        self.top_k_start = 5
        self.top_k = top_k
        self.top_k_rampup_steps = self.total_steps * 0.5
        self.top_k_current = self.top_k_start
        self.dynamic_topk = dynamic_topk
        
        self.best_reward = -float('inf')

        self.v_gamma = v_gamma
        self.wandb_run = wandb_run

        # Initialize Ornstein-Uhlenbeck noise for exploration
        self.ou_noise = OrnsteinUhlenbeckNoise(
            size=self.act_dim,
            mu=0.0,
            theta=0.15,
            sigma=self.expl_sigma,
            dt=1e-2
        )

        if self.wandb_run is not None:
            wandb.run.log_code(".")

    def _exploration_sigma(self):
        """Linear decay: 0..decay_steps ⇒ sigma goes start → final."""
        t = float(self.steps_collected)
        frac = min(1.0, t / max(1, self.expl_decay_steps))
        # linear interpolation
        return self.expl_sigma_start + frac * (self.expl_sigma_final - self.expl_sigma_start)

    def _get_current_top_k(self):
        """
        Linear ramp-up for top_k: starts at 5, increases to top_k over half training steps.
        After rampup_steps, stays at top_k.
        """
        t = float(self.steps_collected)
        if t >= self.top_k_rampup_steps:
            return int(self.top_k)
        
        # Linear interpolation from top_k_start to top_k
        frac = t / max(1, self.top_k_rampup_steps)
        current_k = self.top_k_start + frac * (self.top_k - self.top_k_start)
        return int(max(self.top_k_start, current_k))
    
    def differentiable_topk(self, S_full, k_per_sample):
        """
        Fast vectorized differentiable top-k selection with per-sample k values.
        
        Args:
            S_full: [B, M] tensor of similarities/scores
            k_per_sample: [B] int tensor where each element specifies k for that sample
        
        Returns:
            S_masked: [B, M] tensor with -inf for non-top-k elements, preserving gradients
            top_vals: [B, K_max] tensor of top-k values (padded with -inf)
            top_idx: [B, K_max] tensor of top-k indices (padded with last valid index)
        """
        B, M = S_full.shape
        device = S_full.device
        
        # Get maximum k across all samples
        K_max = int(k_per_sample.max().item())
        K_max = min(K_max, M)  # Ensure K_max doesn't exceed available elements
        
        # Get top-K_max for all samples at once (vectorized)
        top_vals_full, top_idx_full = torch.topk(S_full, k=K_max, dim=1, largest=True)  # [B, K_max]
        
        # Create a mask for valid top-k elements per sample
        # For each row, we want to keep only the first k_per_sample[i] elements
        k_mask = torch.arange(K_max, device=device).unsqueeze(0).expand(B, -1)  # [B, K_max]
        valid_mask = k_mask < k_per_sample.unsqueeze(1)  # [B, K_max]
        
        # Apply mask to top_vals (set invalid entries to -inf)
        top_vals = torch.where(valid_mask, top_vals_full, torch.tensor(float('-inf'), device=device))
        top_idx = top_idx_full  # Keep all indices for reference
        
        # Create the masked similarity matrix S_masked
        # For each sample, get the threshold (minimum value to include)
        # We need to handle the case where k_per_sample[i] might be 0
        k_clamped = k_per_sample.clamp(min=1, max=K_max)  # [B]
        
        # Gather the k-th largest value for each sample (the threshold)
        # Use k_clamped - 1 as index since we're 0-indexed
        batch_idx = torch.arange(B, device=device)
        threshold_vals = top_vals_full[batch_idx, k_clamped - 1]  # [B]
        
        # Create differentiable mask: S_full >= threshold for each row
        threshold_vals = threshold_vals.unsqueeze(1)  # [B, 1]
        S_masked = torch.where(
            S_full >= threshold_vals,
            S_full,
            torch.tensor(float('-inf'), device=device)
        )  # [B, M]
        
        # Handle edge case: if k_per_sample[i] == 0, mask everything
        zero_k_mask = (k_per_sample == 0).unsqueeze(1)  # [B, 1]
        S_masked = torch.where(zero_k_mask, torch.tensor(float('-inf'), device=device), S_masked)
        
        return S_masked, top_vals, top_idx

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.tensor(
            obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(obs_tensor).detach().cpu().numpy()[0]

        return action

    def train_distance(self):

        for _ in range(self.update_epochs_val):
            start_time = time.time()
            obs, next_obs, actions, rewards, dones, rtg, n_returns = self.buffer.get_batch(
                self.batch_size)

            if self.rtg_enabled:
                rewards = rtg
            else:
                rewards = n_returns

            d_embeddings = self.distance(obs, actions)

            with torch.no_grad():
                next_actions = self.actor_target(next_obs)
                
                # add noise to next actions for smoothing
                # noise = (torch.randn_like(next_actions) * 0.2).clamp(-0.5, 0.5)
                # next_actions = (next_actions + noise).clamp(
                #     self.env.action_space.low[0], self.env.action_space.high[0])
                
                d_embeddings_next = self.distance_target(
                    next_obs, next_actions)
                
            distance_loss, info = recursive_nstep_cosine_loss(
                embeddings=d_embeddings,
                next_embeddings=d_embeddings_next,
                dones=dones,
                nreturns=rewards,
                discount=self.discount,
                n=self.K,
                gamma_shape=self.v_gamma,
            )

            self.distance_optimizer.zero_grad()
            distance_loss.backward()
            # calculate the norm of the gradients, dont clip
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.distance.parameters(), max_norm=1.0)
            self.distance_optimizer.step()

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {"train/dist_loss": distance_loss.item(),
                     **{f"train/dist_{k}": v for k, v in info.items()},
                     "train/dist_grad_norm": grad_norm,
                     "time/dist_step_time": time.time() - start_time},
                    step=self.steps_collected)

    def train_policy(self):
        """
        Deterministic policy update that uses ONLY the distance metric's cosine similarity.
        - For each batch state s_i, embed z_i = distance(s_i, actor(s_i)).
        - Build a large candidate pool {(s_c, a_c, rtg_c)} from replay.
        - Embed all candidates z_c = distance(s_c, a_c) (no-grad).
        - Select top-K neighbors per row by cosine(z_i, z_c).
        - Convert their RTGs to a row-wise target distribution P_tgt.
        - Convert similarities to P_pred.
        - Minimize CE(P_tgt || P_pred).
        """
        start_time = time.time()

        # ---- sample current batch & large candidate pool ----
        obs, _, _, _, _, _, _ = self.buffer.get_batch(self.batch_size)
        obs_c, _, act_c, _, _, rtg_c_dis, n_returns = self.buffer.get_batch(
            self.comp_samples)

        if self.rtg_enabled:
            returns = rtg_c_dis
        else:
            returns = n_returns

        # ---- freeze distance during policy update ----
        for p in self.distance.parameters():
            p.requires_grad = False

        # current actions and embeddings
        a_pred = self.actor(obs)                               # [B, A]
        z_i = self.distance(obs, a_pred)                    # [B, H]
        z_i = nn.functional.normalize(z_i, p=2, dim=1)      # cosine

        # candidate embeddings (no-grad)
        with torch.no_grad():
            z_c = self.distance(obs_c, act_c)                  # [M, H]
            z_c = nn.functional.normalize(z_c, p=2, dim=1)

        # ---- cosine similarities & top-K selection by cosine ----
        S_full = (z_i @ z_c.T)  # / max(1e-6, tau_sim)           # [B, M]
        if self.dynamic_topk:
            K_eff = min(self._get_current_top_k(), S_full.size(1))
        else:
            K_eff = min(self.top_k, S_full.size(1))
        
        # Get top-K with minimum similarity threshold
        top_vals, top_idx = torch.topk(S_full, k=K_eff, dim=1, largest=True)
        
        # Apply minimum similarity threshold - filter out low similarities
        min_similarity_threshold = 0.0  # Can be tuned (e.g., 0.1, 0.2)
        valid_mask = top_vals > min_similarity_threshold  # [B, K_eff]
        
        # Ensure at least one neighbor per batch element
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        
        # For elements below threshold, set them to -inf so they get zero weight in softmax
        top_vals_filtered = torch.where(valid_mask, top_vals, torch.tensor(float('-inf'), device=top_vals.device))

        # gather per-row top-K candidates
        RTG_top = returns.index_select(
            0, top_idx.reshape(-1)).reshape(a_pred.size(0), K_eff)      # [B,K]
        # [B,K] - use filtered values for policy training
        S_top = top_vals_filtered

        # ---- target (future-aware) distribution from RTG ----
        # Only include valid neighbors (above threshold) in the distribution
        # Mask out RTG values for invalid neighbors
        RTG_top_masked = torch.where(valid_mask, RTG_top, torch.tensor(0.0, device=RTG_top.device))
        
        # per-row baseline for stability (advantage-like) - computed only over valid neighbors
        valid_sum = (RTG_top_masked * valid_mask.float()).sum(dim=1, keepdim=True)
        G_base = valid_sum / valid_counts.float()
        
        scores = (RTG_top - G_base) * valid_mask.float()  # Zero out invalid scores
        P_tgt = torch.softmax(scores, dim=1)           # [B,K]

        # ---- predicted distribution from similarities ----
        S_shift = S_top - S_top.max(dim=1, keepdim=True).values
        # [B,K]
        P_pred = torch.softmax(S_shift, dim=1)
        
        # print(f'p_pred: {P_pred}')
        # print(f'p_tgt: {P_tgt}')
        # print(f'P_pred.clamp_min(eps).log(): {P_pred.clamp_min(1e-8).log()}')                
        # print(f'ce: {(P_tgt * (P_pred.clamp_min(1e-8).log()))}')
        # print(f'ce2: {(P_tgt * (P_pred.clamp_min(1e-8).log())).sum(dim=1)}\n')

        # ---- main loss: cross-entropy CE(P_tgt || P_pred) ----
        eps = 1e-8
        ce = -(P_tgt * (P_pred.clamp_min(eps).log())).sum(dim=1).mean()
        policy_loss = ce

        # ---- optimize actor ----
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        # unfreeze distance
        for p in self.distance.parameters():
            p.requires_grad = True

        # ---- logs ----
        if self.wandb_run is not None:
            self.wandb_run.log({
                "train_p/policy_loss": policy_loss.item(),
                "train_p/rtg_c_mean": returns.mean().item(),
                "train_p/rtg_c_max": returns.max().item(),
                "train_p/topk_mean": S_top.mean().item(),
                "train_p/current_top_k": K_eff,
                "time/policy_step_time": time.time() - start_time
            }, step=self.steps_collected)

        # Update the frozen target models
        for param, target_param in zip(self.distance.parameters(), self.distance_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def train_policy_reward_only(self):
        start_time = time.time()

        # ---- sample current batch & large candidate pool ----
        obs, _, _, _, _, _, _ = self.buffer.get_batch(self.batch_size)
        obs_c, _, act_c, _, _, rtg_c_dis, n_returns = self.buffer.get_batch(
            self.comp_samples)

        if self.rtg_enabled:
            returns = rtg_c_dis
        else:
            returns = n_returns

        # ---- freeze distance during policy update ----
        for p in self.distance.parameters():
            p.requires_grad = False

        # current actions and embeddings
        a_pred = self.actor(obs)                               # [B, A]
        z_i = self.distance(obs, a_pred)                    # [B, H]
        z_i = nn.functional.normalize(z_i, p=2, dim=1)      # cosine

        # candidate embeddings (no-grad)
        with torch.no_grad():
            z_c = self.distance(obs_c, act_c)                  # [M, H]
            z_c = nn.functional.normalize(z_c, p=2, dim=1)

        # ---- cosine similarities & top-K selection by cosine ----
        S_full = (z_i @ z_c.T)  # / max(1e-6, tau_sim)           # [B, M]        
        
        # Compute per-sample k based on cosine similarity threshold
        cos_sim_threshold = 0.9
        num_above_threshold = (S_full > cos_sim_threshold).sum(dim=1)  # [B]
        k_per_sample = torch.clamp(num_above_threshold, min=5, max=self.top_k)  # [B]
        
        # Use custom differentiable top-k with per-sample k
        S_masked, top_vals, top_idx = self.differentiable_topk(S_full, k_per_sample)
        
        # Filter out -inf values for logging
        # valid_top_vals = top_vals[top_vals != float('-inf')]
        
        # Debug prints (can be removed later)
        # print(f'k_per_sample: {k_per_sample}')
        # print(f'top_vals (valid): {valid_top_vals}')
        # print(f'top_vals.mean(): {valid_top_vals.mean().item() if valid_top_vals.numel() > 0 else 0.0}')        
        # print(f'S_masked: {S_masked}')
        # print(f'S_full: {S_full}')
        # print(f'top_idx: {top_idx}')
        # print(f'top_vals: {top_vals}\n\n')
        
        
        # Softmax to create differentiable weights [B, M]
        selection_weights = torch.softmax(S_masked, dim=1)  # [B, M]
        
        # print(f'selection_weights: {selection_weights}')
        # print(f'returns: {returns}')
        
        # Differentiable weighted selection of returns
        RTG_top = (selection_weights @ returns.unsqueeze(1)).squeeze(1)  # [B, M] @ [M, 1] -> [B, 1] -> [B]
        
        # Keep S_top for logging
        S_top = top_vals
        
        # print(f'RTG_top: {RTG_top}')
        # print(f'S_top: {S_top}')
        
        # Policy loss: maximize weighted return
        policy_loss = -RTG_top.mean()

        # ---- optimize actor ----
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        # unfreeze distance
        for p in self.distance.parameters():
            p.requires_grad = True

        # ---- logs ----
        if self.wandb_run is not None:
            # Filter out -inf values from S_top for statistics
            valid_S_top = S_top[S_top != float('-inf')]
            
            self.wandb_run.log({
                "train_p/policy_loss": policy_loss.item(),
                "train_p/rtg_c_mean": returns.mean().item(),
                "train_p/rtg_c_max": returns.max().item(),
                # get the mean and std of non-inf values in S_top
                "train_p/topk_mean": valid_S_top.mean().item() if valid_S_top.numel() > 0 else 0.0,
                "train_p/topk_std": valid_S_top.std().item() if valid_S_top.numel() > 0 else 0.0,
                "train_p/topk_min": valid_S_top.min().item() if valid_S_top.numel() > 0 else 0.0,
                "train_p/topk_max": valid_S_top.max().item() if valid_S_top.numel() > 0 else 0.0,
                "train_p/k_per_sample_mean": k_per_sample.float().mean().item(),
                "train_p/k_per_sample_min": k_per_sample.min().item(),
                "train_p/k_per_sample_max": k_per_sample.max().item(),
                "time/policy_step_time": time.time() - start_time
            }, step=self.steps_collected)

        # Update the frozen target models
        for param, target_param in zip(self.distance.parameters(), self.distance_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

    def evaluate_policy(self):
        total_reward = 0.0
        ep_steps = []
        for _ in range(self.eval_episodes):
            eval_obs, _ = self.eval_env.reset()
            eval_done = False
            eval_truncated = False
            env_steps = 0
            while not eval_done and not eval_truncated:
                eval_action = self.get_action(eval_obs)
                eval_obs, eval_reward, eval_done, eval_truncated, _ = self.eval_env.step(
                    eval_action)
                total_reward += eval_reward
                env_steps += 1

            ep_steps.append(env_steps)

        avg_reward = total_reward / self.eval_episodes
        print(
            f"[Eval.] Reward {avg_reward:10.2f}, Steps: {np.mean(ep_steps):6.1f}")  # (Episodes: {self.eval_episodes})")

        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            print(f"  New best reward!")
            self.save(self.model_save_path + f"/best")

        if self.wandb_run is not None:
            self.wandb_run.log({"eval/avg_reward": avg_reward,
                                "eval/best_reward": self.best_reward,
                                "eval/avg_ep_length": np.mean(ep_steps)},
                               step=self.steps_collected)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.distance.state_dict(), filename + "_distance.pth")

        save_path = '/'.join(filename.split('/')[:-1])
        print(f"Model saved to {save_path}/")

    def train(self):
        self.steps_since_eval = 0
        env_step = 0
        ep_reward = 0        

        obs, _ = self.env.reset(seed=self.seed)

        while self.steps_collected < self.total_steps:

            action = self.get_action(obs)

            if self.noise_type == "OU":
                # Update sigma and use Ornstein-Uhlenbeck noise
                self.ou_noise.set_sigma(self.expl_sigma)
                noise = self.ou_noise.sample()
            elif self.noise_type == "SchedOU":
                # use scheduled Gaussian noise
                # Update sigma and use Ornstein-Uhlenbeck noise                
                self.expl_sigma = self._exploration_sigma()   # decays 0.3 → 0.05
                # noise = np.random.normal(0, self.expl_sigma, size=self.act_dim)
                self.ou_noise.set_sigma(self.expl_sigma)
                noise = self.ou_noise.sample()
            else:
                noise = np.random.normal(0, self.expl_sigma, size=self.act_dim)

            # per-dimension clip to env bounds
            low, high = self.env.action_space.low, self.env.action_space.high
            action = np.clip(action + noise, low, high)

            next_obs, reward, done, truncated, _ = self.env.step(action)

            ep_reward += reward

            self.buffer.add(obs, next_obs, action, reward, done or truncated)

            self.steps_collected += 1
            self.steps_since_eval += 1
            env_step += 1

            # Train distance model
            if self.steps_collected > self.val_training_start:
                self.train_distance()

            # Training policy
            if self.steps_collected > self.policy_training_start:
                self.train_policy_reward_only()

            if self.steps_since_eval >= self.eval_freq and self.steps_collected > self.policy_training_start:
                self.evaluate_policy()
                self.steps_since_eval = 0

            obs = next_obs

            if done or truncated:
                if self.steps_collected < self.policy_training_start:
                    print(
                        f"[Collect: {self.steps_collected}/{self.policy_training_start:<5d}] Reward {ep_reward:10.2f}, Steps: {np.mean(env_step):6.1f}")
                else:
                    print(
                        f"[Train: {self.steps_collected}/{self.total_steps:<5d}] Reward {ep_reward:10.2f}, Steps: {np.mean(env_step):6.1f}")

                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {"rollout/ep_reward": ep_reward,
                         "rollout/ep_length": env_step,
                         "rollout/sigma": self.expl_sigma},
                        step=self.steps_collected)

                env_step = 0
                ep_reward = 0
                obs, _ = self.env.reset()
                self.ou_noise.reset()  # Reset OU noise for new episode

        # Final evaluation
        self.evaluate_policy()
        self.save(self.model_save_path + f"/final")
        wandb.finish()



    def train_offline(self):
        """
        Train the agent offline using only the replay buffer.
        """
        print(f"\n{'='*70}")
        print(f"Starting Offline Training")
        print(f"{'='*70}\n")            
            
        
        for iteration in tqdm(range(self.total_steps), desc="Offline Training"):
            self.steps_collected = iteration  # Use iteration count as "steps"
            
            # Train distance model
            for _ in range(self.update_epochs_val):
                self.train_distance()
            
            # Train policy
            for _ in range(self.update_epochs_policy):
                self.train_policy_reward_only()
            
            # Evaluate periodically
            if (iteration + 1) % self.eval_freq == 0:                
                self.evaluate_policy()
                                
        self.evaluate_policy()
        self.save(self.model_save_path + "/final")

        if self.wandb_run is not None:
            wandb.finish()