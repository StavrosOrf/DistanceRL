from dataclasses import dataclass

@dataclass
class DistRLConfig:
    # Default configuration for DistRL experiments for every environment
    halfcheetah = {
        "total_steps": 1_000_000,
        "buffer_size": 1_000_000,
        "K": 64,
        "batch_size": 256,
        "lr": 1e-3,
        "target_entropy_scale": 1,
        "expl_sigma": 0.2,
        "hidden_size": 256,
        "rep_gamma_shape": 2.0,
        "kernel_adaptive_tau": 1, # 1 for True, 0 for False
        "normalize_obs": 1, # 1 for True, 0 for False
        "eval_episodes": 10,
        "eval_freq": 5000,
        "warmup_steps": 5000,
        "expl_sigma": 0.2,
        "updates_per_step": 1,
    }
    
    humanoid = {
        "total_steps": 1_000_000,
        "buffer_size": 1_000_000,
        "K": 256,
        "batch_size": 256,
        "lr": 1e-3,
        "target_entropy_scale": 1,
        "expl_sigma": 0.2,
        "hidden_size": 256,
        "rep_gamma_shape": 2.0,
        "kernel_adaptive_tau": 1, # 1 for True, 0 for False
        "normalize_obs": 1, # 1 for True, 0 for False
        "eval_episodes": 10,
        "eval_freq": 5000,
        "warmup_steps": 5000,
        "updates_per_step": 1,
    }
    
    ant = {
        "total_steps": 1_000_000,
        "buffer_size": 1_000_000,
        "K": 256,
        "batch_size": 256,
        "lr": 1e-3,
        "target_entropy_scale": 1,
        "expl_sigma": 0.2,
        "hidden_size": 256,
        "rep_gamma_shape": 2.0,
        "kernel_adaptive_tau": 1, # 1 for True, 0 for False
        "normalize_obs": 0, # 1 for True, 0 for False
        "eval_episodes": 10,
        "eval_freq": 5000,
        "warmup_steps": 5000,
        "updates_per_step": 1,
    }