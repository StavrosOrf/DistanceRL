import os
import re
from typing import Optional

from dist_rl.config import DistRLConfig

# Target tasks for full runs
TASKS = {
    "HalfCheetah-v5": 1_000_000,
    "Humanoid-v5": 1_000_000,
}

# Baseline algorithm to sweep
ALGO = "v2DistAgent"

# Grids for one-at-a-time ablations
REP_GAMMA_SHAPE_GRID = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
K_GRID = [16, 64, 128, 256, 512]


def build_command(env: str, seed: int, rep_gamma_shape: float, K: int,
                  project_name: str = "DistRL_RepK_Ablations", device: str = "cuda") -> str:
    cfg = getattr(DistRLConfig(), env.split('-')[0].lower(), {})
    steps = TASKS[env]
    exp_prefix = f"{ALGO}_gshape{rep_gamma_shape}_K{K}_seed{seed}_{env}"

    cmd = (
        "python main.py"
        f" --env-id {env}"
        f" --algo {ALGO}"
        f" --device {device}"
        f" --seed {seed}"
        f" --exp-prefix {exp_prefix}"
        f" --group-name RepK_Ablation_Fixed"
        f" --project_name {project_name}"
        f" --total-steps {steps}"
        " --log_to_wandb"
        " --lightweight_wandb"
        f" --rep-gamma-shape {rep_gamma_shape}"
        f" --K {K}"
    )

    if cfg:
        cmd += (
            f" --batch-size {cfg['batch_size']}"
            f" --lr {cfg['lr']}"
            f" --hidden-size {cfg['hidden_size']}"
            f" --buffer-size {cfg['buffer_size']}"
            f" --eval-episodes {cfg['eval_episodes']}"
            f" --eval-freq {cfg['eval_freq']}"
            f" --expl-sigma {cfg['expl_sigma']}"
            f" --normalize-obs {cfg['normalize_obs']}"
            f" --updates-per-step {cfg['updates_per_step']}"
            f" --target-entropy-scale {cfg['target_entropy_scale']}"
            f" --kernel-adaptive-tau {cfg['kernel_adaptive_tau']}"
        )
    return cmd


def _tmux_safe_name(name: str) -> str:
    x = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    x += "_" * (2 - (len(x) % 2))
    return x


def write_and_submit(commands, partition: str = "gpu", hours: int = 12, cpu_cores: int = 1,
                     use_tmux: bool = False, run_name_prefix: Optional[str] = None):
    if not os.path.exists('./slurm_logs'):
        os.makedirs('./slurm_logs')

    for idx, cmd in enumerate(commands):
        base_name = run_name_prefix or "repK_ablate"
        run_name = f"{base_name}_{idx}"
        if use_tmux:
            session = _tmux_safe_name(run_name)
            tmux_cmd = (
                "tmux new-session -d -s " + session +
                " 'source ~/.bashrc; conda activate distrl; " + cmd + "'"
            )
            os.system(tmux_cmd)
            print(f"Launched tmux session {session} -> {cmd}")
            continue

        script = f"""#!/bin/bash
#SBATCH --job-name="distrl_repK"
#SBATCH --partition={partition}
#SBATCH --time={hours}:00:00
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpu_cores}
#SBATCH --mem-per-cpu=5300
#SBATCH --account=research-eemcs-ese
#SBATCH --output=./slurm_logs/{run_name}.out
#SBATCH --error=./slurm_logs/{run_name}.err

module load 2024r1 openmpi miniconda3 py-pip
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dt3

srun {cmd}

conda deactivate
"""
        with open('run_tmp.sh', 'w') as f:
            f.write(script)
        with open(f'./slurm_logs/{run_name}.sh', 'w') as f:
            f.write(script)
        os.system('sbatch run_tmp.sh')


def main():
    seeds = [10,20,30,40]
    use_tmux = False  # set to False to use SLURM

    for env in TASKS:
        cfg = getattr(DistRLConfig(), env.split('-')[0].lower(), {})
        hours = 20 if env == "Humanoid-v5" else 15
        cpu_cores = 1
        # cpu_cores = 3 if env == "Humanoid-v5" else 1

        # Stage 1: ablate K only (rep_gamma_shape fixed to config default)
        base_gamma = cfg.get("rep_gamma_shape", 0.5)
        for K in K_GRID:
            for seed in seeds:
                cmd = build_command(env, seed, rep_gamma_shape=base_gamma, K=K)
                prefix = _tmux_safe_name(f"{ALGO}_{env}_K{K}_g{base_gamma}_seed{seed}")
                write_and_submit([cmd], hours=hours, cpu_cores=cpu_cores,
                                 use_tmux=use_tmux, run_name_prefix=prefix)

        # Stage 2: ablate rep_gamma_shape only (K fixed to config default/best)
        base_K = cfg.get("K", 64)
        for rep_gamma_shape in REP_GAMMA_SHAPE_GRID:
            for seed in seeds:
                cmd = build_command(env, seed, rep_gamma_shape=rep_gamma_shape, K=base_K)
                prefix = _tmux_safe_name(f"{ALGO}_{env}_K{base_K}_g{rep_gamma_shape}_seed{seed}")
                write_and_submit([cmd], hours=hours, cpu_cores=cpu_cores,
                                 use_tmux=use_tmux, run_name_prefix=prefix)


if __name__ == "__main__":
    main()
