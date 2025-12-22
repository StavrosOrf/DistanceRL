import os
import re
from typing import Optional
from dist_rl.config import DistRLConfig

# Target tasks for full runs
TASKS = {
    "Walker2d-v5": 5_000_000,
    # "Humanoid-v5": 10_000_000,
}

# Run all ablations
ABLATION_ALGOS = [
    # "DistAgent",  # baseline
    # "DistAblationA1",
    # "DistAblationA2",
    # "DistAblationA3",
    # "DistAblationA4",
    # "DistAblationA5",
    # "DistAblationB1",
    # "DistAblationB2",
    # "DistAblationB3",
    # "DistAblationB4",
    # "DistAblationB5",
    # "DistAblationB6",
    # "DistAblationB7",
    # "DistAblationB8",
    "MICo",
    "DBC",
    "DBCDet",
]


def build_command(env: str, algo: str, seed: int, project_name: str = "DistRL_Ablations", device: str = "cuda") -> str:
    cfg = getattr(DistRLConfig(), env.split('-')[0].lower(), {})
    steps = TASKS[env]
    exp_prefix = f"{algo}_seed{seed}_{env}"

    cmd = (
        "python main.py"
        f" --env-id {env}"
        f" --algo {algo}"
        f" --device {device}"
        f" --seed {seed}"
        f" --exp-prefix {exp_prefix}"
        f" --group-name Ablations_"
        f" --project_name {project_name}"
        f" --total-steps {steps}"
        " --log_to_wandb"
        " --lightweight_wandb"
    )

    if cfg:
        cmd += (
            f" --batch-size {cfg['batch_size']}"
            f" --K {cfg['K']}"
            f" --lr {cfg['lr']}"
            f" --hidden-size {cfg['hidden_size']}"
            f" --buffer-size {cfg['buffer_size']}"
            f" --eval-episodes {cfg['eval_episodes']}"
            f" --eval-freq {cfg['eval_freq']}"
            f" --expl-sigma {cfg['expl_sigma']}"
            f" --normalize-obs {cfg['normalize_obs']}"
            f" --updates-per-step {cfg['updates_per_step']}"
            f" --rep-gamma-shape {cfg['rep_gamma_shape']}"
            f" --target-entropy-scale {cfg['target_entropy_scale']}"
            f" --kernel-adaptive-tau {cfg['kernel_adaptive_tau']}"
        )
    return cmd


def _tmux_safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def write_and_submit(commands, partition: str = "gpu", hours: int = 12, cpu_cores: int = 1, use_tmux: bool = False, run_name_prefix: Optional[str] = None):
    if not os.path.exists('./slurm_logs'):
        os.makedirs('./slurm_logs')

    for idx, cmd in enumerate(commands):
        base_name = run_name_prefix or "ablate"
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
#SBATCH --job-name="distrl_ablate"
#SBATCH --partition={partition}
#SBATCH --time={hours}:00:00
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpu_cores}
#SBATCH --mem-per-cpu=6000
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
    seeds = [0]
    algos = ABLATION_ALGOS
    use_tmux = True  # set to False to use SLURM
    for env in TASKS:
        hours = 36 if env == "Humanoid-v5" else 20
        cpu_cores = 3 if env == "Humanoid-v5" else 1
        for algo in algos:
            for seed in seeds:
                cmd = build_command(env, algo, seed)
                prefix = _tmux_safe_name(f"{algo}_{env}_seed{seed}")
                write_and_submit([cmd], hours=hours, cpu_cores=cpu_cores, use_tmux=use_tmux, run_name_prefix=prefix)


if __name__ == "__main__":
    main()
