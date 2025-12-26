'''
This script is used to run the batch of training sciprts for every algorithms evaluated
#old command# srun --mpi=pmix --job-name=interactive-gpu --partition=gpu --gres=gpu:1 --qos=normal --time=01:00:00 --mem-per-cpu=4096 --pty /bin/bash -il
srun --mpi=pmix --job-name=interactive-gpu --partition=gpu --gpus-per-task=1 --qos=normal --time=01:00:00 --mem-per-cpu=5300 --cpus-per-task=1 --ntasks=1 --pty --account research-eemcs-ese /bin/bash -il
srun --mpi=pmix --job-name=interactive-gpu --partition=gpu-a100-small --gpus-per-task=1 --qos=normal --time=01:00:00 --mem-per-cpu=7000 --cpus-per-task=1 --ntasks=1 --pty /bin/bash -il
srun --mpi=pmix --job-name=interactive-gpu --partition=gpu-a100 --gpus-per-task=1 --qos=normal --time=01:00:00 --mem-per-cpu=5500 --cpus-per-task=1 --ntasks=1 --pty /bin/bash -il
srun --mpi=pmix --job-name=interactive --partition=compute --cpus-per-task=2 --qos=normal --time=01:00:00 --mem-per-cpu=3800 --ntasks=1 --pty /bin/bash -il
'''

import os
import random
from dist_rl.config import DistRLConfig

steps_per_env = {
    'HalfCheetah-v5': 5_000_000,
    'Ant-v5': 5_000_000,
    'Hopper-v5': 3_000_000,
    'Humanoid-v5': 10_000_000,
    'InvertedDoublePendulum-v5': 500_000,
    'InvertedPendulum-v5': 500_000,
    'Reacher-v5': 500_000,
    'Swimmer-v5': 1_000_000,
    'Walker2d-v5': 5_000_000,
}

hours_per_env_sb3 = {
    'HalfCheetah-v5': 20,
    'Ant-v5': 20,
    'Hopper-v5': 12,
    'Humanoid-v5': 45,
    'InvertedDoublePendulum-v5': 6,
    'InvertedPendulum-v5': 6,
    'Reacher-v5': 3,
    'Swimmer-v5': 5,
    'Walker2d-v5': 12,
}

def batch_runner():
    # ---------------- General configuration ----------------

    partition = 'gpu'  # gpu-a100 # gpu-a100-small # gpu, compute
    algo_list = ['v2DistAgent']
    group_name = "HyperparamGrid"
    project_name = "DistRL_Hyper"  # DistRL_Rep

    # resource configuration
    device = 'cuda' if partition != 'compute' else 'cpu'
    memory_per_cpu = '5300' if partition != 'compute' else '3800'
    memory_per_cpu = "8000" if partition == 'gpu-a100' else memory_per_cpu
    batch_arg = '#SBATCH --gpus-per-task=1' if partition != 'compute' else '\n'

    SB3_ALGOS = ["ppo", "td3", "sac", "tqc"]
    MUJOCO_ENVS = ['HalfCheetah-v5', 'Ant-v5', 'Hopper-v5', 'Humanoid-v5', 'InvertedDoublePendulum-v5',
                'InvertedPendulum-v5', 'Reacher-v5', 'Swimmer-v5', 'Walker2d-v5'] #number of envs: 9
    
    F_ENVS = ['Walker2d-v5', 'HalfCheetah-v5', 'Humanoid-v5']

    # ---------------- Ablation grids ----------------

    seed_grid = [42, 32]
    center_qhat_grid = [0, 1]
    kernel_adaptive_tau_grid = [0]
    rep_gamma_shape_grid = [0.5, 2.0]
    K_grid = [32, 64, 128, 256]

    # if directory does not exist, create it
    if not os.path.exists('./slurm_logs'):
        os.makedirs('./slurm_logs')

    for env in F_ENVS:
        # Default CPU allocation; adjust if heavier envs are added later
        cpu_cores = 1 #if env not in ['Humanoid-v5'] else 3

        for algo in algo_list:
            base_cfg = getattr(DistRLConfig(), env.split('-')[0].lower(), None)
            if base_cfg is None:
                print(f"Skipping {env}: missing base config")
                continue

            # Use longer budgets for non-SB3 algos (similar to batch_runner)
            job_hours = 10 # int(hours_per_env_sb3.get(env, 8) * 1.5)
            if job_hours > 45:
                job_hours = 45

            training_steps = 1_000_000 #steps_per_env.get(env, base_cfg.get("total_steps", 1_000_000))

            for seed in seed_grid:
                for center_qhat in center_qhat_grid:
                    for kernel_adaptive_tau in kernel_adaptive_tau_grid:
                        for rep_gamma_shape in rep_gamma_shape_grid:
                            for K in K_grid:
                                # Use per-env best K when K is None
                                K_val = base_cfg["K"] if K is None else K

                                run_id = (
                                    f"{algo}_{env}_K{K_val}_cq{center_qhat}_kat{kernel_adaptive_tau}"
                                    f"_rgs{rep_gamma_shape}_seed{seed}"
                                )
                                run_name = f"{run_id}_{random.randint(0, 99999)}"
                                print(f"Submitting {run_name}")

                                python_command = 'python main.py' + \
                                    f' --env-id {env}' + \
                                    f' --algo {algo}' + \
                                    f' --device {device}' + \
                                    f' --batch-size {base_cfg["batch_size"]}' + \
                                    f' --K {K_val}' + \
                                    f' --lr {base_cfg["lr"]}' + \
                                    f' --hidden-size {base_cfg["hidden_size"]}' + \
                                    f' --total-steps {training_steps}' + \
                                    f' --buffer-size {base_cfg["buffer_size"]}' + \
                                    f' --seed {seed}' + \
                                    f' --exp-prefix {run_name}' + \
                                    f' --group-name {group_name}' + \
                                    f' --eval-episodes {base_cfg["eval_episodes"]}' + \
                                    f' --eval-freq {base_cfg["eval_freq"]}' + \
                                    f' --expl-sigma {base_cfg["expl_sigma"]}' + \
                                    f' --normalize-obs {base_cfg["normalize_obs"]}' + \
                                    f' --updates-per-step {base_cfg["updates_per_step"]}' + \
                                    f' --rep-gamma-shape {rep_gamma_shape}' + \
                                    f' --project_name {project_name}' + \
                                    f' --target-entropy-scale {base_cfg["target_entropy_scale"]}' + \
                                    f' --kernel-adaptive-tau {kernel_adaptive_tau}' + \
                                    f' --center-qhat {center_qhat}' + \
                                    ' --log_to_wandb' + \
                                    ' --lightweight_wandb'

                                print(python_command)

                                command = '''#!/bin/bash
#SBATCH --job-name="dist_rl"
''' + \
                                    f'#SBATCH --partition={partition}\n' + \
                                    f'#SBATCH --time={job_hours}:00:00\n' + \
                                    f'{batch_arg}' + \
                                    '''
#SBATCH --ntasks=1
''' + \
                                    f'#SBATCH --cpus-per-task={cpu_cores}' + \
                                    '''
''' + \
                                    f'#SBATCH --mem-per-cpu={memory_per_cpu}' + \
                                    '''
#SBATCH --account=research-eemcs-ese

''' + \
                                    f'#SBATCH --output=./slurm_logs/{run_name}.out' + \
                                    '''
''' + \
                                    f'#SBATCH --error=./slurm_logs/{run_name}.err' + \
                                    '''

module load 2024r1 openmpi miniconda3 py-pip

# Set conda env:

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate dt3
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

''' + 'srun ' + python_command + \
                                    '''

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

conda deactivate
'''

                                with open('run_tmp.sh', 'w') as f:
                                    f.write(command)

                                with open(f'./slurm_logs/{run_name}.sh', 'w') as f:
                                    f.write(command)

                                os.system('sbatch run_tmp.sh')


if __name__ == "__main__":
    batch_runner()
    
    