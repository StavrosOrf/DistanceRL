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
from dist_rl_fix.config import DistRLConfig


def batch_runner():
    # ---------------- General configuration ----------------

    partition = 'gpu'  # gpu-a100 # gpu-a100-small # gpu, compute
    algo = ['DistAgent']
    group_name = "Ablation_"
    project_name = "DistRL_Rep"  # DistRL_Rep

    # resource configuration
    device = 'cuda' if partition != 'compute' else 'cpu'
    job_hours = 6 if partition == 'gpu-a100-small' else 15

    cpu_cores = 1 if partition != 'compute' else 2
    memory_per_cpu = '5300' if partition != 'compute' else '3800'
    batch_arg = '#SBATCH --gpus-per-task=1' if partition != 'compute' else '\n'

    SB3_ALGOS = ["ppo", "td3", "sac", "tqc"]
    MUJOCO_ENVS = ['HalfCheetah-v5', 'Ant-v5', 'Hopper-v5', 'Humanoid-v5', 'InvertedDoublePendulum-v5',
                   'InvertedPendulum-v5', 'Reacher-v5', 'Swimmer-v5', 'Walker2d-v5']  # number of envs: 9
    #ENVS to try 
    F_ENVS = ['Walker2d-v5', 'Hopper-v5', 'Swimmer-v5', 'Reacher-v5', 'InvertedDoublePendulum-v5',
              'InvertedPendulum-v5', 'Pusher-v5']
    BOX2D_ENVS = ['LunarLanderContinuous-v3',
                  'MountainCarContinuous-v0', 'Pendulum-v1']  # number of envs: 3
    CLASSIC_ENVS = ['CartPole-v1', 'Acrobot-v1']  # number of envs: 2

    HARD_MUJOCO_ENVS = ['Humanoid-v5', 'Ant-v5',
                        'HalfCheetah-v5'] 
    


    continuous_envs = MUJOCO_ENVS + BOX2D_ENVS

    seed_grid = [42, 32, 22, 12, 2]
    training_steps = 1_000_000

    # if directory does not exist, create it
    if not os.path.exists('./slurm_logs'):
        os.makedirs('./slurm_logs')

    for env in HARD_MUJOCO_ENVS:
        for algo in ['DistAgent']:
            for seed in seed_grid:

                if algo == 'DistAgent':
                    g = getattr(DistRLConfig(), env.split(
                        '-')[0].lower(), None)

                if algo in SB3_ALGOS:
                    job_hours = 5
                    if env in ['Humanoid-v5']:
                        job_hours = 7

                run_id = (
                    f"{algo}_K{g['K']}_bs{g['batch_size']}_lr{g['lr']}_tes{g['target_entropy_scale']}_es{g['expl_sigma']}")
                # add the other grid variables to the run_id
                run_id += (
                    f"_rgs{g['rep_gamma_shape']}_kat{g['kernel_adaptive_tau']}_norm{g['normalize_obs']}")
                run_id += f"_seed{seed}"
                run_name = f"{run_id}_{random.randint(0, 99999)}"
                print(f"Submitting {run_name}")

                python_command = 'python main.py' + \
                    f' --env-id {env}' + \
                    f' --algo {algo}' + \
                    f' --device {device}' + \
                    f' --batch-size {g["batch_size"]}' + \
                    f' --K {g["K"]}' + \
                    f' --lr {g["lr"]}' + \
                    f' --hidden-size {g["hidden_size"]}' + \
                    f' --total-steps {training_steps}' + \
                    f' --buffer-size {g["buffer_size"]}' + \
                    f' --seed {seed}' + \
                    f' --exp-prefix {run_name}' + \
                    f' --group-name {group_name}' + \
                    f' --eval-episodes {g["eval_episodes"]}' + \
                    f' --eval-freq {g["eval_freq"]}' + \
                    f' --expl-sigma {g["expl_sigma"]}' + \
                    f' --normalize-obs {g["normalize_obs"]}' + \
                    f' --updates-per-step {g["updates_per_step"]}' + \
                    f' --rep-gamma-shape {g["rep_gamma_shape"]}' + \
                    f' --project_name {project_name}' + \
                    f' --target-entropy-scale {g["target_entropy_scale"]}' + \
                    f' --kernel-adaptive-tau {g["kernel_adaptive_tau"]}' + \
                    ' --log_to_wandb' + \
                    ' --lightweight_wandb'

                print(python_command + '\n')

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
