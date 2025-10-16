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

def batch_runner():
    # ---------------- General configuration ----------------

    partition = 'gpu'  # gpu-a100 # gpu-a100-small # gpu, compute
    algo = ['DistAgent']
    group_name = "Ablationv2_"
    project_name = "DistRL" # DistRL_Rep

    # resource configuration
    device = 'cuda' if partition != 'compute' else 'cpu'
    job_hours = 6 if partition == 'gpu-a100-small' else 8

    cpu_cores = 1 if partition != 'compute' else 2
    memory_per_cpu = '5300' if partition != 'compute' else '3800'
    batch_arg = '#SBATCH --gpus-per-task=1' if partition != 'compute' else '\n'

    # training defaults (kept constant across ablations)
    v_gamma = 1.0
    hidden_size_default = 256  # !!!
    top_k_default = 32
    comp_samples_default = 4096
    noise_type_default = "OU"
    eval_episodes_default = 10
    policy_training_start_default = 10_000
    val_training_start_default = 10_000
    total_steps_default = 1_000_000
    buffer_size_default = total_steps_default  #!!!
    eval_freq_default = 5000
    updates_per_step_default = 1
    expl_sigma = 0.1

    rep_gamma_shape = 0.5

    alpha_cql_default = 0.0
    kernel_temp_default = 0.5
    kernel_cand_default = 2048
    kernel_state_k_default = 64
    kernel_aux_weight = 0.1

    SB3_ALGOS = ["ppo", "td3", "sac", "tqc"]
    MUJOCO_ENVS = ['HalfCheetah-v5', 'Ant-v5', 'Hopper-v5', 'Humanoid-v5', 'InvertedDoublePendulum-v5',
                'InvertedPendulum-v5', 'Reacher-v5', 'Swimmer-v5', 'Walker2d-v5'] #number of envs: 9
    BOX2D_ENVS = ['LunarLanderContinuous-v3',
                'MountainCarContinuous-v0', 'Pendulum-v1'] #number of envs: 3
    CLASSIC_ENVS = ['CartPole-v1', 'Acrobot-v1'] #number of envs: 2

    HARD_MUJOCO_ENVS = ['Humanoid-v5', 'Ant-v5'] #number of envs: 2
    
    F_ENVS = ['Walker2d-v5', 'Hopper-v5', 'Swimmer-v5', 'Reacher-v5', 'InvertedDoublePendulum-v5',
              'InvertedPendulum-v5']

    continuous_envs = MUJOCO_ENVS + BOX2D_ENVS

    # ---------------- Ablation grids ----------------


    lr_grid = [1e-3]  # [1e-3]

    seed_grid = [42, 32]

    K_grid = [32, 64, 256]
    expl_sigma_grid = [0.2]
    target_entropy_scale_grid = [1]
    batch_size_grid = [256] #!
    rep_gamma_shape_grid = [0.5, 2.0] #!
    kernel_adaptive_tau_grid = [1] #!
    normalize_obs_grid = [1] #!

    # if directory does not exist, create it
    if not os.path.exists('./slurm_logs'):
        os.makedirs('./slurm_logs')

    for env in F_ENVS:
        # for algo in ['tqc']:  # SB3_ALGOS:
        for algo in ['DistAgent']:
            for rep_gamma_shape in rep_gamma_shape_grid:
                for kernel_adaptive_tau in kernel_adaptive_tau_grid:
                    for normalize_obs in normalize_obs_grid:
                        for batch_size in batch_size_grid:
                            for K in K_grid:            
                                for lr in lr_grid:
                                    for seed in seed_grid:
                                        for target_entropy_scale in target_entropy_scale_grid:
                                            for expl_sigma in expl_sigma_grid:
                                                
                                                if algo in SB3_ALGOS:
                                                    job_hours = 5
                                                    if env in ['Humanoid-v5']:
                                                        job_hours = 7
                                                    
                                                if env in ['Humanoid-v5']:
                                                    buffer_size_default = 800_000
                                                    cpu_cores = 1 if partition == 'compute' else 2
                                                
                                                run_id = (f"{algo}_K{K}_bs{batch_size}_lr{lr}_tes{target_entropy_scale}_es{expl_sigma}")
                                                #add the other grid variables to the run_id
                                                run_id += (f"_rgs{rep_gamma_shape}_kat{kernel_adaptive_tau}_norm{normalize_obs}")
                                                run_id += f"_seed{seed}"
                                                run_name = f"{run_id}_{random.randint(0, 99999)}"
                                                print(f"Submitting {run_name}")

                                                python_command = 'python main.py' + \
                                                    f' --env-id {env}' + \
                                                    f' --algo {algo}' + \
                                                    f' --device {device}' + \
                                                    f' --batch-size {batch_size}' + \
                                                    f' --K {K}' + \
                                                    f' --lr {lr}' + \
                                                    f' --hidden-size {hidden_size_default}' + \
                                                    f' --total-steps {total_steps_default}' + \
                                                    f' --buffer-size {buffer_size_default}' + \
                                                    f' --seed {seed}' + \
                                                    f' --exp-prefix {run_name}' + \
                                                    f' --group-name {group_name}' + \
                                                    f' --eval-episodes {eval_episodes_default}' + \
                                                    f' --eval-freq {eval_freq_default}' + \
                                                    f' --noise-type {noise_type_default}' + \
                                                    f' --expl-sigma {expl_sigma}' + \
                                                    f' --normalize-obs {normalize_obs}' + \
                                                    f' --updates-per-step {updates_per_step_default}' + \
                                                    f' --rep-gamma-shape {rep_gamma_shape}' + \
                                                    f' --project_name {project_name}' + \
                                                    f' --target-entropy-scale {target_entropy_scale}' + \
                                                    f' --kernel-adaptive-tau {kernel_adaptive_tau}' + \
                                                    ' --log_to_wandb' + \
                                                    ' --lightweight_wandb'

                                                print(python_command)

                                                command = '''#!/bin/sh
#!/bin/bash
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
    
    