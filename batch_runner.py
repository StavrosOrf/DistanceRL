'''
This script is used to run the batch of training sciprts for every algorithms evaluated
#old command# srun --mpi=pmix --job-name=interactive-gpu --partition=gpu --gres=gpu:1 --qos=normal --time=01:00:00 --mem-per-cpu=4096 --pty /bin/bash -il
srun --mpi=pmix --job-name=interactive-gpu --partition=gpu --gpus-per-task=1 --qos=normal --time=01:00:00 --mem-per-cpu=5300 --cpus-per-task=1 --ntasks=1 --pty /bin/bash -il
srun --mpi=pmix --job-name=interactive-gpu --partition=gpu-a100-small --gpus-per-task=1 --qos=normal --time=01:00:00 --mem-per-cpu=7000 --cpus-per-task=1 --ntasks=1 --pty /bin/bash -il
srun --mpi=pmix --job-name=interactive-gpu --partition=gpu-a100 --gpus-per-task=1 --qos=normal --time=01:00:00 --mem-per-cpu=5500 --cpus-per-task=1 --ntasks=1 --pty /bin/bash -il
srun --mpi=pmix --job-name=interactive --partition=compute --cpus-per-task=2 --qos=normal --time=01:00:00 --mem-per-cpu=3800 --ntasks=1 --pty /bin/bash -il
'''
import os
import random

# ---------------- General configuration ----------------

partition = 'gpu-a100'  # gpu-a100 # gpu-a100-small # gpu, compute
algo = 'sacDistRL'
group_name = "SACDistRL_ablation_"

# resource configuration
device = 'cuda' if partition != 'compute' else 'cpu'
job_hours = 4 if partition == 'gpu-a100-small' else 6
 
cpu_cores = 1 if partition != 'compute' else 2
memory_per_cpu = '5300' if partition != 'compute' else '3800'
batch_arg = '#SBATCH --gpus-per-task=1' if partition != 'compute' else '\n'

# training defaults (kept constant across ablations)
v_gamma = 1.0
hidden_size_default = 512#!!!
buffer_size_default = 1_000_000
top_k_default = 32
comp_samples_default = 4096
noise_type_default = "OU"
eval_episodes_default = 10
policy_training_start_default = 10_000
val_training_start_default = 10_000
total_steps_default = 1_500_000
eval_freq_default = 5000
updates_per_step_default = 1

alpha_cql_default = 0.0
kernel_temp_default = 0.5
kernel_cand_default = 2048
kernel_state_k_default = 64
kernel_adaptive_tau_default = 1

# ---------------- Ablation grids ----------------
env_grid = ['Humanoid-v5'] #
batch_size_grid = [256]
K_grid = [256, 512]
expl_sigma_grid = [0.1]
lr_grid = [3e-4]# [1e-3]
seed_grid = [42]
rep_gamma_shape_grid = [1.0]
rep_loss_weight_grid = [0.0, 0.1]
target_entropy_scale_grid = [0.7, 0.9]
kernel_aux_weight_grid = [0.0, 0.1]

# if directory does not exist, create it
if not os.path.exists('./slurm_logs'):
    os.makedirs('./slurm_logs')
# for envs in ['Pendulum-v1', 'MountainCarContinuous-v0','LunarLanderContinuous-v3]:
for env in env_grid:
    for batch_size in batch_size_grid:
        for K in K_grid:
            for expl_sigma in expl_sigma_grid:
                for lr in lr_grid:
                    for seed in seed_grid:
                        for rep_loss_weight in rep_loss_weight_grid:
                            for target_entropy_scale in target_entropy_scale_grid:
                                for rep_gamma_shape in rep_gamma_shape_grid:
                                    for kernel_aux_weight in kernel_aux_weight_grid:

                                        env_tag = env.replace('-', '')
                                        run_id = (
                                            f"bs{batch_size}_k{K}_sig{expl_sigma}"
                                            f"_lr{lr}_sd{seed}_rg{rep_gamma_shape}"
                                            f"_rw{rep_loss_weight}_te{target_entropy_scale}"
                                            f"_ka{kernel_aux_weight}"
                                        )
                                        run_name = f"{run_id}"#_{random.randint(0, 99999)}"
                                        print(f"Submitting {run_name}")

                                        python_command = 'python main.py' + \
                                            f' --env-id {env}' + \
                                            f' --algo {algo}' + \
                                            f' --device {device}' + \
                                            f' --batch-size {batch_size}' + \
                                            f' --K {K}' + \
                                            f' --v_gamma {v_gamma}' + \
                                            f' --top-k {top_k_default}' + \
                                            f' --lr {lr}' + \
                                            f' --hidden-size {hidden_size_default}' + \
                                            f' --total-steps {total_steps_default}' + \
                                            f' --buffer-size {buffer_size_default}' + \
                                            f' --seed {seed}' + \
                                            f' --exp-prefix {run_name}' + \
                                            f' --group-name {group_name}' + \
                                            f' --eval-episodes {eval_episodes_default}' + \
                                            f' --eval-freq {eval_freq_default}' + \
                                            f' --policy-training-start {policy_training_start_default}' + \
                                            f' --val-training-start {val_training_start_default}' + \
                                            f' --comp-samples {comp_samples_default}' + \
                                            f' --noise-type {noise_type_default}' + \
                                            f' --expl-sigma {expl_sigma}' + \
                                            f' --updates-per-step {updates_per_step_default}' + \
                                            f' --rep-gamma-shape {rep_gamma_shape}' + \
                                            f' --rep-loss-weight {rep_loss_weight}' + \
                                            f' --target-entropy-scale {target_entropy_scale}' + \
                                            f' --alpha-cql {alpha_cql_default}' + \
                                            f' --kernel-aux-weight {kernel_aux_weight}' + \
                                            f' --kernel-temp {kernel_temp_default}' + \
                                            f' --kernel-cand {kernel_cand_default}' + \
                                            f' --kernel-state-k {kernel_state_k_default}' + \
                                            f' --kernel-adaptive-tau {kernel_adaptive_tau_default}' + \
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

