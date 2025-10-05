'''
This script is used to run the batch of training sciprts for every algorithms evaluated
#old command# srun --mpi=pmix --job-name=interactive-gpu --partition=gpu --gres=gpu:1 --qos=normal --time=01:00:00 --mem-per-cpu=4096 --pty /bin/bash -il
srun --mpi=pmix --job-name=interactive-gpu --partition=gpu --gpus-per-task=1 --qos=normal --time=01:00:00 --mem-per-cpu=5300 --cpus-per-task=2 --ntasks=1 --pty /bin/bash -il
srun --mpi=pmix --job-name=interactive-gpu --partition=gpu-a100-small --gpus-per-task=1 --qos=normal --time=01:00:00 --mem-per-cpu=7000 --cpus-per-task=1 --ntasks=1 --pty /bin/bash -il
srun --mpi=pmix --job-name=interactive-gpu --partition=gpu-a100 --gpus-per-task=1 --qos=normal --time=01:00:00 --mem-per-cpu=5500 --cpus-per-task=1 --ntasks=1 --pty /bin/bash -il
srun --mpi=pmix --job-name=interactive --partition=compute --cpus-per-task=2 --qos=normal --time=01:00:00 --mem-per-cpu=3800 --ntasks=1 --pty /bin/bash -il
'''
import os
import random

partition = 'gpu'  # gpu-a100 # gpu-a100-small # gpu, compute
algo = 'DistRL'
group_name = "param_abl_"
eval_episodes = 10
v_gamma = 2.0

# if directory does not exist, create it
if not os.path.exists('./slurm_logs'):
    os.makedirs('./slurm_logs')

# for envs in ['Pendulum-v1', 'MountainCarContinuous-v0','LunarLanderContinuous-v3]:
for env in ['HalfCheetah-v5']:
        for batch_size in [256]:
            for K in [256, 1000]:
                for top_k in [32, 64, 128]:  # 2, 8, 32, 64
                    for comp_samples in [2048, 4096, 8192]:
                        for rtg_enabled in [False]:
                            for noise_type in ["OU"]:  # "OU", "Sched", "Normal"
                                for expl_sigma in [0.2]:  # 0.1, 0.2, 0.3                                
                                    for lr in [2e-4]:
                                        for hidden_size in [256]:
                                            for seed in [42]:

                                                extra = " --rtg-enabled" if rtg_enabled else ""

                                                run_name = f"{algo}-K={K}-top_k={top_k}-noise_type={noise_type}-s={expl_sigma}-lr={lr}-hs={hidden_size}-seed={seed}"

                                                run_name += str(random.randint(0, 100000))
                                                print(f"Running {run_name}")
                                                time = '23'  # in hours

                                                if partition == 'compute':
                                                    cpu_cores = 2
                                                    memory = '3800'  # memory per cpu core
                                                    batch_arg = '\n'

                                                else:
                                                    cpu_cores = 1
                                                    memory = '5300'  # memory per cpu core
                                                    batch_arg = '#SBATCH --gpus-per-task=1'

                                                device = 'cpu' if partition == 'compute' else 'cuda'  # cpu or cuda

                                                python_command = 'python main.py' + \
                                                    f' --env-id {env}' + \
                                                    f' --algo {algo}' + \
                                                    f' --device {device}' + \
                                                    f' --batch-size {batch_size}' + \
                                                    f' --K {K}' + \
                                                    f' --v_gamma {v_gamma}' + \
                                                    f' --top-k {top_k}' + \
                                                    f' --lr {lr}' + \
                                                    f' --hidden-size {hidden_size}' + \
                                                    f' --buffer-size 500_000' + \
                                                    f' --seed {seed}' + \
                                                    f' --exp-prefix {run_name}' + \
                                                    f' --eval-episodes {eval_episodes}' + \
                                                    f' --policy-training-start 10_000' + \
                                                    f' --val-training-start 10_000' + \
                                                    f' --comp-samples {comp_samples}' + \
                                                    f' --group-name {group_name}' + \
                                                    f' --noise-type {noise_type}' + \
                                                    ' --log_to_wandb' + \
                                                    f' --expl-sigma {expl_sigma}' + \
                                                    f' {extra}'

                                                print(python_command)

                                                command = '''#!/bin/sh
#!/bin/bash
#SBATCH --job-name="dist_rl"
''' + \
                                    f'#SBATCH --partition={partition}\n' + \
                                    f'#SBATCH --time={time}:00:00\n' + \
                                    f'{batch_arg}' + \
                                    '''
#SBATCH --ntasks=1
''' + \
                                    f'#SBATCH --cpus-per-task={cpu_cores}' + \
                                    '''
''' + \
                                    f'#SBATCH --mem-per-cpu={memory}' + \
                                    '''
# SBATCH --account=research-eemcs-ese

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
previous =$(/usr/bin/nvidia-smi - -query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' - -format='csv' | /usr/bin/tail - n '+2')

''' + 'srun ' + python_command + \
                                    '''

/usr/bin/nvidia-smi - -query-accounted-apps = 'gpu_utilization,mem_utilization,max_memory_usage,time' - -format = 'csv' | /usr/bin/grep - v - F "$previous"

conda deactivate
'''
                                                with open(f'run_tmp.sh', 'w') as f:
                                                    f.write(command)

                                                with open(f'./slurm_logs/{run_name}.sh', 'w') as f:
                                                    f.write(command)

                                                os.system('sbatch run_tmp.sh')
