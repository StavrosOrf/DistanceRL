'''
This script is used to run the batch of training sciprts for every algorithms evaluated
#old command# srun --mpi=pmix --job-name=interactive-gpu --partition=gpu --gres=gpu:1 --qos=normal --time=01:00:00 --mem-per-cpu=4096 --pty /bin/bash -il
srun --mpi=pmix --job-name=interactive-gpu --partition=gpu --gpus-per-task=1 --qos=normal --time=01:00:00 --mem-per-cpu=5300 --cpus-per-task=3 --ntasks=1 --pty /bin/bash -il
srun --mpi=pmix --job-name=interactive-gpu --partition=gpu-a100-small --gpus-per-task=1 --qos=normal --time=01:00:00 --mem-per-cpu=7000 --cpus-per-task=1 --ntasks=1 --pty /bin/bash -il
srun --mpi=pmix --job-name=interactive-gpu --partition=gpu-a100 --gpus-per-task=1 --qos=normal --time=01:00:00 --mem-per-cpu=7000 --cpus-per-task=1 --ntasks=1 --pty /bin/bash -il
srun --mpi=pmix --job-name=interactive --partition=compute --cpus-per-task=2 --qos=normal --time=01:00:00 --mem-per-cpu=3800 --ntasks=1 --pty /bin/bash -il
'''
import os
import random

seeds = [10]

# gpu = 'gpu' #gpu-a100 # gpu-a100-small # gpu
partition= 'compute'
algo = 'DistRL'

# if directory does not exist, create it
if not os.path.exists('./slurm_logs'):
    os.makedirs('./slurm_logs')

# for envs in ['Pendulum-v1', 'MountainCarContinuous-v0','LunarLanderContinuous-v3]:
for envs in ['LunarLanderContinuous-v3']:
    for batch_size in [256]:
        for K in [64]:  # 512
        # for K in [2, 6, 15, 32, 64]:  # 512
            for v_gamma in [1.1]:
                for lr in [3e-4]:
                    for hidden_size in [64]:
                        for seed in [42]:
                            run_name = f"{algo}-K{K}-v_gamma{v_gamma}-bs{batch_size}-lr{lr}-hs{hidden_size}-seed{seed}"

                            run_name += str(random.randint(0, 100000))
                            print(f"Running {run_name}")
                            time = '24'  # in hours
                            cpu_cores = 2   
                            memory = '3800'  # memory per cpu core
                            
                            device = 'cpu'  if partition == 'compute' else 'cuda'  # cpu or cuda
                                
                            command = '''#!/bin/sh
#!/bin/bash
#SBATCH --job-name="pi_rl"
''' + \
                    f'#SBATCH --partition={partition}\n' + \
                    f'#SBATCH --time={time}:00:00' + \
                    '''
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
''' + \
                    f'#SBATCH --cpus-per-task={cpu_cores}' + \
                    '''
''' + \
                    f'#SBATCH --mem-per-cpu={memory}' + \
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

''' + 'srun python train.py' + \
                    ' --env-id ' + envs + \
                    ' --algo ' + algo + \
                    ' --device ' + device + \
                    f' --batch-size {batch_size}' + \
                    f' --K {K}' + \
                    f' --v_gamma {v_gamma}' + \
                    f' --lr {lr}' + \
                    f' --hidden-size {hidden_size}' + \
                    f' --seed {seed}' + \
                    f' --exp_prefix {run_name}' + \
                    ' --log_to_wandb' + \
                    '' + \
                    '''
            
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

conda deactivate
'''

                            with open(f'run_tmp.sh', 'w') as f:
                                f.write(command)

                            with open(f'./slurm_logs/{run_name}.sh', 'w') as f:
                                f.write(command)

                            os.system('sbatch run_tmp.sh')
