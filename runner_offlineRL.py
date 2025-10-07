"""
This file is used to run various experiments in different tmux panes each.

TMux exit server fix:

tmux kill-server 2>/dev/null
pkill -x tmux 2>/dev/null
rm -rf "${TMUX_TMPDIR:-/tmp}/tmux-$(id -u)" 2>/dev/null
"""

import os
import time

device = "cuda"  # "cpu" or "cuda"
eval_episodes = 10
v_gamma = 2.0
batch_size = 256

MUJOCO_ENVS = ['Ant-v5', 'HalfCheetah-v5', 'Hopper-v5', 'Humanoid-v5', 'InvertedDoublePendulum-v5',
               'InvertedPendulum-v5', 'Reacher-v5', 'Swimmer-v5', 'Walker2d-v5']
BOX2D_ENVS = ['LunarLanderContinuous-v3',
              'MountainCarContinuous-v0', 'Pendulum-v1']
CLASSIC_ENVS = ['CartPole-v1', 'Acrobot-v1']

PYTHON_ENV = "/home/sorfanouda/anaconda3/envs/dt/bin/python"
# PYTHON_ENV = "/home/sorfanoudakis/.conda/envs/distrl/bin/python"

# for envs in ['Pendulum-v1', 'MountainCarContinuous-v0','LunarLanderContinuous-v3,"Hopper-v5"]:
for algo in ['DistRL']:  # 'RTGRecDistRL', 'StochRTGRecRL'
    for dataset in ['mujoco/halfcheetah/expert-v0', 'mujoco/halfcheetah/medium-v0', 'mujoco/halfcheetah/simple-v0']:
        env_mapping = {
            'halfcheetah': 'HalfCheetah-v5',
            'hopper': 'Hopper-v5',
            'walker2d': 'Walker2d-v5',
            'ant': 'Ant-v5',
            'humanoid': 'Humanoid-v5',
        }
        dataset_env_name = dataset.split('/')[1]
        # Get last part after / and before -
        dataset_name = dataset.split('/')[-1].split('-')[0]
        env = env_mapping[dataset_env_name.lower()]

        for max_dataset_episodes in [500]:  # out of 1000
            for K in [128]:
                for comp_samples in [4096]:
                    for rtg_enabled in [False]:
                        for noise_type in ["OU"]:  # "OU", "SchedOU", "Normal"
                            for expl_sigma in [0.1]:  # 0.1, 0.2, 0.3
                                for top_k in [64]:  # 2, 8, 32, 64
                                    for lr in [1e-3]:
                                        for hidden_size in [256]:
                                            for seed in [42]:

                                                extra = " --rtg-enabled" if rtg_enabled else ""

                                                name = f"-lr={lr}-K={K}-seed={seed}"

                                                name = f'{algo}-{dataset_name}_d={max_dataset_episodes}_s={expl_sigma}-noise-type={noise_type}_cmp={comp_samples}_topk={top_k}' + '-' + name

                                                command = 'tmux new-session -d \; send-keys "  ' + PYTHON_ENV + ' main.py' + \
                                                    f' --env-id {env}' + \
                                                    f' --dataset {dataset}' + \
                                                    f' --max-dataset-episodes {max_dataset_episodes}' + \
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
                                                    f' --exp-prefix {name}' + \
                                                    f' --eval-episodes {eval_episodes}' + \
                                                    f' --policy-training-start 10_000' + \
                                                    f' --val-training-start 10_000' + \
                                                    f' --comp-samples {comp_samples}' + \
                                                    f' --noise-type {noise_type}' + \
                                                    ' --log_to_wandb' + \
                                                    f' --expl-sigma {expl_sigma}' + \
                                                    extra + \
                                                    '" Enter'

                                                os.system(command=command)
                                                print(command)
                                                time.sleep(3)
