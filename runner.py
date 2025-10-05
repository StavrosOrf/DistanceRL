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

MUJOCO_ENVS = ['Ant-v5', 'HalfCheetah-v5', 'Hopper-v5', 'Humanoid-v5', 'InvertedDoublePendulum-v5',
               'InvertedPendulum-v5', 'Reacher-v5', 'Swimmer-v5', 'Walker2d-v5']
BOX2D_ENVS = ['LunarLanderContinuous-v3', 'MountainCarContinuous-v0', 'Pendulum-v1']
CLASSIC_ENVS = ['CartPole-v1', 'Acrobot-v1']

# PYTHON_ENV = "/home/sorfanouda/anaconda3/envs/dt/bin/python"
PYTHON_ENV = "/home/sorfanoudakis/.conda/envs/distrl/bin/python"

# for envs in ['Pendulum-v1', 'MountainCarContinuous-v0','LunarLanderContinuous-v3,"Hopper-v5"]:
for algo in ['DistRL']:  # 'RTGRecDistRL', 'StochRTGRecRL'
    for env in ['HalfCheetah-v5']:  # 'Pendulum-v1', 'MountainCarContinuous-v0','LunarLanderContinuous-v3',
        # if env in ['CartPole-v1'] and algo in ['RTGRecDistRL']:
        #     continue
        for batch_size in [256]:
            for K in [512]:
                for comp_samples in [4096]:
                    for rtg_enabled in [True]:                        
                        for noise_type in ["OU", "Sched"]: # "OU", "Sched", "Normal"
                            for expl_sigma in [0.1, 0.2, 0.3]:  # 0.1, 0.2, 0.3
                                for lr in [2e-4]:
                                    for hidden_size in [256]:
                                        for seed in [42]:
                                            
                                            extra = " --rtg-enabled" if rtg_enabled else ""
                                            
                                            name = f"{algo}-K={K}-bs={batch_size}-lr={lr}-seed={seed}"

                                            name = f'rtg_enabled={rtg_enabled}-expl_sigma={expl_sigma}-noise_type={noise_type}' + '-' + name
                                            
                                            command = 'tmux new-session -d \; send-keys "  ' + PYTHON_ENV + ' main.py' + \
                                                f' --env-id {env}' + \
                                                f' --algo {algo}' + \
                                                f' --device {device}' + \
                                                f' --batch-size {batch_size}' + \
                                                f' --K {K}' + \
                                                f' --v_gamma {v_gamma}' + \
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
