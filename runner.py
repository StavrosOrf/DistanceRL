"""
This file is used to run various experiments in different tmux panes each.
"""

import os
import time

algo = "SABLE-PI" # "DistRL" or "RecDistRL" or "SGPO"
device = "cuda"  # "cpu" or "cuda"
eval_episodes = 10

# for envs in ['Pendulum-v1', 'MountainCarContinuous-v0','LunarLanderContinuous-v3]:
for envs in ['LunarLanderContinuous-v3']:
    for batch_size in [128]:
        for K in [1]:
            for dynamic_beta in [True]:
                for v_gamma in [0.95]:
                    for lr in [3e-4]:
                        for hidden_size in [128]:
                            for seed in [42]:

                                extra = " --dynamic-beta" if dynamic_beta else ""

                                name = f"{algo}-K={K}-v_gamma={v_gamma}-dyn_beta={dynamic_beta}-bs={batch_size}-lr={lr}-hs={hidden_size}-seed={seed}"

                                command = 'tmux new-session -d \; send-keys "  /home/sorfanouda/anaconda3/envs/dt/bin/python main.py' + \
                                    f' --env-id {envs}' + \
                                    f' --algo {algo}' + \
                                    f' --device {device}' + \
                                    f' --batch-size {batch_size}' + \
                                    f' --K {K}' + \
                                    f' --v_gamma {v_gamma}' + \
                                    f' --lr {lr}' + \
                                    f' --hidden-size {hidden_size}' + \
                                    f' --seed {seed}' + \
                                    f' --exp-prefix {name}' + \
                                    f' --eval-episodes {eval_episodes}' + \
                                    ' --log_to_wandb' + \
                                    f' {extra}' + \
                                    '" Enter'

                                os.system(command=command)
                                print(command)
                                time.sleep(3)
