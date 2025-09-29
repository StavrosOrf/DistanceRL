"""
This file is used to run various experiments in different tmux panes each.
"""

import os
import time

algo = "DistRL"
device = "cuda"  # "cpu" or "cuda"

# for envs in ['Pendulum-v1', 'MountainCarContinuous-v0']:
for envs in ['MountainCarContinuous-v0']:
    for batch_size in [128]:
        for K in [2, 32]:  # 512
        # for K in [2, 6, 15, 32, 64]:  # 512
            for v_gamma in [0.8, 1, 1.2]:
                for lr in [3e-4]:
                    for hidden_size in [64]:
                        for seed in [42]:
                            name = f"{algo}-K{K}-v_gamma{v_gamma}-bs{batch_size}-lr{lr}-hs{hidden_size}-seed{seed}"

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
                                f' --exp_prefix {name}' + \
                                ' --log_to_wandb' + \
                                '" Enter'

                            os.system(command=command)
                            print(command)
                            time.sleep(3)
