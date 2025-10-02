"""
This file is used to run various experiments in different tmux panes each.
"""

import os
import time

algo = "DistRL" # "DistRL" or "RecDistRL" or "SGPO"
device = "cuda"  # "cpu" or "cuda"
eval_episodes = 10
K = 1

# for envs in ['Pendulum-v1', 'MountainCarContinuous-v0','LunarLanderContinuous-v3]:
for envs in ['LunarLanderContinuous-v3']:
    for batch_size in [256]:
        for q_percentile in [0.7]:
            for top_k in [64]:
                for dynamic_beta in [False]:
                    for v_gamma in [2]:
                        for lr in [2e-4]:
                            for hidden_size in [256]:
                                for seed in [42]:

                                    extra = " --dynamic-beta" if dynamic_beta else ""

                                    name = f"MemBank_NegSamples_ReLUWithNorm{algo}-K={K}-v_gamma={v_gamma}-dyn_beta={dynamic_beta}-bs={batch_size}-lr={lr}-hs={hidden_size}-seed={seed}"

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
                                        f' --q-percentile {q_percentile}' + \
                                        f' --top-k {top_k}' + \
                                        f' --eval-episodes {eval_episodes}' + \
                                        f' --policy-training-start 10_000' + \
                                        f' --val-training-start 10_000' + \
                                        ' --log_to_wandb' + \
                                        f' {extra}' + \
                                        '" Enter'

                                    os.system(command=command)
                                    print(command)
                                    time.sleep(3)
