"""
This file is used to run various experiments in different tmux panes each.
"""

import os
import time

algo = "RTGRecDistRL" # "DistRL" or "RecDistRL" or "SGPO"
device = "cuda"  # "cpu" or "cuda"
eval_episodes = 10
q_percentile = 0.7
dynamic_beta = False

# for envs in ['Pendulum-v1', 'MountainCarContinuous-v0','LunarLanderContinuous-v3]:
for envs in ['LunarLanderContinuous-v3']:
    for batch_size in [256]:
        for K in [16, 64]:
            for top_k in [64]:
                # for dynamic_beta in [False]:
                for beta in [5]:                    
                    for v_gamma in [2]:
                        for lr in [2e-4]:
                            for hidden_size in [256]:
                                for seed in [42]:

                                    extra = " --dynamic-beta" if dynamic_beta else ""

                                    # name = f"Mem_newlr_LayerNorm_{algo}_top_k-{top_k}-K={K}-v_gamma={v_gamma}-dyn_beta={dynamic_beta}-bs={batch_size}-lr={lr}-hs={hidden_size}-seed={seed}"                                    
                                    name = f"LinearPolicyObj_NORTG_{algo}-K={K}-top_k-{top_k}-dyn_beta={dynamic_beta}-bs={batch_size}-lr={lr}-hs={hidden_size}-seed={seed}"

                                    command = 'tmux new-session -d \; send-keys "  /home/sorfanouda/anaconda3/envs/dt/bin/python main.py' + \
                                        f' --env-id {envs}' + \
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
                                        f' --q-percentile {q_percentile}' + \
                                        f' --top-k {top_k}' + \
                                        f' --beta {beta}' + \
                                        f' --eval-episodes {eval_episodes}' + \
                                        f' --policy-training-start 10_000' + \
                                        f' --val-training-start 10_000' + \
                                        ' --log_to_wandb' + \
                                        f' {extra}' + \
                                        '" Enter'

                                    os.system(command=command)
                                    print(command)
                                    time.sleep(3)
