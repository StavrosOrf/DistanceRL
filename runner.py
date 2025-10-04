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

# for envs in ['Pendulum-v1', 'MountainCarContinuous-v0','LunarLanderContinuous-v3]:
for algo in ['RTGRecDistRL', 'StochRTGRecDistRL']:
    for env in ['Pendulum-v1', 'MountainCarContinuous-v0', 'Hopper-v5',"CartPole-v1"]:
        if env in ['CartPole-v1'] and algo in ['RTGRecDistRL']:
            continue

        for batch_size in [256]:
            for K in [64]:            
                for comp_samples in [4096]:
                    for v_gamma in [2]:
                        for lr in [2e-4]:
                            for hidden_size in [256]:
                                for seed in [42]:

                                    # name = f"Mem_newlr_LayerNorm_{algo}_top_k-{top_k}-K={K}-v_gamma={v_gamma}-dyn_beta={dynamic_beta}-bs={batch_size}-lr={lr}-hs={hidden_size}-seed={seed}"
                                    name = f"{algo}-K={K}-bs={batch_size}-lr={lr}-seed={seed}"

                                    command = 'tmux new-session -d \; send-keys "  /home/sorfanouda/anaconda3/envs/dt/bin/python main.py' + \
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
                                        ' --log_to_wandb' + \
                                        '" Enter'

                                    os.system(command=command)
                                    print(command)
                                    time.sleep(3)
