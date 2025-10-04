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

# for envs in ['Pendulum-v1', 'MountainCarContinuous-v0','LunarLanderContinuous-v3]:
for algo in ['RTGRecDistRL']:
    for env in ['MountainCarContinuous-v0']:
        if env in ['CartPole-v1'] and algo in ['RTGRecDistRL']:
            continue
        for batch_size in [256]:
            for K in [64, 128, 256, 512]:
                for comp_samples in [4096]:
                    for rtg_enabled in [False]:
                        for noise_type in ["Sched"]: # "OU", "Sched"
                            for expl_sigma in [0.5]:
                                for lr in [2e-4]:
                                    for hidden_size in [256]:
                                        for seed in [42]:
                                            
                                            extra = " --rtg-enabled" if rtg_enabled else ""
                                            
                                            name = f"{algo}-K={K}-bs={batch_size}-lr={lr}-seed={seed}"

                                            name = f'rtg_enabled={rtg_enabled}-expl_sigma={expl_sigma}-noise_type={noise_type}' + '-' + name

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
                                                f' --noise-type {noise_type}' + \
                                                ' --log_to_wandb' + \
                                                f' --expl-sigma {expl_sigma}' + \
                                                extra + \
                                                '" Enter'

                                            os.system(command=command)
                                            print(command)
                                            time.sleep(3)
