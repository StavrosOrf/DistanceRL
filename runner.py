"""
This file is used to run various experiments in different tmux panes each.

TMux exit server fix:

tmux kill-server 2>/dev/null
pkill -x tmux 2>/dev/null
rm -rf "${TMUX_TMPDIR:-/tmp}/tmux-$(id -u)" 2>/dev/null
"""

import os
import time

device = "cuda:1"  # "cpu" or "cuda"
eval_episodes = 10
batch_size = 256
total_steps = 1_000_000
eval_freq = 5000
updates_per_step = 1
rep_loss_weight = 0.1
rep_gamma_shape = 0.5
target_entropy_scale = 1
comp_samples = 4096
noise_type = "OU"  # "OU", "SchedOU", "Normal"

expl_sigma = 0.3  # 0.1, 0.2,
K = 128

alpha_cql = 0.0

kernel_temp = 0.5
kernel_cand = 2048
kernel_state_k = 64
kernel_adaptive_tau = 1  # 1=True, 0=False
ot_eta = 1000

MUJOCO_ENVS = ['Ant-v5', 'HalfCheetah-v5', 'Hopper-v5', 'Humanoid-v5', 'InvertedDoublePendulum-v5',
               'InvertedPendulum-v5', 'Reacher-v5', 'Swimmer-v5', 'Walker2d-v5']
BOX2D_ENVS = ['LunarLanderContinuous-v3',
              'MountainCarContinuous-v0', 'Pendulum-v1']
CLASSIC_ENVS = ['CartPole-v1', 'Acrobot-v1']

# PYTHON_ENV = "/home/sorfanouda/anaconda3/envs/dt/bin/python"
PYTHON_ENV = "/home/sorfanoudakis/.conda/envs/distrl/bin/python"

# # for envs in ['Pendulum-v1', 'MountainCarContinuous-v0','LunarLanderContinuous-v3,"Hopper-v5"]:
# for algo in ['SACDistanceDiffusionAgent']:  # 'DistAgent'
#     for env in ['HalfCheetah-v5']:
#         for v_gamma in [0.5]:  # 0.99, 0.95, 1.0
#             rep_gamma_shape = v_gamma
#             for kernel_aux_weight in [100]:  # 0.0, 0.1
#                 for lr in [1e-3]:
#                     for hidden_size in [256]:
#                         for seed in [42]:

#                             # name = f"gamma-{v_gamma}-lr={lr}-seed={seed}"

#                             # +  '-' + name
#                             name = f'PullWeight1.5-{algo}-K{K}'

#                             command = 'tmux new-session -d \; send-keys "  ' + PYTHON_ENV + ' main.py' + \
#                                 f' --env-id {env}' + \
#                                 f' --algo {algo}' + \
#                                 f' --device {device}' + \
#                                 f' --batch-size {batch_size}' + \
#                                 f' --K {K}' + \
#                                 f' --v_gamma {v_gamma}' + \
#                                 f' --lr {lr}' + \
#                                 f' --hidden-size {hidden_size}' + \
#                                 f' --total-steps {total_steps}' + \
#                                 f' --buffer-size 1_000_000' + \
#                                 f' --seed {seed}' + \
#                                 f' --exp-prefix {name}' + \
#                                 f' --eval-episodes {eval_episodes}' + \
#                                 f' --eval-freq {eval_freq}' + \
#                                 f' --policy-training-start 10_000' + \
#                                 f' --val-training-start 10_000' + \
#                                 f' --comp-samples {comp_samples}' + \
#                                 f' --rep-gamma-shape {rep_gamma_shape}' + \
#                                 f' --rep-loss-weight {rep_loss_weight}' + \
#                                 f' --updates-per-step {updates_per_step}' + \
#                                 f' --target-entropy-scale {target_entropy_scale}' + \
#                                 f' --alpha-cql {alpha_cql}' + \
#                                 f' --kernel-aux-weight {kernel_aux_weight}' + \
#                                 f' --kernel-temp {kernel_temp}' + \
#                                 f' --kernel-cand {kernel_cand}' + \
#                                 f' --kernel-state-k {kernel_state_k}' + \
#                                 f' --kernel-adaptive-tau {kernel_adaptive_tau}' + \
#                                 f' --ot-eta {ot_eta}' + \
#                                 f' --noise-type {noise_type}' + \
#                                 ' --log_to_wandb' + \
#                                 f' --expl-sigma {expl_sigma}' + \
#                                 '" Enter'

#                             os.system(command=command)
#                             print(command)
#                             time.sleep(3)


# --- REDQ launcher with defaults and wandb logging ---
# for env in ['Walker2d-v5','Humanoid-v5','Ant-v5']:  # MUJOCO_ENVS:
#     for seed in [0]:
#         name = f'REDQ-{env}-seed{seed}'
#         command = 'tmux new-session -d \; send-keys "  ' + PYTHON_ENV + ' main.py' + \
#             f' --env-id {env}' + \
#             f' --algo REDQ' + \
#             f' --device {device}' + \
#             f' --seed {seed}' + \
#             f' --exp-prefix {name}' + \
#             ' --log_to_wandb' + \
#             '" Enter'

#         os.system(command=command)
#         print(command)
#         time.sleep(3)


total_steps = 10_000_000
use_one_hot_actions = 1
# -- DiscreteDistAgent launcher with defaults and wandb logging ---
for env in ['ALE/Pong-v5']:  # MUJOCO_ENVS:
    for seed in [100]:
        for K in [128]:
            for shared_encoder in [1]:
                name = f'DiscreteDistAgent-{env.strip("ALE/")}-seed{seed}'
                command = 'tmux new-session -d \; send-keys "  ' + PYTHON_ENV + ' main.py' + \
                    f' --env-id {env}' + \
                    f' --algo DiscreteDistAgent' + \
                    f' --device {device}' + \
                    f' --seed {seed}' + \
                    f' --K {K}' + \
                    f' --total-steps {total_steps}' + \
                    f' --batch-size {batch_size}' + \
                    f' --warmup-steps 50000' + \
                    f' --center-qhat 0' + \
                    f' --shared-encoder {shared_encoder}' + \
                    f' --use-one-hot-actions {use_one_hot_actions}' + \
                    f' --exp-prefix CHangedParamOpt2Fixed_{name}' + \
                    ' --log_to_wandb' + \
                    '" Enter'
                os.system(command=command)
                print(command)
                time.sleep(3)
