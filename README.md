# DistanceRL

DistanceRL is an experimental reinforcement-learning sandbox that explores a
state-action distance objective for continuous-control benchmarks. The project
contains both the custom DistanceRL agent and reference PPO/SAC/DDPG baselines
for comparison.

## Key features
- **Distance-based value learning** implemented with LSTM or Transformer
  sequence models for multi-step distance estimation.
- **Replay-buffer driven training loop** that separates policy and distance
  updates with configurable horizons.
- **Baseline classic RL algorithms** (PPO, SAC, TD3, DDPG) powered by the
  `classic_rl` package for head-to-head evaluations.
- **Weights & Biases logging** hooks for lightweight or full experiment tracking.

## Installation
Create a Python 3.10+ environment and install the dependencies:

```bash
pip install -r requirements.txt  # coming soon
pip install gymnasium[box2d] torch wandb pyyaml numpy
```

Box2D support is required for environments such as
`LunarLanderContinuous-v3` and `BipedalWalker-v3`.

## Usage
Launch training with the DistanceRL agent:

```bash
python main.py --env-id LunarLanderContinuous-v3 --total-steps 2000000 \
  --device cuda --log_to_wandb --exp_prefix dist_experiment
```

Switch to a classic baseline by setting `--algo` to `PPO`, `SAC`, `TD3`, or
`DDPG`. Hyperparameters are pulled automatically from
`classic_rl/hyperparams/<algo>.yaml`.

Useful arguments:
- `--K`: horizon length for distance estimates.
- `--policy-training-start` / `--val-training-start`: warm-up periods before
  policy or value updates.
- `--value-model-type`: choose between `LSTM` and `Transformer` backbones.
- `--dynamic-beta`: enable adaptive scaling of the distance loss.

## Project structure
```
classic_rl/         # Baseline algorithm wrappers and hyperparameters
models.py           # Actor, LSTM, and Transformer network definitions
loss.py             # Reward-aware cosine distance objective
utils.py            # Replay buffer and helper utilities
main.py             # CLI entry point for training experiments
distRL.py           # DistanceRL agent implementation
```

## Logging & evaluation
Pass `--log_to_wandb` to push metrics to Weights & Biases. Evaluation rollouts
are run every training epoch using the configured `--eval-episodes` count.

## Citation
If you build on this repository in academic work, please cite it as:

> Ongoing DistanceRL project, 2024. https://github.com/stavrosorf/DistanceRL
