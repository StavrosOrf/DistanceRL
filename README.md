# NewRLAlgo
Experimenting with State-Action Distance RL

## PPO Implementation for Gymnasium Pendulum

This repository contains a custom implementation of Proximal Policy Optimization (PPO) for solving the Farama Gymnasium Pendulum-v1 task.

### Features

- Custom PPO algorithm with Actor-Critic architecture
- Generalized Advantage Estimation (GAE)
- Experience replay buffer
- Comprehensive test suite

### Installation

```bash
pip install -r requirements.txt
```

### Usage

#### Quick Training Test
```bash
python test_quick.py
```

#### Full Training
```bash
python train_pendulum.py
```

#### Run Tests
```bash
python -m pytest tests/test_ppo.py -v
```

### Project Structure

```
├── src/ppo/           # PPO implementation
│   ├── __init__.py
│   └── ppo.py        # Main PPO algorithm
├── tests/            # Test suite
│   ├── __init__.py
│   └── test_ppo.py   # Comprehensive tests
├── train_pendulum.py # Training script
├── test_quick.py     # Quick validation script
└── requirements.txt  # Dependencies
```

### PPO Implementation Details

The PPO implementation includes:
- **ActorCritic**: Neural network with shared layers and separate actor/critic heads
- **PPOBuffer**: Experience buffer with GAE computation
- **PPO**: Main algorithm with clipped surrogate objective

The algorithm uses:
- Clipping ratio: 0.2
- Discount factor (γ): 0.99
- GAE lambda: 0.95
- Learning rate: 3e-4
- Buffer size: 2048
- PPO epochs per update: 10
