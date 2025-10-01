"""Classic reinforcement learning algorithms used alongside DistRL."""

from .ppo import PPOAgent
from .td3 import TD3Agent
from .sac import SACAgent

ALGO_REGISTRY = {
    "PPO": PPOAgent,
    "TD3": TD3Agent,
    "SAC": SACAgent,
}


def make_agent(algo_name: str, **kwargs):
    try:
        agent_cls = ALGO_REGISTRY[algo_name]
    except KeyError as exc:
        raise ValueError(f"Unknown classic RL algorithm: {algo_name}") from exc
    return agent_cls(**kwargs)
