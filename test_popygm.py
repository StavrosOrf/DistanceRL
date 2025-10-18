"""Utility script to evaluate DistAgent on POPGym POMDP benchmarks.

NOTE: POPGym environments currently have discrete action spaces, but this script
uses the continuous-control DistAgent. For discrete action spaces, consider using
the DiscreteDistAgent from discreteDistRL module instead.

This script instantiates the continuous-distance agent (``DistAgent``) on a
collection of partially observable tasks provided by the `popgym` suite.
The script mirrors the CLI from ``main.py`` but focuses on benchmark evaluation
across multiple environments.

WARNING: This script may not work correctly with POPGym environments as they
use discrete action spaces. It's kept for reference/future continuous POPGym envs.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

try:
    import popgym  # type: ignore  # noqa: F401
except ImportError as err:  # pragma: no cover - configuration guard
    raise SystemExit(
        "popgym is required for POPGym evaluation. Install it with `pip install popgym`."
    ) from err

from dist_rl.config import DistRLConfig
from dist_rl.dist_agent import DistAgent
from dist_rl.utils import set_seed


def _discover_sota_envs() -> List[str]:
    """Attempt to recover the SOTA POPGym benchmark environment identifiers.

    The POPGym library exposes multiple entry points for its benchmark suites.
    Because the public API can differ across versions, we dynamically check a
    handful of likely modules/attributes and aggregate any identifiers that are
    discovered.  The function returns an alphabetically sorted, de-duplicated
    list of environment IDs.
    """

    candidates: List[str] = []
    
    # First, try to find environments from POPGym benchmark metadata
    module_attr_pairs = [
        ("popgym.benchmarks", "SOTA_POMDPS"),
        ("popgym.benchmarks", "SOTA_ENVS"),
        ("popgym.benchmarks.sota", "SOTA_POMDPS"),
        ("popgym.benchmarks.sota", "ENV_IDS"),
        ("popgym.benchmarks.sota", "BENCHMARK"),
    ]

    for module_name, attr_name in module_attr_pairs:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

        attribute = getattr(module, attr_name, None)
        if attribute is None:
            continue

        if callable(attribute):
            try:
                values = attribute()
            except TypeError:
                # Some modules expose callables that expect kwargs. Ignore them.
                continue
        else:
            values = attribute

        if isinstance(values, dict):
            iterable: Iterable[str] = values.keys()
        elif isinstance(values, (list, tuple, set)):
            iterable = values
        else:
            continue

        for item in iterable:
            if isinstance(item, str):
                candidates.append(item)

    # If no metadata found, query gymnasium registry for continuous control POPGym envs
    if not candidates:
        try:
            import gymnasium as gym
            # Get all registered POPGym environments that are continuous control
            # (CartPole and Pendulum variants which have continuous action spaces)
            all_envs = list(gym.envs.registry.keys())
            continuous_keywords = ['CartPole', 'Pendulum']
            for env_id in all_envs:
                if 'popgym' in env_id.lower():
                    if any(kw in env_id for kw in continuous_keywords):
                        candidates.append(env_id)
        except Exception:
            pass  # Fall through to hardcoded defaults

    # Fall back to a minimal set of well-known POPGym continuous POMDP tasks if
    # no metadata could be recovered from the installed version. These IDs come
    # from the POPGym continuous control benchmark environments.
    if not candidates:
        candidates = [
            "popgym-PositionOnlyCartPoleEasy-v0",
            "popgym-PositionOnlyCartPoleMedium-v0",
            "popgym-PositionOnlyPendulumEasy-v0",
        ]

    return sorted(set(candidates))


def _build_agent_kwargs(
    env_id: str,
    args: argparse.Namespace,
    config_overrides: Dict[str, float | int | None],
) -> Dict[str, float | int | str | None]:
    """Compose keyword arguments for ``DistAgent`` from CLI args and config."""

    env_key = env_id.replace("-v", "").replace("-", "").replace("/", "").lower()
    base_config = getattr(DistRLConfig, env_key, {}) or {}

    merged: Dict[str, float | int | str | None] = dict(base_config)
    merged["env_id"] = env_id
    merged["seed"] = args.seed
    merged["device"] = args.device
    merged["gamma"] = args.gamma
    merged["tau"] = args.tau
    merged.setdefault("total_steps", args.total_steps)
    merged.setdefault("buffer_size", args.buffer_size)
    merged.setdefault("batch_size", args.batch_size)
    merged.setdefault("hidden_size", args.hidden_size)
    merged.setdefault("K", args.K)
    merged.setdefault("lr", args.lr)
    merged.setdefault("expl_sigma", args.expl_sigma)
    merged.setdefault("target_entropy_scale", args.target_entropy_scale)
    merged.setdefault("updates_per_step", args.updates_per_step)
    merged.setdefault("kernel_adaptive_tau", int(args.kernel_adaptive_tau))
    merged.setdefault("rep_gamma_shape", args.rep_gamma_shape)
    merged.setdefault("rep_lam", args.rep_lam)
    merged.setdefault("rep_huber", args.rep_huber)
    merged.setdefault("normalize_obs", int(args.normalize_obs))
    merged.setdefault("warmup_steps", args.warmup_steps)
    merged.setdefault("eval_episodes", args.eval_episodes)
    merged.setdefault("eval_freq", args.eval_freq)
    merged["alpha"] = args.alpha
    merged["save_dir"] = str(Path(args.save_dir) / env_id.replace("/", "_"))

    # Convert binary flags to int form expected by DistAgent
    merged["kernel_adaptive_tau"] = int(bool(merged.get("kernel_adaptive_tau", 0)))
    merged["normalize_obs"] = int(bool(merged.get("normalize_obs", 0)))

    merged.update(config_overrides)

    return merged


def evaluate_suite(env_ids: Sequence[str], args: argparse.Namespace) -> Dict[str, float]:
    """Train and evaluate the distance agent on each POPGym environment."""

    import gymnasium as gym
    
    # Check if environments have continuous action spaces
    print("\n[Check] Verifying action space compatibility...")
    incompatible_envs = []
    for env_id in env_ids[:3]:  # Check first 3 as representative sample
        try:
            env = gym.make(env_id)
            if not isinstance(env.action_space, gym.spaces.Box):
                incompatible_envs.append((env_id, type(env.action_space).__name__))
            env.close()
        except Exception as e:
            print(f"[Warning] Could not check {env_id}: {e}")
    
    if incompatible_envs:
        print("\n" + "="*80)
        print("[ERROR] POPGym environments have DISCRETE action spaces!")
        print("="*80)
        print("\nThe DistAgent (dist_rl.dist_agent) is designed for CONTINUOUS action spaces.")
        print("However, the following POPGym environments have discrete actions:\n")
        for env_id, space_type in incompatible_envs:
            print(f"  - {env_id}: {space_type}")
        print("\n" + "="*80)
        print("SOLUTION: Use DiscreteDistAgent from discreteDistRL module instead:")
        print("  from discreteDistRL.dist_agent import DiscreteDistAgent")
        print("="*80 + "\n")
        raise SystemExit(
            "Cannot use continuous-control DistAgent with discrete-action POPGym environments. "
            "Please use DiscreteDistAgent or choose continuous-control environments."
        )

    results: Dict[str, float] = {}
    config_overrides: Dict[str, float | int | None] = {}

    if args.total_steps_override is not None:
        config_overrides["total_steps"] = args.total_steps_override
    if args.buffer_size_override is not None:
        config_overrides["buffer_size"] = args.buffer_size_override
    if args.batch_size_override is not None:
        config_overrides["batch_size"] = args.batch_size_override

    for idx, env_id in enumerate(env_ids):
        print("=" * 80)
        print(f"[Benchmark] ({idx + 1}/{len(env_ids)}) Evaluating DistAgent on {env_id}")
        set_seed(args.seed + idx)

        agent_kwargs = _build_agent_kwargs(env_id, args, config_overrides)
        agent = DistAgent(**agent_kwargs)
        agent.train()
        score = agent.evaluate()
        results[env_id] = score
        print(f"[Benchmark] Finished {env_id}: avg_return={score:.2f}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the DistanceRL agent on POPGym SOTA POMDP environments",
    )

    parser.add_argument("--env-ids", nargs="*", default=None,
                        help="Explicit POPGym environment IDs to benchmark. If omitted,"
                             " the script attempts to detect the SOTA suite from the"
                             " installed popgym package.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed for the experiments.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Compute device for DistAgent (cuda or cpu).")
    parser.add_argument("--save-dir", type=str, default="./popygm_runs",
                        help="Directory to store checkpoints per environment.")

    # DistAgent hyperparameters mirroring ``main.py`` defaults.
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--expl-sigma", type=float, default=0.2)
    parser.add_argument("--target-entropy-scale", type=float, default=1.0)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument(
        "--kernel-adaptive-tau",
        action="store_true",
        default=True,
        help="Enable adaptive kernel temperature scheduling (use --no-kernel-adaptive-tau to disable).",
    )
    parser.add_argument(
        "--no-kernel-adaptive-tau",
        action="store_false",
        dest="kernel_adaptive_tau",
        help="Disable adaptive kernel temperature scheduling.",
    )
    parser.add_argument(
        "--normalize-obs",
        action="store_true",
        default=True,
        help="Enable observation normalization in DistAgent (use --no-normalize-obs to disable).",
    )
    parser.add_argument(
        "--no-normalize-obs",
        action="store_false",
        dest="normalize_obs",
        help="Disable observation normalization in DistAgent.",
    )
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--rep-gamma-shape", type=float, default=2.0)
    parser.add_argument("--rep-lam", type=float, default=0.5)
    parser.add_argument("--rep-huber", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=None,
                        help="Fixed entropy temperature (omit for autotuning).")

    parser.add_argument("--total-steps-override", type=int, default=None,
                        help="Optional override applied to every environment's total_steps.")
    parser.add_argument("--buffer-size-override", type=int, default=None,
                        help="Optional override applied to every environment's buffer_size.")
    parser.add_argument("--batch-size-override", type=int, default=None,
                        help="Optional override applied to every environment's batch_size.")

    args = parser.parse_args()

    # Respect CPU fallback if CUDA is unavailable
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU execution.")
        args.device = "cpu"

    return args


def main() -> None:
    args = parse_args()

    env_ids = args.env_ids if args.env_ids else _discover_sota_envs()
    if not env_ids:
        raise SystemExit(
            "Could not determine any POPGym environments to evaluate. "
            "Pass explicit IDs via --env-ids."
        )

    print("POPGym evaluation will run on the following environments:")
    for env in env_ids:
        print(f"  - {env}")

    results = evaluate_suite(env_ids, args)

    print("=" * 80)
    print("POPGym benchmark summary:")
    for env_id, score in results.items():
        print(f"  {env_id}: avg_return={score:.2f}")


if __name__ == "__main__":
    main()
