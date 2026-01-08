import wandb
import pandas as pd
import numpy as np
import tqdm as tqdm
import json

# Initialize API
import os
SB3_ALGOS = ["ppo", "td3", "sac", "tqc"]
ABLATION_ALGOS = [
    "DistAblationA1", "DistAblationA2", "DistAblationA3", "DistAblationA4", "DistAblationA5",
    "DistAblationB1", "DistAblationB2", "DistAblationB3", "DistAblationB4", "DistAblationB5",
    "DistAblationB6", "DistAblationB7", "DistAblationB8", "DistAblationB9NoAdaptiveTau",
]

os.environ['WANDB_HTTP_TIMEOUT'] = '300'

# API timeout used for all W&B calls (seconds)
API_TIMEOUT_SECS = 120

MUJOCO_ENVS = ['HalfCheetah-v5', 'Ant-v5', 'Hopper-v5', 'Humanoid-v5', 'InvertedDoublePendulum-v5',
                   'InvertedPendulum-v5', 'Reacher-v5', 'Swimmer-v5', 'Walker2d-v5']  # number of envs: 9

steps_per_env = {
    'HalfCheetah-v5': 5_000_000,
    'Ant-v5': 5_000_000,
    'Hopper-v5': 3_000_000,
    'Humanoid-v5': 10_000_000,
    'InvertedDoublePendulum-v5': 500_000,
    'InvertedPendulum-v5': 500_000,
    'Reacher-v5': 500_000,
    'Swimmer-v5': 1_000_000,
    'Walker2d-v5': 5_000_000,
}

ENV_TO_RUN = [env for env in MUJOCO_ENVS if steps_per_env[env] >= 1_000_000]


def _fetch_history_with_retry(run, keys, pandas=True, max_attempts=3, delay_secs=10):
    """Fetch run history with a few retries to ride out transient 5xx/timeouts."""
    attempts = 0
    while attempts < max_attempts:
        try:
            return run.history(keys=keys, pandas=pandas)
        except wandb.errors.CommError as err:
            attempts += 1
            print(f"    History fetch attempt {attempts} failed for {run.name}: {err}")
            if attempts >= max_attempts:
                break
            # brief pause before retrying
            import time
            time.sleep(delay_secs)
        except Exception as err:
            attempts += 1
            print(f"    History fetch attempt {attempts} failed for {run.name}: {err}")
            if attempts >= max_attempts:
                break
            import time
            time.sleep(delay_secs)
    return None


def data_fetcher():
    api = wandb.Api(timeout=API_TIMEOUT_SECS)

    entity_name = "stavrosorf"

    # Display the filtered runs with group names

    run_results = pd.DataFrame()
    result_summary = []
    # use tqdm to display a progress bar
    for project_name in ["DistRL_RepK_Ablations"]:
        # Fetch runs from the specified project
        runs = api.runs(f"{entity_name}/{project_name}")
        print(f"Total runs fetched: {len(runs)}")
        for i, run in tqdm.tqdm(enumerate(runs), total=len(runs)):
                        
            #only parse form gorup_name RepK_Ablation_FixedHumanoid-v5 and RepK_Ablation_FixedHalfCheetah-v5
            if not (run.group and (run.group.startswith("RepK_Ablation_Fixed"))):
                # print(f'Skipping run {run.name} with group {run.group}')
                continue

            # Handle config - W&B wraps values in {"value": ...} format
            config_raw = run.config
            config = {}
            config_raw = json.loads(config_raw)

            # Extract actual values from W&B format: {"key": {"value": actual_value}}
            if isinstance(config_raw, dict):
                for key, val in config_raw.items():
                    if isinstance(val, dict) and 'value' in val:
                        config[key] = val['value']
                    else:
                        config[key] = val

            algo = config['algo']
            env_id = config['env_id']
            seed = config['seed']
            

            if project_name == "DistRL_Rep" and algo == 'DistAgent' and env_id in ENV_TO_RUN:
                continue


            print(
                f"Run {i+1}/{len(runs)}: - Algo: {algo} - Env: {env_id} - Seed: {seed}")

            # data = extract_eval_rewards(run, algo, env_id, seed)
            data = extract_eval_rewards_Faster(run, algo, env_id, seed)

            if data is None:
                continue

            run_results = pd.concat([run_results, data], ignore_index=True)

            history = _fetch_history_with_retry(run, keys=['_runtime'])
            if history is None:
                print(f"Run {run.id} skipped due to history fetch failures")
                continue
            if '_runtime' not in history:
                print(f"Run {run.id} has no _runtime key")
                continue

            best_reward = data['eval_reward'].max()
            train_steps = int(data['step'].max())

            is_dist_agent = algo in ['DistAgent', 'v2DistAgent']
            k_val = config.get('K', -999) if is_dist_agent else -999
            rep_gamma_shape = config.get('rep_gamma_shape', -999) if is_dist_agent else -999
            
            print(f'rep_gamma_shape: {rep_gamma_shape}, k_val: {k_val}')

            if np.array(history["_runtime"])[-1]/3600 < 1:
                continue

            results = {
                "algorithm": algo,
                "env": env_id,
                "seed": seed,
                "runtime": round(np.array(history["_runtime"])[-1]/3600, 2),
                "train_steps": train_steps,
                "K": k_val,
                "rep_gamma_shape": rep_gamma_shape,
                "best": best_reward

            }
            result_summary.append(results)

    # Convert the results to a pandas DataFrame
    df = pd.DataFrame(result_summary)
    print(df.head())
    print(df.shape)

    print(df.describe())

    # make directory if not exists
    if not os.path.exists("./results_analysis/data"):
        os.makedirs("./results_analysis/data")

    df.to_csv("./results_analysis/data/repK_results_summary.csv",
              index=False)

    run_results.to_csv("./results_analysis/data/repK_results_full.csv",
                       index=False)
    print("Results saved to repK_results_full.csv")


def extract_eval_rewards(run, algo, env, seed) -> pd.DataFrame:
    """
    Extract evaluation rewards from a single run.

    Args:
        run: W&B run object
        algo_name: Algorithm name (for determining step key)

    Returns:
        DataFrame with columns: step, eval_reward, seed
"""
    if algo in SB3_ALGOS:
        metric_names = [
            'eval/mean_reward',
        ]
        step_name = 'global_step'
    else:
        metric_names = [
            'eval/avg_reward',
        ]
        step_name = 'step'

    history = run.scan_history(keys=metric_names + [step_name])
    data_points = []

    step_prev = -1
    counter = 0
    for row in history:

        reward = row.get(metric_names[0])
        reward = round(reward, 4)
        step = row.get(step_name)
        if step_prev == step:
            continue
        step_prev = step

        data_points.append({
            'step': step,
            'eval_reward': reward,
            'algo': algo,
            'env': env,
            'seed': seed
        })

        counter += 1

        # if counter > 3:
        #     break

    if counter == 0:
        print(f"    Warning: No evaluation data found for {run.name}")
        return None

    df = pd.DataFrame(data_points)
    df = df.sort_values('step')

    return df


def extract_eval_rewards_Faster(run, algo, env, seed) -> pd.DataFrame:
    """
    Faster version: Extract evaluation rewards by fetching entire history at once.

    Args:
        run: W&B run object
        algo: Algorithm name
        env: Environment name
        seed: Random seed

    Returns:
        DataFrame with columns: step, eval_reward, algo, env, seed
    """
    if algo in SB3_ALGOS:
        metric_name = 'eval/mean_reward'
        step_name = 'global_step'
    else:
        metric_name = 'eval/avg_reward'
        step_name = 'step'

    try:
        # Fetch entire history at once - much faster than scan_history
        history = _fetch_history_with_retry(run, keys=[metric_name, step_name], pandas=True)

        if history is None or history.empty:
            print(f"    Warning: No history data found for {run.name}")
            return None

        # Filter out rows where metric is NaN
        history = history.dropna(subset=[metric_name])

        if history.empty:
            print(f"    Warning: No evaluation data found for {run.name}")
            return None

        # Rename columns to our standard format
        df = pd.DataFrame({
            'step': history[step_name],
            'eval_reward': history[metric_name].round(4),
            'algo': algo,
            'env': env,
            'seed': seed
        })

        # Remove duplicate steps (keep first occurrence)
        df = df.drop_duplicates(subset=['step'], keep='first')

        # Sort by step
        df = df.sort_values('step').reset_index(drop=True)

        print(f"    Extracted {len(df)} evaluation points from {run.name}")
        return df

    except Exception as e:
        print(f"    Error extracting data from {run.name}: {e}")
        return None


if __name__ == "__main__":
    data_fetcher()
