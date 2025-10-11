import wandb
import pandas as pd
import numpy as np
import tqdm as tqdm
import json

# Initialize API
import os
SB3_ALGOS = ["ppo", "td3", "sac", "tqc"]

os.environ['WANDB_HTTP_TIMEOUT'] = '300'


def data_fetcher():
    api = wandb.Api()

    # Replace 'your_project_name' and 'your_entity_name' with your actual project and entity
    project_name = "DistRL_Rep"
    entity_name = "stavrosorf"

    # Fetch runs from the specified project
    runs = api.runs(f"{entity_name}/{project_name}")
    print(f"Total runs fetched: {len(runs)}")

    # Display the filtered runs with group names

    run_results = pd.DataFrame()
    result_summary = []
    # use tqdm to display a progress bar
    for i, run in tqdm.tqdm(enumerate(runs), total=len(runs)):

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

        if algo not in SB3_ALGOS:
            continue

        print(
            f"Run {i+1}/{len(runs)}: - Algo: {algo} - Env: {env_id} - Seed: {seed}")

        data = extract_eval_rewards(run, algo, env_id, seed)

        run_results = pd.concat([run_results, data], ignore_index=True)

        history = run.history()
        if '_runtime' not in history:
            print(f"Run {run.id} has no _runtime key")
            continue
        
        best_reward = data['eval_reward'].max()

        if np.array(history["_runtime"])[-1]/3600 < 1:
            continue

        results = {
            "algorithm": algo,
            "seed": seed,
            "runtime": round(np.array(history["_runtime"])[-1]/3600, 2),
            "best": best_reward

        }
        result_summary.append(results)

    # Convert the results to a pandas DataFrame
    df = pd.DataFrame(result_summary)
    print(df.head())
    print(df.shape)

    print(df.describe())

    df.to_csv("./results_analysis/data/results_summary.csv",
              index=False)

    run_results.to_csv("./results_analysis/data/results_full.csv",
                       index=False)
    print("Results saved to results_full.csv")


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
        return pd.DataFrame(columns=['step', 'eval_reward', 'seed'])

    df = pd.DataFrame(data_points)
    df = df.sort_values('step')

    # print(
    #     f"  Extracted {len(df)} evaluation points from {run.name} (seed={seed})")
    return df


if __name__ == "__main__":
    data_fetcher()
