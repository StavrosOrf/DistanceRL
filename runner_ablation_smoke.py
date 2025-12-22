"""
Lightweight smoke runner to verify each ablation agent initializes and trains for a few steps.
Runs Pendulum-v1 on CPU with a tiny horizon to keep memory small.
Use from a tmux pane: `python runner_ablation_smoke.py`.
"""
import subprocess
from pathlib import Path

# Priority list: all A* and B* agents
ABLATION_ALGOS = [
    # "DistAblationA1",
    # "DistAblationA2",
    # "DistAblationA3",
    # "DistAblationA4",
    # "DistAblationA5",
    # "DistAblationB1",
    # "DistAblationB2",
    # "DistAblationB3",
    # "DistAblationB4",
    # "DistAblationB5",
    # "DistAblationB6",
    # "DistAblationB7",
    # "DistAblationB8",
    # "DBC",
    # "DBCDet",
    "MICo",
]

# Minimal, low-memory run settings
BASE_CMD = [
    "python", "main.py",
    "--env-id", "Pendulum-v1",
    "--device", "cpu",
    "--total-steps", "2000",
    "--batch-size", "64",
    "--buffer-size", "50000",
    "--hidden-size", "64",
    "--eval-episodes", "1",
    "--eval-freq", "1000",
    "--warmup-steps", "500",
    "--updates-per-step", "1",
    "--normalize-obs", "1",
    "--expl-sigma", "0.2",
    "--rep-fixed-scale", "1.0",
]


def run_smoke():
    logs_dir = Path("./logs/smoke_runs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    for algo in ABLATION_ALGOS:
        cmd = BASE_CMD + ["--algo", algo, "--exp-prefix", f"smoke_{algo}"]
        print(f"\n>>> Running smoke test for {algo}")
        log_path = logs_dir / f"{algo}.log"
        with log_path.open("w") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        print(f"Done {algo} (exit {proc.returncode}), log: {log_path}")


if __name__ == "__main__":
    run_smoke()
