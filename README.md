# Learning State-Action Value Geometry with Cosine Similarity for Continuous Control

Official code repository for **SAVGO: Learning State-Action Value Geometry with Cosine Similarity for Continuous Control**.

[Arxiv Link of the Paper](https://arxiv.org/abs/2605.00787)

SAVGO studies continuous-control reinforcement learning through the geometry of state-action values. Instead of relying only on scalar value estimates, the method learns a representation space where cosine similarity captures useful structure among state-action pairs and supports policy improvement. This repository contains the implementation used for the paper, including the main SAVGO/DistanceRL agent, ablations, representation models, and Stable-Baselines3 PPO/TD3/SAC/TQC comparison baselines.

<img width="1667" height="1361" alt="image" src="https://github.com/user-attachments/assets/2bc5bfb3-2268-4d54-9751-4d8b10af3f92" />

## Installation

Create a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

For GPU runs, install a PyTorch build that matches your CUDA version before running experiments.

## Run

Train DistanceRL on a Box2D task:

```bash
python main.py --algo DistAgent --env-id LunarLanderContinuous-v3 --device cuda
```

Train DistanceRL on a MuJoCo task:

```bash
python main.py --algo DistAgent --env-id HalfCheetah-v5 --device cuda --total-steps 1000000
```

Run a Stable-Baselines3 baseline:

```bash
python main.py --algo sac --env-id HalfCheetah-v5 --device cuda --total-steps 1000000
```

Use CPU instead of CUDA:

```bash
python main.py --algo DistAgent --env-id LunarLanderContinuous-v3 --device cpu
```

Optional Weights & Biases logging:

```bash
python main.py --algo DistAgent --env-id HalfCheetah-v5 --log_to_wandb --project_name DistRL
```

Model checkpoints are written to `saved_models/`, and logs are written to `logs/`.

## Citation

If you use this code, please cite the accompanying publication.
```
@misc{orfanoudakis2026savgolearningstateactionvalue,
      title={SAVGO: Learning State-Action Value Geometry with Cosine Similarity for Continuous Control}, 
      author={Stavros Orfanoudakis and Pedro P. Vergara},
      year={2026},
      eprint={2605.00787},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.00787}, 
}
