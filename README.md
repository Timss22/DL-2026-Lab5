# DL-2026 Lab 5: Value-Based Reinforcement Learning

**Course:** 535518 Deep Learning, Spring 2026  
**Student ID:** 411856114  
**Deadline:** 2026/05/05, 23:59

## Overview

Implementation of Deep Q-Networks (DQN) and its enhancements across three tasks:

| Task | Environment | Method |
|------|------------|--------|
| 1 | CartPole-v1 | Vanilla DQN (FC network) |
| 2 | ALE/Pong-v5 | Vanilla DQN (CNN + frame stack) |
| 3 | ALE/Pong-v5 | DDQN + Prioritized Experience Replay + 3-step return |

## Project Structure

```
LAB5_411856114/
├── LAB5_411856114.pdf              # Technical report
├── LAB5_411856114_task1.pt         # Task 1 submission model
├── LAB5_411856114_task2.pt         # Task 2 submission model
├── LAB5_411856114_task3_600000.pt  # Task 3 snapshot @ 600k steps
├── LAB5_411856114_task3_1000000.pt # Task 3 snapshot @ 1M steps
├── LAB5_411856114_task3_1500000.pt # Task 3 snapshot @ 1.5M steps
├── LAB5_411856114_task3_2000000.pt # Task 3 snapshot @ 2M steps
├── LAB5_411856114_task3_2500000.pt # Task 3 snapshot @ 2.5M steps
├── LAB5_411856114_task3_best.pt    # Task 3 snapshot when score >= 19
└── LAB5_411856114_Code/
    ├── dqn.py                      # Main implementation (DQNAgent, DQN, PER, preprocessor)
    ├── test_model_task1.py         # Eval script for Task 1 (20 seeds, seeds 0-19)
    ├── test_model_task3.py         # Eval script for Tasks 2 & 3 (20 seeds, seeds 0-19)
    ├── test_model.py               # Video recording eval script (Atari)
    ├── run_task1.sh                # Train Task 1
    ├── run_task2.sh                # Train Task 2
    ├── run_task3.sh                # Train Task 3 (+ ablation comments)
    ├── run_resume_task2.sh         # Resume Task 2 from checkpoint
    ├── run_resume_task3.sh         # Resume Task 3 from checkpoint
    └── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate

# NVIDIA GPU (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# AMD GPU (ROCm 6.1) — used on the OpenHW course server:
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1
# CPU only:
pip install torch torchvision

pip install -r LAB5_411856114/LAB5_411856114_Code/requirements.txt
```

## Training

All scripts must be run from `LAB5_411856114/LAB5_411856114_Code/`.

```bash
cd LAB5_411856114/LAB5_411856114_Code

# Task 1 — CartPole 
bash run_task1.sh

# Task 2 — Pong vanilla DQN 
bash run_task2.sh

# Task 3 — Enhanced DQN, DDQN + PER + 3-step 
bash run_task3.sh

# Resume from a wandb checkpoint (place checkpoint.pt in Code dir first)
bash run_resume_task2.sh
bash run_resume_task3.sh
```

Checkpoints are written to `results/task{1,2,3}/checkpoint.pt` every 100 episodes. Task 3 milestone snapshots (`LAB5_411856114_task3_{N}.pt`) are auto-saved at 600k, 1M, 1.5M, 2M, and 2.5M environment steps.

## Evaluation

```bash
cd LAB5_411856114/LAB5_411856114_Code

# Task 1 — CartPole (20 seeds, seeds 0-19)
python test_model_task1.py --model_path LAB5_411856114_task1.pt

# Task 2 — Pong vanilla DQN
python test_model_task3.py --model_path LAB5_411856114_task2.pt --env_steps 0

# Task 3 — milestone snapshots (example: 2.5M steps)
python test_model_task3.py --model_path LAB5_411856114_task3_2500000.pt --env_steps 2500000
python test_model_task3.py --model_path LAB5_411856114_task3_best.pt --env_steps 0
```

Evaluation runs 20 episodes with seeds 0–19 and prints per-episode reward and the mean.

## Implementation Details

All code lives in `dqn.py`. Key classes:

- **`DQN`** — fully-connected network for CartPole; CNN (3 conv layers → 512 FC → actions) for Atari
- **`AtariPreprocessor`** — grayscale + 84×84 resize + 4-frame stack
- **`PrioritizedReplayBuffer`** — priority-based sampling (alpha=0.6), IS-weight correction (beta annealed 0.4→1.0 over 2M steps)
- **`DQNAgent`** — orchestrates training; supports epsilon-greedy, DDQN target, PER, n-step return

### Task 3 hyperparameters

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam, lr=1e-4 |
| Batch size | 32 |
| Replay buffer | 200k |
| Discount factor γ | 0.99 |
| Epsilon decay | 0.9999955 (1.0 → 0.05) |
| Target update frequency | 10 000 steps |
| Replay start size | 50 000 |
| n-step return | 3 |
| PER α | 0.6 |
| PER β start | 0.4 |

Training is logged to [Weights & Biases](https://wandb.ai) under project `DLP-Lab5-DQN-CartPole`.
