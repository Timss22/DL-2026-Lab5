#!/bin/bash
# Task 3: Enhanced DQN (DDQN + PER + 3-step return) on ALE/Pong-v5.
# Milestone snapshots are automatically saved at 600k, 1M, 1.5M, 2M, 2.5M env steps.
# task3_best.pt is saved the first time eval score >= 19.
#
# To resume a previous session:
#   add --resume-from ./results/task3/checkpoint_step<N>.pt

python dqn.py \
  --env-name ALE/Pong-v5 \
  --wandb-run-name task3-pong-ddqn-per-multistep \
  --save-dir ./results/task3 \
  --episodes 10000 \
  --student-id 411856114 \
  --batch-size 32 \
  --memory-size 200000 \
  --lr 0.0001 \
  --discount-factor 0.99 \
  --epsilon-start 1.0 \
  --epsilon-decay 0.9999955 \
  --epsilon-min 0.05 \
  --target-update-frequency 1000 \
  --replay-start-size 10000 \
  --max-episode-steps 100000 \
  --train-per-step 1 \
  --use-ddqn \
  --use-per \
  --n-step 3 \
  --per-alpha 0.6 \
  --per-beta-start 0.4 \
  --per-beta-frames 2000000

# -----------------------------------------------------------------------
# Ablation runs (uncomment and run separately for the ablation study)
# -----------------------------------------------------------------------

# DDQN only:
# python dqn.py --env-name ALE/Pong-v5 --wandb-run-name task3-ablation-ddqn \
#   --save-dir ./results/ablation_ddqn --episodes 10000 --student-id 411856114 \
#   --batch-size 32 --memory-size 200000 --lr 0.0001 --discount-factor 0.99 \
#   --epsilon-start 1.0 --epsilon-decay 0.9999955 --epsilon-min 0.05 \
#   --target-update-frequency 1000 --replay-start-size 10000 \
#   --max-episode-steps 100000 --use-ddqn

# PER only:
# python dqn.py --env-name ALE/Pong-v5 --wandb-run-name task3-ablation-per \
#   --save-dir ./results/ablation_per --episodes 10000 --student-id 411856114 \
#   --batch-size 32 --memory-size 200000 --lr 0.0001 --discount-factor 0.99 \
#   --epsilon-start 1.0 --epsilon-decay 0.9999955 --epsilon-min 0.05 \
#   --target-update-frequency 1000 --replay-start-size 10000 \
#   --max-episode-steps 100000 --use-per --per-alpha 0.6 --per-beta-start 0.4

# Multi-step only (n=3):
# python dqn.py --env-name ALE/Pong-v5 --wandb-run-name task3-ablation-nstep \
#   --save-dir ./results/ablation_nstep --episodes 10000 --student-id 411856114 \
#   --batch-size 32 --memory-size 200000 --lr 0.0001 --discount-factor 0.99 \
#   --epsilon-start 1.0 --epsilon-decay 0.9999955 --epsilon-min 0.05 \
#   --target-update-frequency 1000 --replay-start-size 10000 \
#   --max-episode-steps 100000 --n-step 3
