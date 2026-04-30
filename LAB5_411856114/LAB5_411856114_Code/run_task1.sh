#!/bin/bash
# Task 1: Vanilla DQN on CartPole-v1
# Expected to converge in ~100k-300k env steps (minutes on GPU).

python dqn.py \
  --env-name CartPole-v1 \
  --wandb-run-name task1-cartpole-vanilla \
  --save-dir ./results/task1 \
  --episodes 3000 \
  --student-id 411856114 \
  --batch-size 64 \
  --memory-size 10000 \
  --lr 0.0005 \
  --discount-factor 0.99 \
  --epsilon-start 1.0 \
  --epsilon-decay 0.9995 \
  --epsilon-min 0.01 \
  --target-update-frequency 500 \
  --replay-start-size 1000 \
  --max-episode-steps 500 \
  --train-per-step 1
