#!/bin/bash
# Resume Task 2 training from a checkpoint downloaded from wandb (./checkpoint.pt).

python3 dqn.py \
  --env-name ALE/Pong-v5 \
  --wandb-run-name task2-pong-vanilla \
  --save-dir ./results/task2 \
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
  --resume-from ./checkpoint.pt
