"""
Evaluation script for Task 2 and Task 3: DQN on ALE/Pong-v5.

Runs 20 episodes with seeds 0–19 and prints per-episode reward + average.
Output format matches the assignment Figure 4.

Usage:
    python test_model_task3.py --model_path LAB5_411856114_task3_2500000.pt
    python test_model_task3.py --model_path LAB5_411856114_task2.pt --env_steps 0
"""

import torch
import torch.nn as nn
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import argparse
from collections import deque

gym.register_envs(ale_py)


class DQN(nn.Module):
    """CNN Q-network matching the Atari architecture in dqn.py."""
    def __init__(self, input_channels, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.network(x / 255.0)


class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    num_actions = env.action_space.n

    model = DQN(4, num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    preprocessor = AtariPreprocessor()
    rewards = []

    for seed in range(20):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        obs, _ = env.reset(seed=seed)
        state = preprocessor.reset(obs)
        done = False
        total_reward = 0.0

        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = preprocessor.step(next_obs)

        rewards.append(total_reward)
        print(f"Environment steps: {args.env_steps}, seed: {seed}, eval reward: {total_reward:.0f}")

    avg = np.mean(rewards)
    print(f"Average reward: {avg:.2f}")
    env.close()
    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model snapshot (.pt)")
    parser.add_argument("--env_steps", type=int, default=0,
                        help="Environment steps at which this snapshot was saved (for display)")
    args = parser.parse_args()
    evaluate(args)
