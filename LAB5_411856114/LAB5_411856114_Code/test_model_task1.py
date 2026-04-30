"""
Evaluation script for Task 1: Vanilla DQN on CartPole-v1.

Usage:
    python test_model_task1.py --model_path LAB5_411856114_task1.pt
"""

import torch
import torch.nn as nn
import numpy as np
import random
import gymnasium as gym
import argparse


class DQN(nn.Module):
    """Fully-connected Q-network matching the CartPole architecture in dqn.py."""
    def __init__(self, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        return self.network(x)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    num_actions = env.action_space.n

    model = DQN(num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    rewards = []
    for seed in range(20):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        obs, _ = env.reset(seed=seed)
        state = obs
        done = False
        total_reward = 0.0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = obs

        rewards.append(total_reward)
        print(f"seed: {seed}, eval reward: {total_reward:.0f}")

    avg = np.mean(rewards)
    print(f"Average reward: {avg:.2f}")
    env.close()
    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the Task 1 model snapshot (.pt)")
    args = parser.parse_args()
    evaluate(args)
