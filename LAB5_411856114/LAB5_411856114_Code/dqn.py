# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, num_actions, input_channels=None):
        super(DQN, self).__init__()
        ########## YOUR CODE HERE (5~10 lines) ##########
        self.is_atari = input_channels is not None
        if input_channels is None:
            self.network = nn.Sequential(
                nn.Linear(4, 64), # start with 4 input from CartPole
                nn.ReLU(), # ReLU activation function after the first hidden layer
                nn.Linear(64, 64), # second hidden layer with 64 units since cartpole is simple, we can start with a small network
                nn.ReLU(), #ReLU activation function after the second hidden layer
                nn.Linear(64, num_actions), # output layer compressing 64 units to num_actions (2 for CartPole)
            ) #basic DQN architecture with 2 hidden layers of 64 units each, and ReLU activations. 
            # The output layer has num_actions units, representing the Q-values for each action.
        else:
            #Atari CNN:
            self.network = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3136, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
            )
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        if self.is_atari:
            x = x / 255.0 # normalize pixel values to [0,1]
        return self.network(x)


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ##########

        ########## END OF YOUR CODE (for Task 3) ##########
        return
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ##########

        ########## END OF YOUR CODE (for Task 3) ##########
        return
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ##########

        ########## END OF YOUR CODE (for Task 3) ##########
        return


class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.is_atari = "Atari" in env_name or "ALE" in env_name
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()
        self.test_preprocessor = AtariPreprocessor()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        print("Using device:", self.device)

        input_channels = 4 if self.is_atari else None
        self.q_net = DQN(self.num_actions, input_channels=input_channels).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions, input_channels=input_channels).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21 if self.is_atari else 0  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.memory = deque(maxlen=args.memory_size) #added, initializing replay buffer as a deque

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000, start_ep=0):
        for ep in range(start_ep, episodes):
            obs, _ = self.env.reset()
            if self.is_atari:
                state = self.preprocessor.reset(obs)
            else:
                state = obs #change to obs since we want to store the raw observation in the replay buffer, and preprocess it during training
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                if self.is_atari:
                    reward = np.clip(reward, -1, 1) #clip rewards for stability in Atari
                done = terminated or truncated
                if self.is_atari:
                    next_state = self.preprocessor.step(next_obs)
                else:
                    next_state = next_obs #change to next.obs since we want to store the raw observation in the replay buffer, and preprocess it during training
                self.memory.append((state, action, reward, next_state, done))

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed
                    wandb.log({
                        "Env Step Count": self.env_count,
                        "Loss": getattr(self, "last_loss", 0.0),
                        "Q Mean": getattr(self, "last_q_mean", 0.0),
                        "Q Max": getattr(self, "last_q_max", 0.0),
                    })
                    ########## END OF YOUR CODE ##########
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed
            wandb.log({
                "Episode": ep,
                "Episode Length": step_count,
                "Buffer Size": len(self.memory),
            })

            ########## END OF YOUR CODE ##########
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                checkpoint_path = os.path.join(self.save_dir, "checkpoint.pt")
                torch.save({
                    "q_net": self.q_net.state_dict(),
                    "target_net": self.target_net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "env_count": self.env_count,
                    "train_count": self.train_count,
                    "epsilon": self.epsilon,
                    "best_reward": self.best_reward,
                    "last_episode": ep,
                }, checkpoint_path)
                wandb.save(checkpoint_path)

                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    checkpoint_path = os.path.join(self.save_dir, "checkpoint.pt")
                    torch.save({
                        "q_net": self.q_net.state_dict(),
                        "target_net": self.target_net.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "env_count": self.env_count,
                        "train_count": self.train_count,
                        "epsilon": self.epsilon,
                        "best_reward": self.best_reward,
                        "last_episode": ep,
                    }, checkpoint_path)
                    wandb.save(checkpoint_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()
        if self.is_atari:
            state = self.test_preprocessor.reset(obs)
        else:
            state = obs
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            if self.is_atari:
                state = self.test_preprocessor.step(next_obs)
            else:
                state = next_obs

        return total_reward
    
    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.env_count = ckpt["env_count"]
        self.train_count = ckpt["train_count"]
        self.epsilon = ckpt["epsilon"]
        self.best_reward = ckpt["best_reward"]
        start_ep = ckpt["last_episode"] + 1
        print(f"Resumed from {path}: episode {start_ep}, env_count {self.env_count}, epsilon {self.epsilon:.4f}")
        return start_ep


    def train(self):

        if len(self.memory) < self.replay_start_size:
            return

        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        batch = random.sample(self.memory, self.batch_size) # get 32 randomly sampled transitions from replay buffer(memory)
        states, actions, rewards, next_states, dones = zip(*batch) # unzip the batch into separate outputs
        # the "*" in zip(*batch) is used to unpack the list of transitions into separate arguments for zip,
        # allowing us to group each component (state, action, reward, next_state, done) together 
        # across the batch



        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates
        with torch.no_grad(): #no_grad is to turn off gradient tracking, bcs no backpropagation.
            next_q_values = self.target_net(next_states).max(1)[0] #output format (batch_size, num_actions), and num_actions is 2 for CartPole, so max(1) will give us the max q value for each transition in the batch
            #to simplify, q_values basically is a 32x2 tensor (a tensor is a generalization of matrices to higher dimensions)
            # ^ run all 32 transitions, then take max of actions (dimension 1)
            targets = rewards + self.gamma * next_q_values * (1-dones)
        
        
        loss = F.smooth_l1_loss(q_values, targets) #Huber loss is stable because it is linear for
        # large errors and quadraticfor small errors.
        #If we use mse, the initial large errors can cause very large gradients, which is unstable
        self.last_loss = loss.item()
        self.last_q_mean = q_values.mean().item()
        self.last_q_max = q_values.max().item()
        self.optimizer.zero_grad() #zero out gradients before backpropagation
        loss.backward() #backpropagation to compute gradients
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10) #gradient clipping to prevent exploding gradients
        self.optimizer.step() #update the network parameters using the computed gradients

        ########## END OF YOUR CODE ##########

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint.pt to resume training from")
    parser.add_argument("--env-name", type=str, default="CartPole-v1")
    #added command line arguments for all the hyperparameters, so that we can easily change 
    # them when running the code without having to modify the code itself. 
    # This also allows us to easily log the hyperparameters in wandb for better tracking 
    # and visualization of our experiments.
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--student-id", type=str, default="")


    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(env_name=args.env_name, args=args)
    #previous was a bug because, we were initializing the agent with default arguments 
    #instead of the parsed arguments, which meant that the agent was not using 
    # the specified hyperparameters from the command line, and instead was using 
    # the default values defined in the DQNAgent class. This could lead to unexpected behavior 
    # during training, such as a different learning rate, batch size, or epsilon decay 
    # than what was intended. By passing the parsed arguments to the DQNAgent constructor, 
    # we ensure that the agent is initialized with the correct hyperparameters for training.
    start_ep = 0
    if args.resume_from is not None:
        start_ep = agent.load_checkpoint(args.resume_from)
    agent.run(episodes=args.episodes, start_ep=start_ep)
