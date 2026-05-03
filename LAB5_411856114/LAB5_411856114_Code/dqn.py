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
        self.size = 0
        self.max_priority = 1.0

    def __len__(self):
        return self.size

    def add(self, transition):
        ########## YOUR CODE HERE (for Task 3) ##########
        if self.size < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity #circular buffer, when pos reaches capacity, it wraps around to the beginning
        self.size = min(self.size + 1, self.capacity) #keep track of the current size of buffer

        ########## END OF YOUR CODE (for Task 3) ##########
        return
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ##########
        prios = self.priorities[:self.size] #get the valid priorities (up to current size of buffer)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs) #sample indices according to the probabilities

        weights = (self.size * probs[indices]) ** (-self.beta) #importance sampling weights
        weights /= weights.max() #normalize weights
        batch = [self.buffer[i] for i in indices] #get the sampled transitions
        return batch, indices, weights.astype(np.float32) #return the batch, indices, and weights as float32 for stability  

        ########## END OF YOUR CODE (for Task 3) ##########
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ##########

        eps = 1e-2
        new_priorities = np.abs(errors) + eps
        self.priorities[indices] = new_priorities
        self.max_priority = max(self.max_priority, new_priorities.max()) #update max priority for new transitions

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
        self.use_ddqn = getattr(args, "use_ddqn", False)
        self.use_per = getattr(args, "use_per", False)
        self.per_beta_start = getattr(args, "per_beta_start", 0.4)
        self.per_beta_frames = getattr(args, "per_beta_frames", 2000000)
        self.n_step = getattr(args, "n_step", 1)
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.gamma_n = self.gamma ** self.n_step
        self.student_id = getattr(args, "student_id", "")
        # Milestones (env steps) for task3 snapshot grading
        self.milestones = [600_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000]
        self._milestones_hit = set()
        self._task3_best_saved = False
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        if self.use_ddqn:
            print("Using Double DQN target")
        if self.use_per:
            print("Using Prioritized Experience Replay")
            self.memory = PrioritizedReplayBuffer(args.memory_size, alpha=args.per_alpha, beta=args.per_beta_start)
        else:
            self.memory = deque(maxlen=args.memory_size)
        if self.n_step > 1:
            print(f"Using {self.n_step}-step return")
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def _push_to_replay(self, transition):
        if self.use_per:
            self.memory.add(transition)
        else:
            self.memory.append(transition)

    def _emit_n_step(self):
        # Compute n-step return over current contents of n_step_buffer.
        # Stops accumulating reward at the first terminal inside the window.
        R = 0.0
        for k, (_, _, r_k, _, d_k) in enumerate(self.n_step_buffer):
            R += (self.gamma ** k) * r_k
            if d_k:
                break
        s0, a0, _, _, _ = self.n_step_buffer[0]
        _, _, _, sn, _ = self.n_step_buffer[-1]
        done_n = any(t[4] for t in self.n_step_buffer)
        self._push_to_replay((s0, a0, R, sn, done_n))

    def _store_transition(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # Wait until we have n transitions, unless the episode ended.
        if len(self.n_step_buffer) < self.n_step and not done:
            return

        # Emit one n-step transition (oldest s/a, accumulated R, newest s').
        self._emit_n_step()

        # If episode ended, drain remaining transitions with progressively
        # shorter horizons so we don't lose the tail of the episode.
        if done:
            self.n_step_buffer.popleft()
            while len(self.n_step_buffer) > 0:
                self._emit_n_step()
                self.n_step_buffer.popleft()

    def run(self, episodes=1000, start_ep=0):
        for ep in range(start_ep, episodes):
            obs, _ = self.env.reset()
            if self.is_atari:
                state = self.preprocessor.reset(obs)
            else:
                state = obs #change to obs since we want to store the raw observation in the replay buffer, and preprocess it during training
            self.n_step_buffer.clear()  # don't leak transitions across episodes
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
                self._store_transition(state, action, reward, next_state, done)

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                # ---- Task 3 milestone snapshot (saves at exact env-step counts) ----
                for ms in self.milestones:
                    if ms not in self._milestones_hit and self.env_count >= ms:
                        sid = self.student_id or "STUDENT"
                        ms_path = os.path.join(self.save_dir, f"LAB5_{sid}_task3_{ms}.pt")
                        torch.save(self.q_net.state_dict(), ms_path)
                        wandb.save(ms_path)
                        self._milestones_hit.add(ms)
                        print(f"[Milestone] Saved {ms_path} at env_step {self.env_count}")

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
                wandb.save(model_path)
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
                    "milestones_hit": list(self._milestones_hit),
                    "task3_best_saved": self._task3_best_saved,
                }, checkpoint_path)
                wandb.save(checkpoint_path)

                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                # First time eval hits >=19 on Pong: save the task3_best snapshot
                if self.is_atari and eval_reward >= 19 and not self._task3_best_saved:
                    sid = self.student_id or "STUDENT"
                    best_task3_path = os.path.join(self.save_dir, f"LAB5_{sid}_task3_best.pt")
                    torch.save(self.q_net.state_dict(), best_task3_path)
                    wandb.save(best_task3_path)
                    self._task3_best_saved = True
                    print(f"[Task3-Best] Saved {best_task3_path} at env_step {self.env_count} (eval_reward={eval_reward})")

                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    wandb.save(model_path)
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
                        "milestones_hit": list(self._milestones_hit),
                        "task3_best_saved": self._task3_best_saved,
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
        self._milestones_hit = set(ckpt.get("milestones_hit", []))
        self._task3_best_saved = ckpt.get("task3_best_saved", False)
        start_ep = ckpt["last_episode"] + 1
        print(f"Resumed from {path}: episode {start_ep}, env_count {self.env_count}, epsilon {self.epsilon:.4f}")
        if self._milestones_hit:
            print(f"Already-hit milestones: {sorted(self._milestones_hit)}")
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
        if self.use_per:
            # Anneal beta from per_beta_start -> 1.0 over per_beta_frames env steps
            frac = min(1.0, self.env_count / self.per_beta_frames)
            self.memory.beta = self.per_beta_start + (1.0 - self.per_beta_start) * frac
            batch, indices, is_weights = self.memory.sample(self.batch_size)
            is_weights_tensor = torch.from_numpy(is_weights).to(self.device)
        else:
            batch = random.sample(self.memory, self.batch_size) # get 32 randomly sampled transitions from replay buffer(memory)
            indices, is_weights, is_weights_tensor = None, None, None
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
            if self.use_ddqn:
                # Double DQN: online net selects the argmax action, target net evaluates it.
                # Mitigates the overestimation bias of max_a Q_target(s', a).
                next_actions = self.q_net(next_states).argmax(1, keepdim=True)        # (B, 1)
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)  # (B,)
            else:
                next_q_values = self.target_net(next_states).max(1)[0] #output format (batch_size, num_actions), and num_actions is 2 for CartPole, so max(1) will give us the max q value for each transition in the batch
                #to simplify, q_values basically is a 32x2 tensor (a tensor is a generalization of matrices to higher dimensions)
                # ^ run all 32 transitions, then take max of actions (dimension 1)
            targets = rewards + self.gamma_n * next_q_values * (1-dones)
        
        if self.use_per:
            td_errors = q_values - targets
            loss_each = F.smooth_l1_loss(q_values, targets, reduction='none') #Huber loss for each transition in the batch
            loss = (is_weights_tensor * loss_each).mean() #weighted loss for PER
        else:
            loss = F.smooth_l1_loss(q_values, targets) #Huber loss is stable because it is linear for
        # large errors and quadraticfor small errors.
        #If we use mse, the initial large errors can cause very large gradients, which is unstable
        self.last_loss = loss.item()
        self.last_q_mean = q_values.mean().item()
        self.last_q_max = q_values.max().item()
        self.optimizer.zero_grad() #zero out gradients before backpropagation
        loss.backward() #backpropagation to compute gradients
        if self.use_per:
            self.memory.update_priorities(indices, td_errors.detach().cpu().numpy()) #update priorities in PER based on TD errors
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
    parser.add_argument("--use-ddqn", action="store_true",
                        help="Use Double DQN target (decouple action selection / evaluation)")
    parser.add_argument("--use-per", action="store_true")
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-beta-frames", type=int, default=2000000)
    parser.add_argument("--n-step", type=int, default=1,
                        help="n-step return (1 = vanilla single-step DQN)")


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
