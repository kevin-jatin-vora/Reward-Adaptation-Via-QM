import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import os

# Setup
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Environment setup
env = gym.make("MountainCar-v0")
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Hyperparameters
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, eps, train_flag):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        if train_flag:
            self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + gamma * q_targets_next * (1 - dones)
        q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

def dqn_fixed_steps(n_steps=500000, max_t=200, eps_start=1.0, eps_end=0.01, eps_decay=0.997, test_interval=300, test_runs=10):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    state = env.reset()[0]
    score = 0
    e_step = 0
    for step in range(1, n_steps + 1):
        e_step += 1
        action = agent.act(state, eps, train_flag=True)
        next_state, reward, done, _, __ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward

        if done or e_step % max_t == 0:
            e_step = 0
            state = env.reset()[0]
            scores_window.append(score)
            print(f'\rStep {step}\tAverage Score: {np.mean(scores_window):.2f}\teps: {eps}', end="")
            score = 0
            eps = max(eps_end, eps_decay * eps)

        if step % test_interval == 0:
            avg_test_score = test_agent(test_runs, max_t)
            scores.append(avg_test_score)
    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_DQN2.pth')
    return scores

def test_agent(test_runs=10, max_t=200):
    env_test = gym.make("MountainCar-v0")
    test_scores = []
    for _ in range(test_runs):
        state = env_test.reset()[0]
        score = 0
        done = False
        step = 0
        while step < max_t and not done:
            step += 1
            action = agent.act(state, eps=0.0, train_flag=False)
            next_state, reward, done, _, __ = env_test.step(action)
            state = next_state
            score += reward
        test_scores.append(score)
    env_test.close()
    return np.mean(test_scores)

run = 10
N_step = 1300000
test_steps = 2000
n_episodes = int(N_step / test_steps)
reward_run = np.zeros((run, n_episodes))

for i in range(run):
    print("########################### " + str(i))
    agent = Agent(state_size=2, action_size=3, seed=0)
    scores = dqn_fixed_steps(n_steps=N_step, max_t=200, eps_start=1.0, eps_end=0.05, eps_decay=0.995, test_interval=test_steps, test_runs=10)
    reward_run[i] = scores
    np.save("dqn_mountaincar3.npy", reward_run)
    plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title(f'Run {i}')
    plt.show()
