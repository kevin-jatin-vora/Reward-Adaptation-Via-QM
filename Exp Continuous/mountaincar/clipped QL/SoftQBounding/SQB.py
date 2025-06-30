import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from bound_utils import bounds, calculate_clip_loss  # Ensure this import is present
# Environment setup
env = gym.make("MountainCar-v0")
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64, seed=0):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

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

class BoundedDQNAgent:
    def __init__(self, state_size, action_size, seed=0, gamma=0.99, lr=5e-4, tau=1e-3,
                 buffer_size=int(1e5), batch_size=64, prior_update_interval=2500, beta=3, eta=0.5):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.prior_update_interval = prior_update_interval
        self.beta = beta  # Temperature for soft Q-learning
        self.eta = eta    # Weight for clipping loss
        self.step_count = 0

        # Q-Networks
        self.q_network = QNetwork(state_size, action_size, seed=seed).to(device)
        self.q_target = QNetwork(state_size, action_size, seed=seed).to(device)
        self.q_hat = QNetwork(state_size, action_size, seed=seed).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        # self.q_hat.load_state_dict(self.q_network.state_dict())  # Initialize q_hat to match q_network
        
        path = r"C:\Users\kevin\Downloads\Continuous domain new attempt 2025\new env MC\mountaincar\behaviors\regression idea for neighbours\checkpoint_b2.pth"
        self.q_hat.load_state_dict(torch.load(path, map_location=device))  # Initialize q_hat to match q_network

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size, batch_size, seed)

    def act(self, state, eps, train_flag=True):
        if random.random() < eps:
            return random.randint(0, self.q_network.fc3.out_features - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_vals = self.q_network(state)
        return q_vals.argmax().item()

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.step_count += 1

        if len(self.memory) > self.batch_size:
            batch = self.memory.sample()
            self.learn(batch)

            if self.step_count % self.prior_update_interval == 0:
                self.update_q_hat()

    def update_q_hat(self):
        self.q_hat.load_state_dict(self.q_network.state_dict())

    def update_target_network(self):
        for target_param, local_param in zip(self.q_target.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    
    def learn(self, batch):
        states, actions, rewards, next_states, dones = batch
    
        # Current Q-values for taken actions
        q_values = self.q_network(states).gather(1, actions)
    
        with torch.no_grad():
            q_target_next = self.q_target(next_states)
            q_target_curr = self.q_target(states)
    
            q_hat_next = self.q_hat(next_states)
            q_hat_curr = self.q_hat(states)
            # q_hat_next = self.q_network(next_states)
            # q_hat_curr = self.q_network(states)
    
            # Compute bounds from both target and q_hat
            lb_target, ub_target = bounds(self.beta, self.gamma, rewards, dones, actions, q_target_next.unsqueeze(0), q_target_curr.unsqueeze(0))
            lb_hat, ub_hat = bounds(self.beta, self.gamma, rewards, dones, actions, q_hat_next.unsqueeze(0), q_hat_curr.unsqueeze(0))
    
            # Tightest bounds from both estimators
            lb = torch.max(lb_target, lb_hat)
            ub = torch.min(ub_target, ub_hat)
    
            # Soft value for target
            nA = q_target_next.shape[-1]
            V = 1 / self.beta * (torch.logsumexp(self.beta * q_target_next, dim=1, keepdim=True) - torch.log(torch.tensor([nA], device=device)))
            
        q_target = rewards + self.gamma * V * (1 - dones)

        # Clamp the soft target
        q_bounded = torch.clamp(q_target, min=lb, max=ub)
    
        # Bellman loss with bounded target
        bellman_loss = F.mse_loss(q_values, q_bounded)
    
        # Clipping loss (quadratic penalty outside bounds)
        clip_loss = ((q_values < lb).float() * (lb - q_values).pow(2) +
                     (q_values > ub).float() * (q_values - ub).pow(2)).mean()
    
        # Total loss
        total_loss = bellman_loss + self.eta * clip_loss
    
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.update_target_network()

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
    # torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_DQN2.pth')
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
    agent = BoundedDQNAgent(state_size=2, action_size=3, seed=0)
    scores = dqn_fixed_steps(n_steps=N_step, max_t=200, eps_start=1.0, eps_end=0.05, eps_decay=0.995, test_interval=test_steps, test_runs=10)
    reward_run[i] = scores
    np.save("sqb_b1_3.npy", reward_run)
    plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title(f'Run {i}')
    plt.show()

    