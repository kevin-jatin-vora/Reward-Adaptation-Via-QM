import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import base64, io
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from collections import deque, namedtuple

env = gym.make("MountainCar-v0") #render_mode='human'
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_local.load_state_dict(torch.load('checkpoint_sfql_combined_init.pth'))
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target.load_state_dict(torch.load('checkpoint_sfql_combined_init.pth'))
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps, train_flag):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        if(train_flag):
            self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # Obtain random minibatch of tuples from D
        states, actions, rewards, next_states, dones = experiences

        ## Compute and minimize the loss
        ### Extract next maximum estimated value from target network
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        ### Calculate target value from bellman equation
        q_targets = rewards + gamma * q_targets_next * (1 - dones)
        ### Calculate expected value from local network
        q_expected = self.qnetwork_local(states).gather(1, actions)
        
        ### Loss calculation (we used Mean squared error)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
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
    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_sfql.pth')
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
# Parameters
start_run = 0        # inclusive
end_run = 10         
run = 10
N_step = 1300000
test_steps = 2000
n_episodes = int(N_step / test_steps)
npy_file = "sfql_mountaincar2.npy"

# Try to load existing results or initialize new array
try:
    reward_run = np.load("sfql_mountaincar3.npy")
    assert reward_run.shape == (run, n_episodes)
    print("Loaded existing reward_run array.")
except (FileNotFoundError, AssertionError):
    reward_run = np.zeros((run, n_episodes))
    print("Initialized new reward_run array.")

# Run only for runs 8 and 9
for i in range(start_run, end_run):
    print(f"########################### Run {i}")
    agent = Agent(state_size=2, action_size=3, seed=0)
    scores = dqn_fixed_steps(
        n_steps=N_step,
        max_t=200,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.995,
        test_interval=test_steps,
        test_runs=10
    )
    reward_run[i] = scores
    np.save(npy_file, reward_run)

    plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title(f'Run {i}')
    plt.show()

# run = 10
# N_step = 1200000
# test_steps = 2000
# n_episodes = int(N_step / test_steps)
# reward_run = np.zeros((run, n_episodes))

# for i in range(run):
#     print("########################### " + str(i))
#     agent = Agent(state_size=2, action_size=3, seed=0)
#     scores = dqn_fixed_steps(n_steps=N_step, max_t=200, eps_start=1.0, eps_end=0.05, eps_decay=0.995, test_interval=test_steps, test_runs=10)
#     reward_run[i] = scores
#     np.save("sfql_mountaincar.npy", reward_run)
#     plt.figure()
#     plt.plot(np.arange(len(scores)), scores)
#     plt.ylabel('Score')
#     plt.xlabel('Episode #')
#     plt.title(f'Run {i}')
#     plt.show()
