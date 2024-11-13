import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import base64, io

import numpy as np
from collections import deque, namedtuple

env = gym.make('LunarLander-v2')
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

import numpy as np
from collections import defaultdict
import pickle

# Define a function to return [0]*4 as default value
def default_value():
    return [0] * 4

with open('Q_UB.pickle', 'rb') as handle:
    ub = pickle.load(handle)
ub = defaultdict(default_value, ub)

with open('Q_LB.pickle', 'rb') as handle:
    lb = pickle.load(handle)
lb = defaultdict(default_value, lb)

sub = set(ub.keys())

slb = set(lb.keys())

s = list(slb.union(sub))
for i in s:
    for a in range(4):
        if(ub[i][a]<lb[i][a]):
            ub[i][a]=lb[i][a]
        if(lb[i][a]>ub[i][a]):
            lb[i][a]=ub[i][a]

# s = list(slb.intersection(sub))
info=[]
final_actions=set(list(range(4)))
prune={}
state_action={}


for i in s:
    alist=[]
    for action_l in range(4):
        for action_u in range(4):
            if(action_l==action_u):
                continue
            if( lb[i][action_l]-ub[i][action_u] > 1e-7):
                info.append((i,action_l, action_u))
                alist.append(action_u)
    prune[i]= set(alist)
    state_action[i]= final_actions.difference(set(alist))
print(((len(s))*4-sum([len(state_action[i]) for i in state_action.keys()]))/((len(s))*4))

env = gym.make("LunarLander-v2")

def set_buckets_and_actions():
    number_of_buckets =  (5,5,5,5,10,15,1,1)#buckets in each dimension
    number_of_actions = env.action_space.n
    
    #Creating a 2-tuple with the original bounds of each dimension
    state_value_bounds = list(zip(env.observation_space.low,env.observation_space.high))
    
    return number_of_buckets, number_of_actions, state_value_bounds

number_of_buckets, number_of_actions, state_value_bounds = set_buckets_and_actions()


def bucketize(state):
    bucket_indexes = []
    for i in range(len(state)):
        if state[i] <= state_value_bounds[i][0]:
            bucket_index = 0
        elif state[i] >= state_value_bounds[i][1]:
            bucket_index = number_of_buckets[i] - 1
        else:
            bound_width = state_value_bounds[i][1] - state_value_bounds[i][0]
            offset = (number_of_buckets[i]-1) * state_value_bounds[i][0]/bound_width
            scaling = (number_of_buckets[i]-1) / bound_width
            bucket_index = int((scaling*state[i] - offset))
        bucket_indexes.append(bucket_index)
    return tuple(bucket_indexes)

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
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
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
        # print(state.numpy()[0])
        features = bucketize(state.numpy()[0])
        if random.random() < eps:
            if(tuple(features) in s):
                if(len(state_action[tuple(features)])!=0):
                    action = np.random.choice(list(state_action[tuple(features)]))
                else:
                    action =  random.choice(np.arange(self.action_size))
            else:
                action =  random.choice(np.arange(self.action_size))
        # Epsilon-greedy action selection
        # if random.random() > eps:
        #     return np.argmax(action_values.cpu().data.numpy())
        else:
            action =  np.argmax(action_values.cpu().data.numpy())
        # print((state.numpy()[0],action))
        return action

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

import numpy as np
from collections import deque

def dqn_fixed_steps(agent, n_steps=500000, max_t=2500, eps_start=1.0, eps_end=0.01, eps_decay=0.997, test_interval=300, test_runs=10):
    """Deep Q-Learning with fixed steps.
    
    Params
    ======
        n_steps (int): maximum number of steps for training
        max_t (int): maximum number of timesteps per episode (before reset if not done)
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per step) for decreasing epsilon
        test_interval (int): number of steps between each test run
        test_runs (int): number of test episodes in each test run
    """
    scores = []                        # list containing scores from each step
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    state = env.reset()[0]
    score = 0
    e_step=0
    for step in range(1, n_steps+1):
        e_step+=1
        # Agent takes an action using epsilon-greedy policy
        action = agent.act(state, eps, train_flag=True)
        next_state, reward, done, _, __ = env.step(action)
        if(reward==-100):
            reward=30
        elif(reward==100):
            reward = 150
        agent.step(state, action, reward, next_state, done)
        
        # Update the state and score
        state = next_state
        score += reward

        # If the episode is done, reset the environment
        if done or e_step%max_t==0:
            e_step=0
            state = env.reset()[0]
            scores_window.append(score)
            print('\rStep {}\tAverage Score: {:.2f}'.format(step, np.mean(scores_window)), end="")
            #scores.append(score)
            score = 0 
            # Decay epsilon (exploration rate)
            eps = max(eps_end, eps_decay * eps)

        # Print progress
        if step % 100000 == 0:
            # print('\rStep {}\tAverage Score: {:.2f}'.format(step, np.mean(scores_window)), end="")
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_RA.pth')
        
        # Call testing function every test_interval steps
        if step % test_interval == 0:
            avg_test_score = test_agent(agent, test_runs, max_t)
            #print('\nTest at step {}: Average Test Score: {:.2f}'.format(step, avg_test_score))
            scores.append(avg_test_score)
            
    return scores


def test_agent(agent, test_runs=1,max_t=2500):
    env_test = gym.make("LunarLander-v2")
    """Run a fixed number of test episodes where the agent acts greedily (no exploration).
    
    Params
    ======
        test_runs (int): number of test episodes to run
    """
    test_scores = []
    
    for i in range(test_runs):
        state = env_test.reset()[0]
        score = 0
        done = False
        step = 0
        while step<max_t and not done:
            step+=1
            action = agent.act(state, eps=0.0, train_flag=False)  # Use greedy action (no exploration)
            next_state, reward, done, _, __ = env_test.step(action)
            if(reward==-100):
                reward=30
            elif(reward==100):
                reward = 150
            state = next_state
            score += reward
        
        test_scores.append(score)
        env_test.close()
    
    return np.mean(test_scores)



run=10
N_step=1000000
test_steps=2000
n_episodes = int(N_step/test_steps)
reward_run=np.zeros((run,n_episodes))
for i in range(run):
    agent = Agent(state_size=8, action_size=4, seed=i)
    scores = dqn_fixed_steps(agent, n_steps=N_step, max_t=2500, eps_start=1.0, eps_end=0.01, eps_decay=0.995, test_interval=test_steps, test_runs=1) #dqn(n_episodes)
    reward_run[i]=scores
    np.save("dqn_RA.npy", reward_run)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()