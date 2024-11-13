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
from collections import defaultdict
import dill
from ple import PLE
from ple.games.pong import Pong
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
game = Pong(MAX_SCORE=5)
env = PLE(game, fps=30, display_screen=True)
from collections import defaultdict
import pickle

# Define a function to return [0]*4 as default value
def default_value():
    return [0] * 3
with open('Q_SFQL_combined.pickle', 'rb') as handle:
    Qc = dill.load(handle)
Qcombined = defaultdict(default_value, Qc)
# Given information
width = 64
height = 48
cpu_speed_ratio = 0.6
players_speed_ratio = 0.4
ball_speed_ratio = 0.75

# Calculate paddle dimensions and distances
paddle_width = round(width * 0.023)
paddle_height = round(height * 0.15)
paddle_dist_to_wall = round(width * 0.0625)

# Calculate speeds
players_speed = players_speed_ratio * height
cpu_speed = cpu_speed_ratio * height
ball_speed = ball_speed_ratio * height

# Calculate maximum velocities
MAX_PADDLE_VELOCITY = players_speed_ratio * height
MAX_BALL_VELOCITY_X = ball_speed_ratio * height
MAX_BALL_VELOCITY_Y = ball_speed_ratio * height

# Calculate screen dimensions
SCREEN_WIDTH = width
SCREEN_HEIGHT = height
def set_buckets_and_actions():
    number_of_buckets = (3, 4, 3, 8, 8, 4, 4)   # Buckets in each dimension
    number_of_actions = 3  # Assuming Pong has 2 actions (up and down)
    
    # Set state value bounds based on the calculated values
    state_value_bounds = [
        [0, SCREEN_HEIGHT - PADDLE_HEIGHT],        # Player Y Position
        [-MAX_PADDLE_VELOCITY, MAX_PADDLE_VELOCITY], # Player Velocity
        [0, SCREEN_HEIGHT - PADDLE_HEIGHT],        # CPU Y Position
        [0, SCREEN_WIDTH],                          # Ball X Position
        [0, SCREEN_HEIGHT],                         # Ball Y Position
        [-MAX_BALL_VELOCITY_X, MAX_BALL_VELOCITY_X], # Ball X Velocity
        [-MAX_BALL_VELOCITY_Y, MAX_BALL_VELOCITY_Y]  # Ball Y Velocity
    ]
    
    return number_of_buckets, number_of_actions, state_value_bounds


def bucketize(state):
    state = state.reshape(-1,1)
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

def de_bucketize(bucket_indexes):
    state = []
    for i in range(len(bucket_indexes)):
        # Calculate the range of values for the current bucket
        bucket_min = state_value_bounds[i][0] + (bucket_indexes[i] / (number_of_buckets[i] - 1)) * (state_value_bounds[i][1] - state_value_bounds[i][0])
        bucket_max = state_value_bounds[i][0] + ((bucket_indexes[i] + 1) / (number_of_buckets[i] - 1)) * (state_value_bounds[i][1] - state_value_bounds[i][0])
        
        # Calculate the center of the bucket
        bucket_center = (bucket_min + bucket_max) / 2
        
        state.append(bucket_center)
    return state

# Calculate values
SCREEN_WIDTH = width
SCREEN_HEIGHT = height
PADDLE_HEIGHT = paddle_height
MAX_PADDLE_VELOCITY = players_speed_ratio * height
MAX_BALL_VELOCITY_X = ball_speed_ratio * height
MAX_BALL_VELOCITY_Y = ball_speed_ratio * height

# Set buckets and actions
number_of_buckets, number_of_actions, state_value_bounds = set_buckets_and_actions()

print('State shape: ',len(env.getGameState()))
print('Number of actions: ', len(env.getActionSet()))



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
LR = 1e-4               # learning rate 
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

    def act(self, state, eps=0.):
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
        self.qnetwork_local.train()
        feature = bucketize(state)
        action = np.argmax(Qcombined[tuple(feature)])
    
        # Map action to numerical value
        if action == 119:  # Move paddle up
            action = 0
        elif action == 115:  # Move paddle down
            action = 1
        else:  # No action
            action = 2
    
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

def dqn(n_episodes=2000, max_t=500, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env.reset_game()
        env.init()
        state = np.array(list(env.getGameState().values())).reshape(1, 7)
        score = 0
        for t in range(max_t):
        # while True:
            action = agent.act(state, eps)
            # reward = env.act(action)
            # Execute the action based on the mapped action value
            if action == 0:  # Mapped action for moving paddle up
                reward = env.act(119)
            elif action == 1:  # Mapped action for moving paddle down
                reward = env.act(115)
            else:  # No action
                reward = env.act(None)
            # vel = game.agentPlayer.vel.y
            # r_vel = -abs(vel) if abs(vel)<=0.2 else -25
            # r_vel = r_vel/3
            # reward*=2
            # reward+=r_vel
            next_state = np.array(list(env.getGameState().values()))
            done = env.game_over()
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}, Current Score: {:.2f}'.format(i_episode, np.mean(scores_window), env.score()), end="")
        # if i_episode % 100 == 0:
        #     print(t)
        #     print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        # if np.mean(scores_window)>=0:
        #     # print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
        #     torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            # break
    return scores


run=1
n_episodes = 2500
reward_run=np.zeros((run,n_episodes))
for i in range(run):
    agent = Agent(state_size=7, action_size=3, seed=i)
    scores = dqn(n_episodes)
    reward_run[i]=scores
    # np.save("dqn_new.npy", reward_run)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_sfql_combined_init.pth')