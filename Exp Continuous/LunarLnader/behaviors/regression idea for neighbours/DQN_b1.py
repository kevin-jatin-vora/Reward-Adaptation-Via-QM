import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import base64, io
from collections import deque, namedtuple, defaultdict
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from collections import deque, namedtuple
import os
import joblib
env = gym.make('LunarLander-v2')
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
UPDATE_EVERY = 4
N_CLUSTERS = 2000
CLUSTER_UPDATE_INTERVAL = 1000
MIN_SAMPLES_FOR_UPDATE = 2500

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OnlineClusterer:
    def __init__(self, state_dim=4, action_dim=2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.kmeans = [MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto') 
                       for _ in range(action_dim)]
        self.state_buffers = [deque(maxlen=MIN_SAMPLES_FOR_UPDATE * 2) for _ in range(action_dim)]
        self.delta_buffers = [deque(maxlen=MIN_SAMPLES_FOR_UPDATE * 2) for _ in range(action_dim)]
        self.cluster_stats = [defaultdict(lambda: {
            'centroid': None,
            'min': np.full(state_dim, np.inf),
            'max': np.full(state_dim, -np.inf)
        }) for _ in range(action_dim)]
        self.steps_since_update = 0

    def add_sample(self, state, action, next_state):
        action_idx = np.argmax(action) if len(action) > 1 else action
        if action_idx >= self.action_dim:
            return

        delta_s = next_state - state
        self.state_buffers[action_idx].append(state)
        self.delta_buffers[action_idx].append(delta_s)
        self.steps_since_update += 1

        if (self.steps_since_update >= CLUSTER_UPDATE_INTERVAL and
            len(self.state_buffers[action_idx]) >= MIN_SAMPLES_FOR_UPDATE):
            self._update_clusters(action_idx)
            self.steps_since_update = 0

    def _update_clusters(self, action_idx):
        states = np.array(self.state_buffers[action_idx])
        deltas = np.array(self.delta_buffers[action_idx])
        kmeans_model = self.kmeans[action_idx]
    
        # Check if model has been fitted before
        if hasattr(kmeans_model, 'cluster_centers_'):
            old_centroids = kmeans_model.cluster_centers_.copy()
        else:
            old_centroids = None
    
        # Fit with new data
        kmeans_model.partial_fit(states)
        new_centroids = kmeans_model.cluster_centers_
        labels = kmeans_model.predict(states)
    
        # Prepare new stats container
        new_stats = defaultdict(lambda: {
            'centroid': None,
            'min': np.full(self.state_dim, np.inf),
            'max': np.full(self.state_dim, -np.inf)
        })
    
        # Assign centroids to new_stats
        for new_label in range(N_CLUSTERS):
            new_stats[new_label]['centroid'] = new_centroids[new_label]
    
        # Migrate old stats using predict if old centroids existed
        if old_centroids is not None:
            old_label_to_new_label = {}
            for old_label, old_centroid in enumerate(old_centroids):
                new_label = kmeans_model.predict(old_centroid.reshape(1, -1))[0]
                old_label_to_new_label.setdefault(new_label, []).append(old_label)
    
            for new_label, old_labels in old_label_to_new_label.items():
                mins = []
                maxs = []
                for old_label in old_labels:
                    old_stat = self.cluster_stats[action_idx][old_label]
                    mins.append(old_stat['min'])
                    maxs.append(old_stat['max'])
                if mins:
                    new_stats[new_label]['min'] = np.minimum.reduce(mins)
                    new_stats[new_label]['max'] = np.maximum.reduce(maxs)
    
        # Update stats with current data from buffer
        for cluster in range(N_CLUSTERS):
            mask = (labels == cluster)
            if np.any(mask):
                cluster_deltas = deltas[mask]
                new_stats[cluster]['min'] = np.minimum(new_stats[cluster]['min'], np.min(cluster_deltas, axis=0))
                new_stats[cluster]['max'] = np.maximum(new_stats[cluster]['max'], np.max(cluster_deltas, axis=0))
    
        # Save updated stats
        self.cluster_stats[action_idx] = new_stats


    def get_bounds(self, state, action):
        action_idx = np.argmax(action) if len(action) > 1 else action
        if action_idx >= self.action_dim or len(self.state_buffers[action_idx]) < MIN_SAMPLES_FOR_UPDATE:
            return None, None

        cluster = self.kmeans[action_idx].predict(state.reshape(1, -1))[0]
        stat = self.cluster_stats[action_idx].get(cluster)
        if stat is None:
            return None, None
        return stat['min'], stat['max']

    def save_models(self, path="clustering_models"):
        os.makedirs(path, exist_ok=True)
        for action_idx in range(self.action_dim):
            cluster_info = []
            for cluster_label, stat in self.cluster_stats[action_idx].items():
                cluster_info.append({
                    'cluster_label': cluster_label,
                    'min': stat['min'],
                    'max': stat['max']
                })
            joblib.dump({
                'kmeans': self.kmeans[action_idx],
                'stats': cluster_info,
                'sample_count': len(self.state_buffers[action_idx])
            }, f"{path}/action_{action_idx}.pkl")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, clusterer=None):
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
        self.clusterer = clusterer

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        # Q-Network mu
        self.qnetwork_local_mu = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target_mu = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer_mu = optim.Adam(self.qnetwork_local_mu.parameters(), lr=LR)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, eps):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        if self.clusterer is not None:
            action_one_hot = np.zeros(self.action_size)
            action_one_hot[action] = 1
            self.clusterer.add_sample(state, action_one_hot, next_state)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA, eps)

    def act(self, state, eps=0, ep=0):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        self.qnetwork_local_mu.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            action_values_mu = self.qnetwork_local_mu(state)
        self.qnetwork_local.train()
        self.qnetwork_local_mu.train()

        # Epsilon-greedy action selection
        if random.uniform(0, 1) < eps:
            return random.choice(np.arange(self.action_size))  # Random action
        else:
            # Select one action from Q_star using argmax and another from Q_mu using argmin
            action_star = np.argmax(action_values.cpu().data.numpy())  # Action that maximizes Q_star
            action_mu = np.argmin(action_values_mu.cpu().data.numpy())  # Action that minimizes Q_mu
            
            # Randomly return one of these two actions
            # return np.random.choice([action_star, action_mu], p=[0.5,0.5])
            if(ep%2==0):
                return action_star
            else:
                return action_mu
        # if random.random() > eps:
        #     return np.argmax(action_values.cpu().data.numpy())
        # else:
        #     return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, eps):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # Obtain random minibatch of tuples from D
        states, actions, rewards, next_states, dones = experiences
        
        # Compute importance sampling ratios
        a_stars = self.qnetwork_target(states).detach().max(1)[0].unsqueeze(1)
        a_mus = self.qnetwork_target_mu(states).detach().min(1)[0].unsqueeze(1)
        w_stars=[]
        w_mus=[]
        for a_star, a_mu, action in list(zip(a_stars,a_mus,actions)):
            # Probability under behavior policy
            b_action = (eps / self.action_size) + ((1 - eps) * 0.5 if action in [a_star, a_mu] else 0)
    
            # Probability under target policy for Q_star
            pi_star_action = (eps / self.action_size) + (1 - eps if action == a_star else 0)
            w_star = pi_star_action / b_action
    
            # Probability under target policy for Q_mu
            pi_mu_action = (eps / self.action_size) + (1 - eps if action == a_mu else 0)
            w_mu = pi_mu_action / b_action
            
            w_stars.append(w_star)
            w_mus.append(w_mu)

        w_stars = torch.tensor(w_stars).reshape(actions.shape)
        w_mus = torch.tensor(w_mus).reshape(actions.shape)
        ## Compute and minimize the loss
        ### Extract next maximum estimated value from target network
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        ### Calculate target value from bellman equation
        q_targets = w_stars*(rewards + gamma * q_targets_next * (1 - dones))
        ### Calculate expected value from local network
        q_expected = self.qnetwork_local(states).gather(1, actions)
        
        ### Loss calculation (we used Mean squared error)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)  

        # for mu#################################################################################
        ## Compute and minimize the loss
        ### Extract next minimum estimated value from target network
        q_targets_next_mu = self.qnetwork_target_mu(next_states).detach().min(1)[0].unsqueeze(1)
        ### Calculate target value from bellman equation
        q_targets_mu = w_mus*(rewards + gamma * q_targets_next_mu * (1 - dones))
        ### Calculate expected value from local network
        q_expected_mu = self.qnetwork_local_mu(states).gather(1, actions)
        
        ### Loss calculation (we used Mean squared error)
        loss_mu = F.mse_loss(q_expected_mu, q_targets_mu)
        self.optimizer_mu.zero_grad()
        loss_mu.backward()
        self.optimizer_mu.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local_mu, self.qnetwork_target_mu, TAU)                       

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

def dqn(n_episodes=2000, max_t=2500, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Modified Deep Q-Learning to collect reward training data."""
    env = gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    clusterer = OnlineClusterer(state_dim=state_size, action_dim=action_size)
    agent = Agent(state_size, action_size, seed=0, clusterer=clusterer)
    
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    
    for i_episode in range(1, n_episodes+1):
        state = env.reset()[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps, i_episode)
            next_state, r, done, _, __ = env.step(action)
            
            # Calculate reward using existing logic
            reward = r/2
            if state[4] < 0:
                reward += 10*abs(state[4]) + abs(state[5])
                if(state[0] > 0.5):
                    reward += 15
            
            if state[4] > 0:
                reward -= 10*abs(state[4]) - abs(state[5])
                if(state[0] < -0.55):
                    reward -= 16
            
            agent.step(state, action, reward, next_state, done, eps)
            state = next_state
            score += reward
            if done:
                break 
                
        scores_window.append(score)      
        scores.append(score)             
        eps = max(eps_end, eps_decay*eps)
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            
    # Save all models
    clusterer.save_models()
    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_b1.pth')
    torch.save(agent.qnetwork_local_mu.state_dict(), 'checkpoint_b1_mu.pth')
    
    return scores, clusterer


# Usage example:
run = 1
n_episodes = 20000

for i in range(run):
    agent = Agent(state_size=8, action_size=4, seed=i)
    scores, clusterer = dqn(n_episodes=n_episodes, eps_decay=0.9997)
    