import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
#np.random.seed(1)
import argparse
import yaml

# Load config
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

# Common parameters
init = cfg['start']
till = cfg['end']
nst_range = cfg['nst_range']
gamma = cfg['gamma']
x_percentage = cfg['x_percentage']

learning_rate = cfg['learning_rate']
epsilon_initial = cfg['epsilon_initial']
epsilon_decay = cfg['epsilon_decay']
epsilon_min = cfg['epsilon_min']
max_steps = cfg['max_steps']
N_steps = cfg['N_steps']
test_steps = cfg['test_steps']

class FrozenLake:
    def __init__(self, size=4):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.start = (0, 0)
        self.goal = (size - 1, 1)
        self.hole_prob = 0.1
        self.holes = [] 
        self.generate_holes()

    def generate_holes(self, num_pairs=4):
        pairs = set()
        while len(pairs) < num_pairs:
          pair = (np.random.randint(0, self.size), np.random.randint(0, self.size))
          if pair not in pairs and pair != self.goal and pair != self.start:
            pairs.add(pair)
        self.holes = list(pairs)
        for x,y in self.holes:
            self.grid[x, y] = -1

    def is_valid_position(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def get_reward1(self, state):
        x, y = state
        if (x, y) == self.holes[0] or (x, y) == self.holes[2]:
            return 1  
        elif(x, y) == self.holes[1] or (x, y) == self.holes[3]:
            return -1
        elif (x, y) == self.goal:
            return 0.6
        else:
            return -0.01  # For all other states

    def get_reward2(self, state):
        x, y = state
        if (x, y) == self.holes[1] or (x, y) == self.holes[3]:
            return 1  
        elif(x, y) == self.holes[0] or (x, y) == self.holes[2]:
            return -1
        elif (x, y) == self.goal:
            return 0.6 
        else:
            return -0.01  # For all other states

    def get_reward(self, state):
        x, y = state
        if (x, y) == self.goal:
            return 0.6*2  # Goal reached
        elif self.grid[x, y] == -1:
            return 0  # Fell into a hole
        else:
            return -0.02  # For all other states

    def get_neighboring_states(self, state):
        x, y = state
        neighbors = []
        for action in ['up', 'down', 'left', 'right']:
            new_x, new_y = self.transition(state, action)
            if self.is_valid_position(new_x, new_y):
                neighbors.append((new_x, new_y))
        return neighbors

    def transition(self, state, action):
        x, y = state
        if action == 'up':
            x -= 1
        elif action == 'down':
            x += 1
        elif action == 'left':
            y -= 1
        elif action == 'right':
            y += 1
        if self.is_valid_position(x, y):
            return x, y
        return state
              
def display_state(env, agent_location):
    state_grid = np.empty((env.size, env.size), dtype=object)
    
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) == agent_location:
                state_grid[i, j] = 'A'  # Agent's current position
            elif (i, j) == env.start:
                state_grid[i, j] = 'S'  # Agent's start position
            elif (i, j) == env.goal:
                state_grid[i, j] = 'G'  # Goal position
            elif env.grid[i, j] == -1:
                state_grid[i, j] = 'H'  # Hole
            else:
                state_grid[i, j] = '-'  # Empty tile
    
    return state_grid

def compute_bounds(Q, dynamics_table, rewards_table, gamma, prior_policy=None):
    S, A, _ = dynamics_table.shape
    beta = 5

    Q_flat = Q.flatten()
    baseline = (np.max(Q_flat) + np.min(Q_flat)) / 2
    Q_flat -= baseline
    exp_beta_Q = np.exp(beta * Q_flat)

    transition_dynamics = dynamics_table.reshape(S * A, S).T
    for i in range(transition_dynamics.shape[1]):
        col_sum = np.sum(transition_dynamics[:, i])
        if col_sum > 0:
            transition_dynamics[:, i] /= col_sum
    transition_dynamics_sparse = csr_matrix(transition_dynamics)

    if prior_policy is None:
        prior_policy = np.ones((S, A)) / A

    def pi_from_Q(Q, beta, prior_policy):
        V = (1 / beta) * np.log(np.sum(np.exp(beta * Q) * prior_policy, axis=1) + 1e-12)
        pi = prior_policy * np.exp(beta * (Q - V[:, None]))
        pi /= np.sum(pi, axis=1, keepdims=True)
        return pi

    policy = pi_from_Q(Q, beta, prior_policy)

    def get_mdp_generator(S, A, transition_dynamics_sparse, policy):
        rows, cols, data = [], [], []
        td = transition_dynamics_sparse.tocoo()
        for s_j, col, prob in zip(td.row, td.col, td.data):
            for a_j in range(A):
                row = s_j * A + a_j
                rows.append(row)
                cols.append(col)
                data.append(prob * policy[s_j, a_j])
        return csr_matrix((data, (rows, cols)), shape=(S * A, S * A))

    mdp_generator = get_mdp_generator(S, A, transition_dynamics_sparse, policy)
    Qj = np.log(mdp_generator.dot(exp_beta_Q) + 1e-12) / beta
    Qj = Qj.reshape(S, A)

    delta_rwd = rewards_table + gamma * Qj - Q
    # delta_min = np.min(delta_rwd)
    # delta_max = np.max(delta_rwd)
    finite_mask = np.isfinite(delta_rwd)
    delta_min = np.min(delta_rwd[finite_mask])
    delta_max = np.max(delta_rwd[finite_mask])

    lb = Q + delta_rwd + gamma * delta_min / (1 - gamma)
    ub = Q + delta_rwd + gamma * delta_max / (1 - gamma)

    r_min = np.min(rewards_table)
    r_max = np.max(rewards_table)
    lb = np.maximum(lb, r_min / (1 - gamma))
    ub = np.minimum(ub, r_max / (1 - gamma))

    return lb, ub

def clipped_q_learning(Qinit, env, T1, N_steps, test_steps, gamma=0.9, learning_rate=0.1, 
                      epsilon_initial=1.0, epsilon_decay=0.9998, epsilon_min=0.01,
                      bound_update_freq=100, ts=[]):
    """Clipped Q-learning implementation for FrozenLake"""
    total_states = env.size * env.size
    # ts = [state[0] * env.size + state[1] for state in env.holes + [env.goal]]
    
    # Initialize Q-table and model-free tracking
    Q = Qinit
    rewards_table = np.zeros((total_states, 4))
    dynamics_table = np.zeros((total_states, 4, total_states))
    
    epsilon = epsilon_initial
    episode_rewards = []
    state = 0
    
    # Initialize bounds
    L = np.full((total_states, 4), -np.inf)
    U = np.full((total_states, 4), np.inf)
    
    for step in range(1, N_steps + 1):
        if state in ts:
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            state = 0  # Reset to start
        
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q[state])
        
        next_state = np.random.choice(total_states, p=T1[state, action, :])
        reward = env.get_reward((next_state // env.size, next_state % env.size))
        
        # Update model-free tables
        rewards_table[state, action] = reward
        dynamics_table[state, action, next_state] += 1
        
        if step % bound_update_freq == 0:
            L, U = compute_bounds(Q, dynamics_table, rewards_table, gamma)

        # Standard Q-update
        target = reward + gamma * np.max(Q[next_state])
        new_q = Q[state, action] + learning_rate * (target - Q[state, action])
        
        # Clip to bounds if they exist
        if L[state, action] > -np.inf and U[state, action] < np.inf:
            Q[state, action] = np.clip(new_q, L[state, action], U[state, action])
        else:
            Q[state, action] = new_q
        
        state = next_state
        
        # Evaluation
        if step % test_steps == 0:
            avg_reward = evaluate_policy(env, Q, T1, ts, num_episodes=30)
            episode_rewards.append(avg_reward)
    
    return episode_rewards

def evaluate_policy(env, Q, T1, terminal_states, num_episodes=30, max_steps=20):
    """Evaluate the current policy"""
    total_rewards = []
    total_states = env.size * env.size
    
    for _ in range(num_episodes):
        state = 0
        episode_reward = 0
        step = 0
        
        while state not in terminal_states and step < max_steps:
            action = np.argmax(Q[state])
            next_state = np.random.choice(total_states, p=T1[state, action, :])
            episode_reward += env.get_reward((next_state // env.size, next_state % env.size))
            state = next_state
            step += 1
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)

def get_env(w_filename, t_filename):
    gridworld = np.load(w_filename)
    env = FrozenLake(size=6)
    env.grid = gridworld
    env.size = gridworld.shape[0]
    env.start = (0, 0)
    env.goal = (env.size - 1, 1)
    a=np.where(env.grid==-1)
    env.holes = list(zip(a[0],a[1]))
    holes = env.holes
    terminal_state = holes + [env.goal]
    T = np.load(t_filename)
    return env,T, terminal_state

# Main execution
def main():
    data = []
    for avg in range(init,till):
        print(avg)
        for nst in nst_range:
            env, T1, terminal_state = get_env(f"{10}//env_{nst}.npy", f"{10}//T_{nst}.npy")
            ts = [state[0] * env.size + state[1] for state in terminal_state]
            start_time = time.time()
            T1[ts, :, :] = 0
            Q1 = np.load(f'{avg}//Q1_{nst}.npy')
            
            # Run clipped Q-learning
            start_time = time.time()
            rewards = clipped_q_learning(Q1,
                env, T1, 
                N_steps=N_steps,  # 4800*5 from original
                test_steps=test_steps,
                gamma=gamma,
                ts=ts
            )
            end_time = time.time()
            
            
if __name__ == "__main__":
    main()