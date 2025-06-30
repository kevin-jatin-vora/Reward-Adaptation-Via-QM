import numpy as np
from scipy.sparse import csr_matrix
import time
import pandas as pd
import argparse
import yaml

# --- Config loading ---
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

# --- Common Config Variables ---
init = cfg['start']
last = cfg['end']
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

# Environment setup (unchanged)
WORLD = np.array([
    ["G", "_", "_", "_", "_", "X", "X"],
    ["G", "_", "_", "_", "_", "_", "_"],
    ["X", "X", "_", "_", "_", "_", "_"],
    ["X", "X", "X", "X", "X", "_", "_"],
    ["X", "X", "_", "_", "_", "_", "_"],
    ["_", "_", "_", "_", "_", "_", "_"],
    ["S", "_", "_", "_", "X", "X", "X"]
])

S = WORLD.size
A = 7
STATES = range(WORLD.size)
STATE2WORLD = {i: (i//7, i%7) for i in range(49)}
START = 42
GOALS = [0, 7]
WALLS = [5,6,14,15,21,22,23,24,25,28,29,46,47,48]

# Reward definitions (unchanged)
CRASH = -10.
WIN = 100.
STEP = -1.

ACTIONS = [
    (0, 0), 
    (-1, 0), (-2, 0),
    (0, 1), (0, 2),
    (0, -1), (0, -2),
]

# Initialize reward tensors (unchanged)
R = np.zeros((S,A,S))
R2 = np.zeros((S,A,S))
R3 = np.zeros((S,A,S))
for s in range(S):
    for a in range(A):
        for sdash in range(S):
            if sdash in GOALS:
                R3[s,a,sdash] = 2
            if sdash not in WALLS and sdash not in GOALS:
                R2[s,a,sdash] = 0.2
                R3[s,a,sdash] = -0.3
            if sdash in WALLS:
                R2[s,a,sdash] = -0.5
                R3[s,a,sdash] = 0.3
            if sdash == 42:
                R[s,a,sdash] = 3
                R3[s,a,sdash] = -4
r = R + R2 + R3

# Transition function (unchanged)
def transition(state, action):
    dx, dy = ACTIONS[action]
    next_state = (STATE2WORLD[state][0] + dx, STATE2WORLD[state][1] + dy)
    state_idx = next_state[0]*7 + next_state[1]
    if (state_idx in WALLS or 
        next_state[0] < 0 or next_state[0] >= 7 or 
        next_state[1] < 0 or next_state[1] >= 7):
        return state_idx, False
    return state_idx, True

def neighbours(s):
    n = []
    for a in range(7):
        ns = transition(s, a)
        if ns[1]:
            n.append(ns[0])
        else:
            if ns[0] in STATE2WORLD:
                n.append(ns[0])
            else:
                n.append(s)
    return list(set(n))

# Initialize transition matrix (unchanged)
T = np.zeros((49,7,49))
for s in range(49):
    if s in WALLS or s in GOALS:
        continue
    reachable = neighbours(s)
    ns = []
    for i in reachable:
        if i in STATE2WORLD:
            ns.append(i)
        else:
            ns.append(s)
    for a in range(7):
        nxt_s = list(transition(s, a))
        if nxt_s[0] not in STATE2WORLD:
            nxt_s[0] = s
        for sdash in range(49):
            if nxt_s[0] == sdash:
                T[s,a,sdash] = 0.88
            elif sdash in ns:
                T[s,a,sdash] = 0.12/(len(ns)-1)
            else:
                T[s,a,sdash] = 0
        if np.sum(T[s,a]) != 1:
            T[s,a,nxt_s[0]] += 1 - np.sum(T[s,a])

# New clipped Q-learning functions
def compute_bounds(Q, dynamics_table, rewards_table, gamma, prior_policy=None):
    S, A, _ = dynamics_table.shape
    beta = 5
    
    # Flatten Q
    Q_flat = Q.flatten()
    baseline = (np.max(Q_flat) + np.min(Q_flat)) / 2
    Q_flat -= baseline
    
    # Create transition matrix: shape (S, S*A)
    transition_dynamics = dynamics_table.reshape(S * A, S).T
    for i in range(transition_dynamics.shape[1]):
        col_sum = np.sum(transition_dynamics[:, i])
        if col_sum > 0:
            transition_dynamics[:, i] /= col_sum
    transition_dynamics_sparse = csr_matrix(transition_dynamics)
    
    # Default to uniform prior if not provided
    if prior_policy is None:
        prior_policy = np.ones((S, A)) / A
    
    # Compute policy π from Q (soft policy)
    def pi_from_Q(Q, beta, prior_policy):
        V = (1 / beta) * np.log(np.sum(np.exp(beta * Q) * prior_policy, axis=1))
        pi = prior_policy * np.exp(beta * (Q - V.reshape(-1, 1)))
        pi /= np.sum(pi, axis=1, keepdims=True)
        return pi
    
    policy = pi_from_Q(Q, beta, prior_policy)  # shape (S, A)
    
    # Build MDP generator: (S*A, S*A)
    def get_mdp_generator(S, A, transition_dynamics_sparse, policy):
        rows, cols, data = [], [], []
        td = transition_dynamics_sparse.tocoo()
        for s_j, col, prob in zip(td.row, td.col, td.data):
            for a_j in range(A):
                row = s_j * A + a_j
                rows.append(row)
                cols.append(col)
                data.append(prob * policy[s_j, a_j])
        shape = (S * A, S * A)
        return csr_matrix((data, (rows, cols)), shape=shape)
    
    mdp_generator = get_mdp_generator(S, A, transition_dynamics_sparse, policy)
    
    # Compute soft Bellman backup
    exp_beta_Q = np.exp(beta * Q_flat)
    Qj = np.log(mdp_generator.dot(exp_beta_Q) + 1e-12) / beta  # shape (S*A,)
    Qj = Qj.reshape(S, A)
    
    # Compute delta_rwd = r + γ Qj - Q
    delta_rwd = np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            # if s in terminal_states:
            #     delta_rwd[s, a] = 0
            # else:
            #     delta_rwd[s, a] = rewards_table[s, a] + gamma * Qj[s, a] - Q[s, a]
            delta_rwd[s, a] = rewards_table[s, a] + gamma * Qj[s, a] - Q[s, a]
            
    delta_min = np.min(delta_rwd)
    delta_max = np.max(delta_rwd)
    
    lb = Q + delta_rwd + gamma * delta_min / (1 - gamma)
    ub = Q + delta_rwd + gamma * delta_max / (1 - gamma)
    
    # Theoretical clipping
    r_min = np.min(rewards_table)
    r_max = np.max(rewards_table)
    lb = np.maximum(lb, r_min / (1 - gamma))
    ub = np.minimum(ub, r_max / (1 - gamma))
    
    return lb, ub
    

def clipped_q_learning(N_steps, test_steps, T1, gamma):
    global Q

    rewards_table = np.zeros((S, A))
    dynamics_table = np.zeros((S, A, S))

    epsilon = epsilon_initial
    episode_rewards = []
    state = START

    L = np.full((S, A), -np.inf)
    U = np.full((S, A), np.inf)

    for step in range(1, N_steps + 1):
        if state in WALLS or state in GOALS:
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            state = START

        action = epsilon_greedy_policy(state, epsilon)
        next_state = np.random.choice(S, p=T1[state, action, :])
        reward = r[state, action, next_state]

        rewards_table[state, action] = reward
        dynamics_table[state, action, next_state] += 1

        if step % bound_update_freq == 0 and step > 5000:
            L, U = compute_bounds(Q, dynamics_table, rewards_table, gamma)

        target = reward + gamma * np.max(Q[next_state])
        td_error = target - Q[state, action]
        new_q = Q[state, action] + learning_rate * td_error

        if L[state, action] > -np.inf and U[state, action] < np.inf:
            Q[state, action] = np.clip(new_q, L[state, action], U[state, action])
        else:
            Q[state, action] = new_q

        state = next_state

        if step % test_steps == 0:
            episode_rewards.append(test_q())

    return episode_rewards

# Helper functions (unchanged)
def epsilon_greedy_policy(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(A)
    return np.argmax(Q[state])

def test_q(e=30):
    episode_rewards = []
    for _ in range(e):
        state = START
        total_reward = 0
        step = 0
        while state not in WALLS and state not in GOALS and step < max_steps:
            step += 1
            action = np.argmax(Q[state])
            next_state = np.random.choice(S, p=T1[state, action, :])
            total_reward += r[state, action, next_state]
            state = next_state
        episode_rewards.append(total_reward)
    return np.mean(episode_rewards)

# Parameters (unchanged)
discount_factor = gamma
bound_update_freq = 100  # How often to update bounds

# Main execution
data = []
data=[]
for avg in range(init,last):
    for nst in nst_range:
        x_percentage = 0
        number_stochastic_transitions = nst #A
        gamma=gamma
        filename = f'{avg}//T_{nst}.npy'
        T1 = np.load(filename)
        
        def compute_q_values(S, A, R, T, gamma):
            # Initialize Q-values to zeros
            Q_new = np.zeros((S, A))
            
            # Maximum number of iterations for value iteration
            max_iterations = 5000
            
            # Value iteration
            for _ in range(max_iterations):
                Q = Q_new.copy()
                for s in range(S):
                    for a in range(A):
                        q_sa = 0
                        for s_prime in range(S):
                            q_sa += T[s][a][s_prime] * (R[s][a][s_prime] + gamma * np.max(Q[s_prime]))
                        Q_new[s][a] = q_sa
                if np.max(np.abs(Q - Q_new)) < 1e-12:  # Check for convergence
                    print("Converged in", _ + 1, "iterations")
                    break
                Q = Q_new
            
            return Q
        
        # Compute Q-values
        Q1 = compute_q_values(S, A, R, T1, discount_factor)
        
        start_time = time.time()
        Q = Q1 #np.ones((S, A)) * -0.5  # Initial Q-values
        
        N_steps = 28000
        test_steps = 4
        
        # Run clipped Q-learning
        rewards_run = np.zeros((1, N_steps // test_steps))
        rewards_run[0] = clipped_q_learning(N_steps, test_steps, T1, discount_factor)
        
        end_time = time.time()
