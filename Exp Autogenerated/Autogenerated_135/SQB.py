import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
import time
import os
import shutil
import runpy
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

# Load configuration
st = cfg["start"]
ed = cfg["end"]
nst_range = cfg["nst_range"]

gamma = cfg["gamma"]
learning_rate = cfg["learning_rate"]
epsilon_initial = cfg["epsilon_initial"]
epsilon_decay = cfg["epsilon_decay"]
epsilon_min = cfg["epsilon_min"]
max_steps = cfg["max_steps"]
N_steps = cfg["N_steps"]
test_steps = cfg["test_steps"]

x_percentage = cfg["x_percentage"]
bound_update_freq = cfg.get("bound_update_freq", 5000)


def read_mdp(mdp):
    """Function to read MDP file"""
    f = open(mdp)
    S = int(f.readline())
    A = int(f.readline())

    R = np.zeros((S, A, S))
    R2 = np.zeros((S, A, S))
    T = np.zeros((S, A, S))

    for s in range(S):
        for a in range(A):
            line = f.readline().split()
            for sPrime in range(S):
                R[s][a][sPrime] = line[sPrime]
                
    for s in range(S):
        for a in range(A):
            line = f.readline().split()
            for sPrime in range(S):
                R2[s][a][sPrime] = line[sPrime]
    
    for s in range(S):
        for a in range(A):
            line = f.readline().split()
            for sPrime in range(S):
                T[s][a][sPrime] = line[sPrime]

    gamma = float(f.readline().rstrip())
    terminal_state = eval((f.readline().rstrip()))
    f.close()

    return S, A, R, R2, T, gamma, terminal_state

def make_transitions_stochastic(T1, x_percentage):
    S, A, _ = T1.shape
    num_deterministic_states = int(S * x_percentage/100)
    deterministic_states = np.random.choice(S, num_deterministic_states, replace=False)
    T_stochastic = np.zeros_like(T1)
    
    for s in deterministic_states:
        for a in range(A):
            max_prob_index = np.argmax(T1[s, a])
            T_stochastic[s, a, max_prob_index] = 1
     
    stochastic_states = np.setdiff1d(np.arange(S), deterministic_states)
    for s in stochastic_states:
        for a in range(A):
            T_stochastic[s,a] = T1[s,a]
    return T_stochastic

def restrict_transition(matrix, max_bf):
    S, A, S_prime = matrix.shape
    n_values = np.random.randint(1, max_bf + 1, size=S)
    sorted_matrix = np.argsort(matrix, axis=2)
    mask = np.zeros_like(matrix)
    
    for s in range(S):
        for a in range(A):
            mask[s, a, sorted_matrix[s, a, -n_values[s]:]] = 1

    restricted_matrix = matrix * mask
    row_sums = restricted_matrix.sum(axis=2, keepdims=True)
    normalized_matrix = restricted_matrix / row_sums
    return normalized_matrix

def compute_bounds(Q, dynamics_table, rewards_table, gamma, prior_policy=None):
    S, A, _ = dynamics_table.shape
    beta = 5

    # Flatten Q for vectorized softmax computation
    Q_flat = Q.flatten()
    baseline = (np.max(Q_flat) + np.min(Q_flat)) / 2
    Q_flat -= baseline
    exp_beta_Q = np.exp(beta * Q_flat)

    # Normalize dynamics
    transition_dynamics = dynamics_table.reshape(S * A, S).T
    for i in range(transition_dynamics.shape[1]):
        col_sum = transition_dynamics[:, i].sum()
        if col_sum > 0:
            transition_dynamics[:, i] /= col_sum
    transition_dynamics_sparse = csr_matrix(transition_dynamics)

    # Default prior: uniform
    if prior_policy is None:
        prior_policy = np.ones((S, A)) / A

    # π from Q
    def pi_from_Q(Q, beta, prior_policy):
        V = (1 / beta) * np.log(np.sum(np.exp(beta * Q) * prior_policy, axis=1) + 1e-12)
        pi = prior_policy * np.exp(beta * (Q - V[:, None]))
        pi /= np.sum(pi, axis=1, keepdims=True)
        return pi

    policy = pi_from_Q(Q, beta, prior_policy)

    # Generator matrix
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

    # Delta and bounds
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


def clipped_q_learning(S, A, T1, r, terminal_state, Q_init, gamma=0.9, 
                      learning_rate=0.1, epsilon_initial=1.0, 
                      epsilon_decay=0.997, epsilon_min=0.01,
                      N_steps=20000, test_steps=4, max_steps=None,
                      bound_update_freq=5000):
    """Clipped Q-learning implementation"""
    if max_steps is None:
        max_steps = S
        
    # Initialize Q-table and model-free tracking
    Q = Q_init #np.zeros((S, A))
    rewards_table = np.zeros((S, A))
    dynamics_table = np.zeros((S, A, S))
    
    # Initialize bounds
    L = np.full((S, A), -np.inf)
    U = np.full((S, A), np.inf)
    
    # Get min/max rewards from environment
    min_reward = np.min(r)
    max_reward = np.max(r)
    
    epsilon = epsilon_initial
    episode_rewards = []
    state = 0
    
    for step in range(1, N_steps + 1):
        if state in terminal_state:
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            state = 0  # Reset to start
        
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.choice(A)
        else:
            action = np.argmax(Q[state])
        
        next_state = np.random.choice(S, p=T1[state, action, :])
        reward = r[state, action, next_state]
        
        # Update model-free tables
        rewards_table[state, action] = reward
        dynamics_table[state, action, next_state] += 1
        
        # Update bounds periodically
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
            avg_reward = evaluate_policy(Q, T1, r, terminal_state, max_steps)
            episode_rewards.append(avg_reward)
    
    return episode_rewards

def evaluate_policy(Q, T1, r, terminal_state, max_steps, num_episodes=30):
    """Evaluate the current policy"""
    total_rewards = []
    S = Q.shape[0]
    
    for _ in range(num_episodes):
        state = 0
        episode_reward = 0
        step = 0
        
        while state not in terminal_state and step < max_steps:
            action = np.argmax(Q[state])
            next_state = np.random.choice(S, p=T1[state, action, :])
            episode_reward += r[state, action, next_state]
            state = next_state
            step += 1
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)

def main():
    data = []
    
    for avg in range(st, ed):
        # Read MDP parameters
        S, A, R, R2, T, gamma, terminal_state = read_mdp(f"{avg}//mdp_exp_{'01'}.txt")
        r = R + R2
        
        for bf in range(1, 6, 2):
            # Load transition matrix
            T1 = np.load(f"{avg}//T_{bf}.npy")
            T1[terminal_state, :, :] = 0
            def compute_q_values(S, A, R, T, gamma, terminal_state):
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
            Q1 = np.load(f"{avg}//Q1_{bf}.npy")#compute_q_values(S, A, R, T1, gamma, terminal_state)
            
            # Run clipped Q-learning
            start_time = time.time()
            rewards = clipped_q_learning(
                S, A, T1, r, terminal_state, Q_init=Q1,
                gamma=gamma,
                N_steps=N_steps,
                test_steps=test_steps,
                max_steps=S
            )
            end_time = time.time()
            
            
if __name__ == "__main__":
    main()