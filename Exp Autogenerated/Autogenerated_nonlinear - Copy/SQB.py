import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import time
import random
from scipy.sparse import csr_matrix
import argparse
import yaml

# Parse command-line argument
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

# Load YAML config
with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

# Unpack config variables
st = cfg["start"]
ed = cfg["end"]
nst_range = cfg["nst_range"]
x_percentage = cfg["x_percentage"]

gamma = cfg["gamma"]
learning_rate = cfg["learning_rate"]
epsilon_initial = cfg["epsilon_initial"]
epsilon_decay = cfg["epsilon_decay"]
epsilon_min = cfg["epsilon_min"]
max_steps = cfg["max_steps"]
N_steps = cfg["N_steps"]
test_steps = cfg["test_steps"]
bound_update_freq = cfg.get("bound_update_freq", 5000)
save_outputs = cfg.get("save_outputs", True)



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

def compute_bounds(Q, dynamics_table, rewards_table, gamma, prior_policy=None):
    S, A, _ = dynamics_table.shape
    beta = 5

    Q_flat = Q.flatten()
    baseline = (np.max(Q_flat) + np.min(Q_flat)) / 2
    Q_flat -= baseline
    # exp_beta_Q = np.exp(beta * Q_flat)
    Q_max = np.max(Q_flat)
    Q_stable = Q_flat - Q_max
    exp_beta_Q = np.exp(beta * Q_stable)


    transition_dynamics = dynamics_table.reshape(S * A, S).T
    for i in range(transition_dynamics.shape[1]):
        col_sum = np.sum(transition_dynamics[:, i])
        if col_sum > 0:
            transition_dynamics[:, i] /= col_sum
    transition_dynamics_sparse = csr_matrix(transition_dynamics)

    if prior_policy is None:
        prior_policy = np.ones((S, A)) / A

    # def pi_from_Q(Q, beta, prior_policy):
    #     V = (1 / beta) * np.log(np.sum(np.exp(beta * Q) * prior_policy, axis=1) + 1e-12)
    #     pi = prior_policy * np.exp(beta * (Q - V[:, None]))
    #     pi /= np.sum(pi, axis=1, keepdims=True)
    #     return pi
    def pi_from_Q(Q, beta, prior_policy):
        Q_max = np.max(Q, axis=1, keepdims=True)  # stabilize with max subtraction
        logits = beta * (Q - Q_max)
        unnormalized = prior_policy * np.exp(logits)
        pi = unnormalized / np.sum(unnormalized, axis=1, keepdims=True)
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
    # Qj = np.log(mdp_generator.dot(exp_beta_Q) + 1e-12) / beta
    Qj = (np.log(mdp_generator.dot(exp_beta_Q) + 1e-12) + beta * Q_max) / beta
    Qj = Qj.reshape(S, A)

    delta_rwd = rewards_table + gamma * Qj - Q
    delta_min = np.min(delta_rwd)
    delta_max = np.max(delta_rwd)

    lb = Q + delta_rwd + gamma * delta_min / (1 - gamma)
    ub = Q + delta_rwd + gamma * delta_max / (1 - gamma)

    r_min = np.min(rewards_table)
    r_max = np.max(rewards_table)
    lb = np.maximum(lb, r_min / (1 - gamma))
    ub = np.minimum(ub, r_max / (1 - gamma))

    return lb, ub


def epsilon_greedy_policy(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(A)
    else:
        return np.argmax(Q[state])

def test_q(e=30):
    global Q
    episode_rewards = []
    for episode in range(e):
        state = 0
        total_reward = 0
        step = 0
        while state not in terminal_state and step < max_steps:
            step += 1
            action = np.argmax(Q[state])
            next_state = np.random.choice(S, p=T1[state, action, :])
            reward = r[state, action, next_state]
            total_reward += reward
            state = next_state
        episode_rewards.append(total_reward)
    return np.mean(episode_rewards)

def clipped_q_learning(N_steps, test_steps):
    global Q
    epsilon = epsilon_initial
    episode_rewards = []
    state = 0
    step = 1
    
    # Initialize bounds
    L = np.full((S, A), -np.inf)
    U = np.full((S, A), np.inf)
    
    # Initialize model estimates
    rewards_table = np.zeros((S, A))
    dynamics_table = np.zeros((S, A, S))
    
    while step < N_steps + 1:
        if state in terminal_state:
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            state = 0
        
        action = epsilon_greedy_policy(state, epsilon)
        next_state = np.random.choice(S, p=T1[state, action, :])
        reward = r[state, action, next_state]
        
        # Update model estimates
        rewards_table[state, action] = reward
        dynamics_table[state, action, next_state] += 1
        
        if step % bound_update_freq == 0:
            L, U = compute_bounds(Q, dynamics_table, rewards_table, gamma)

        
        
        # Compute TD target
        current_q = Q[state, action]
        max_next_q = np.max(Q[next_state])
        td_target = reward + discount_factor * max_next_q
        
        # Compute new Q-value without clipping
        new_q = current_q + learning_rate * (td_target - current_q)
        
        # Clip the Q-value to stay within bounds if they exist
        if L[state, action] > -np.inf and U[state, action] < np.inf:
            Q[state, action] = np.clip(new_q, L[state, action], U[state, action])
        else:
            Q[state, action] = new_q
        
        state = next_state
        step += 1
        
        if step % test_steps == 0:
            tr = test_q()
            episode_rewards.append(tr)
    
    return episode_rewards

# Main experiment loop
data=[]

for avg in range(st,ed):
    S, A, R, R2, T, gamma,terminal_state = read_mdp(f"{avg}//mdp_exp_{'01'}.txt")
    for bf in nst_range:
        T1 = np.load(f"{avg}//T_{bf}.npy")
        gamma = 0.9
        x_percentage = 0
        number_stochastic_transitions = bf
        T1 = np.load(f"{avg}//T_{bf}.npy")
        T1[terminal_state, :, :] = 0
        r = (R + R2)**3
        
        start_time = time.time()
        
        # Q-learning parameters
        discount_factor = gamma
        
        num_episodes = int(N_steps / test_steps)
        
        # Run multiple episodes and average results
        num_runs = 1
        average_rewards = np.zeros(num_episodes)
        rewards_run = np.zeros((num_runs, num_episodes))
        
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
        Q1 = compute_q_values(S, A, R2, T1, gamma, terminal_state)
        
        for run in range(num_runs):
            Q = Q1 #np.zeros((S, A))
            episode_rewards = clipped_q_learning(N_steps, test_steps)
            rewards_run[run] = episode_rewards
            average_rewards += np.array(episode_rewards)
        
        average_rewards /= num_runs
        end_time = time.time()
        