import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

init = cfg['start']
last = cfg['end']
gamma = cfg['gamma']
discount_factor = gamma
terminal_state = cfg['terminal_state']
x_percentage = cfg['x_percentage']
nst_range = cfg['nst_range']
N_steps = cfg.get('N_steps', 18000)
max_steps = cfg.get('max_steps', 30)
test_steps = cfg.get('test_steps', 12)
# Q-learning params
learning_rate = cfg.get('learning_rate', 0.1)
epsilon_initial = cfg.get('epsilon_initial', 1.0)
epsilon_decay = cfg.get('epsilon_decay', 0.999)
epsilon_min = cfg.get('epsilon_min', 0.01)


from scipy import stats
import random
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import time
from scipy.sparse import csr_matrix

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
    f.close()

    return S, A, R, R2, T, gamma

S, A, R, R2, T, gamma = read_mdp("gridworld_mdp(r1+r2).txt")
R[31,1,40] = 0.6
R2[31,1,40] = 0.6
R[41,2,40] = 0.6
R2[41,2,40] = 0.6
R[39,3,40] = 0.6
R2[39,3,40] = 0.6

def compute_state_value(Q, state_idx):
    """Compute V(s) = max_a Q(s,a)"""
    return np.max(Q[state_idx])

data = []
for avg in range(init, last):
    print(avg)
    for nst in nst_range:
        number_stochastic_transitions = nst
        
        filename = f'{avg}//T_{nst}.npy'
        T1 = np.load(filename)
    
        start_time = time.time()
        num_rows = 5
        num_cols = 9
        num_states = num_rows * num_cols
        num_actions = 4
        start_state = (0,4)
        goal_states = [(0, 0), (0, 8), (4, 4)]
        
        rewards = np.ones((num_rows, num_cols)) * -0.0001
        rewards[0, 0] = 1.0
        rewards[0, 8] = 1.0
        rewards[4, 4] = 1.2
        
        learning_rate = 0.1
        discount_factor = gamma
        epsilon_initial = 1
        epsilon_decay = 0.999
        epsilon_min = 0.01
        max_steps = 30
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        batch_size = 32  # Size of batch for computing bounds
        
        def state_to_index(state):
            row, col = state
            return row * num_cols + col
        
        def index_to_state(index):
            row = index // num_cols
            col = index % num_cols
            return row, col
        
        def epsilon_greedy_policy(state, epsilon):
            if np.random.rand() < epsilon:
                return np.random.choice(num_actions)
            else:
                best_actions = np.where(Q[state_to_index(state)] == np.max(Q[state_to_index(state)]))[0]
                return np.random.choice(best_actions)
        
        def test_q(e=30):
            episode_rewards = []
            for episode in range(e):
                state = start_state
                total_reward = 0
                step = 0
                while state not in goal_states and step < max_steps:
                    step += 1
                    action = np.argmax(Q[state_to_index(state)])
                    next_state_index = np.random.choice(num_states, p=T1[state_to_index(state), action, :])
                    next_state = index_to_state(next_state_index)
                    reward = rewards[next_state[0], next_state[1]]
                    total_reward += reward
                    state = next_state
                episode_rewards.append(total_reward)
            return np.mean(episode_rewards)
        
        def clipped_q_learning(N_steps, test_steps):
            global Q
            epsilon = epsilon_initial
            episode_rewards = []
            state = start_state
            step = 1
        
            # Initialize bounds
            L = np.full((num_states, num_actions), -np.inf)
            U = np.full((num_states, num_actions), np.inf)
            bound_update_freq = 5000
        
            # Learnable model (from data)
            rewards_table = np.zeros((num_states, num_actions))
            dynamics_table = np.zeros((num_states, num_actions, num_states))
                    
            def compute_bounds(Q, rewards_table, dynamics_table, gamma, terminal_states, prior_policy=None):
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
                        if s in terminal_states:
                            delta_rwd[s, a] = 0
                        else:
                            delta_rwd[s, a] = rewards_table[s, a] + gamma * Qj[s, a] - Q[s, a]
                        # delta_rwd[s, a] = rewards_table[s, a] + gamma * Qj[s, a] - Q[s, a]
                        
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

            while step < N_steps + 1:
                if state in goal_states:
                    epsilon = max(epsilon * epsilon_decay, epsilon_min)
                    state = start_state
                    continue  # skip update at terminal state
        
                action = epsilon_greedy_policy(state, epsilon)
                next_state_index = np.random.choice(num_states, p=T1[state_to_index(state), action])
                next_state = index_to_state(next_state_index)
                reward = rewards[next_state[0], next_state[1]]
        
                state_idx = state_to_index(state)
        
                # Update empirical model
                rewards_table[state_idx, action] = reward
                dynamics_table[state_idx, action, next_state_index] += 1
        
                # Update bounds periodically
                if step % bound_update_freq == 0 and step > 5500:
                    L, U = compute_bounds(Q, rewards_table, dynamics_table, gamma, terminal_state)
        
                # Q-learning update
                current_q = Q[state_idx, action]
                max_next_q = np.max(Q[next_state_index])
                td_target = reward + gamma * max_next_q
                new_q = current_q + learning_rate * (td_target - current_q)
        
                # Clip Q-value within bounds
                if L[state_idx, action] > -np.inf and U[state_idx, action] < np.inf and (td_target - current_q)>0.1:
                    Q[state_idx, action] = np.clip(new_q, L[state_idx, action], U[state_idx, action])
                else:
                    Q[state_idx, action] = new_q
        
                state = next_state
                step += 1
        
                if step % test_steps == 0:
                    tr = test_q()
                    episode_rewards.append(tr)
        
            return episode_rewards

        num_runs = 1
        N_steps = 18000
        test_steps = 12
        num_episodes = int(N_steps/test_steps)
        average_rewards = np.zeros(num_episodes)
        rewards_run = np.zeros((num_runs, num_episodes))
        
        def compute_q_values(S, A, R, T, gamma, terminal_state):
            # Initialize Q-values to zeros
            Q_new = np.zeros((S, A))
            
            # Maximum number of iterations for value iteration
            max_iterations = 5500
            
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
        
        q_p1 = compute_q_values(S,A,R,T1,gamma, terminal_state)
        for run in range(num_runs):
            Q = q_p1 #np.zeros((num_states, num_actions))#np.random.uniform(-1,1,size=(S,A))#q_p1+q_p2#np.zeros((num_states, num_actions))
            episode_rewards = clipped_q_learning(N_steps, test_steps)
            rewards_run[run] = episode_rewards
            average_rewards += np.array(episode_rewards)
            
        average_rewards /= num_runs
        end_time = time.time()
        