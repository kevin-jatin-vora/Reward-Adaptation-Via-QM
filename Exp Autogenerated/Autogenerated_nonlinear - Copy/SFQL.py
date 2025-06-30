import os
import numpy as np
# np.random.seed(2)
from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
import time
os.chdir(os.getcwd())

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
    #mdp="mdp_new.txt"
    f = open(mdp)

    S = int(f.readline())
    A = int(f.readline())

    # Initialize Transition and Reward arrays
    R = np.zeros((S, A, S))
    R2 = np.zeros((S, A, S))
    T = np.zeros((S, A, S))

    # Update the Reward Function
    for s in range(S):
        for a in range(A):
            line = f.readline().split()
            for sPrime in range(S):
                R[s][a][sPrime] = line[sPrime]
                
     # Update the Reward Function
    for s in range(S):
        for a in range(A):
            line = f.readline().split()
            for sPrime in range(S):
                R2[s][a][sPrime] = line[sPrime]
    
    # Update the Transition Function
    for s in range(S):
        for a in range(A):
            line = f.readline().split()
            #print((s,a))
            for sPrime in range(S):
                #print(line[sPrime], end=" ")
                T[s][a][sPrime] = line[sPrime]
                 
            #print()

    # Read the value of gamma
    gamma = float(f.readline().rstrip())
    terminal_state=eval((f.readline().rstrip()))

    f.close()

    return S, A, R, R2, T, gamma,terminal_state



data=[]

for avg in range(st,ed):
    S, A, R, R2, T, gamma,terminal_state = read_mdp(f"{avg}//mdp_exp_{'01'}.txt")
    for bf in range(1,6,2):
        x_percentage = 0
        number_stochastic_transitions = bf
        T1 = np.load(f"{avg}//T_{bf}.npy")
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
        Q1 = compute_q_values(S, A, R, T1, gamma, terminal_state)
        
        # Compute Q-values
        Q2 = compute_q_values(S, A, R2, T1, gamma, terminal_state)
        
    ################################################################################

        
        r=(R+R2)**3
        
        def policy_evaluation(Q1, transition_probabilities, rewards, discount_factor=gamma, theta=1e-9):
            Q1_eval = np.copy(Q1)
            while True:
                delta = 0
                for state in range(Q1.shape[0]):
                    action = np.argmax(Q1[state])
                    next_state_probs = transition_probabilities[state, action]
                    next_state_rewards = rewards[state, action]
                    Q1_new = np.sum(next_state_probs * (next_state_rewards + discount_factor * np.max(Q1_eval, axis=1)))
                    
                    delta = max(delta, np.abs(Q1_eval[state, action] - Q1_new))
                    Q1_eval[state, action] = Q1_new
                if delta < theta:
                    break
            return Q1_eval
        Q1_e = policy_evaluation(Q1, T1, r)
        Q2_e = policy_evaluation(Q2, T1, r)
        
        # Initialize combined Q-table
        combined_Q = np.zeros((S, A))
        
        # Iterate over each state-action pair
        for s in range(S):
            for a in range(A):
                # Find maximum Q-value across both Q-tables for state s and action a
                max_Q_value = max(Q1_e[s, a], Q2_e[s, a])
                
                # Assign the maximum Q-value to the combined Q-table
                combined_Q[s, a] = max_Q_value
        
        # Print combined Q-table
        # print("Combined Q-table:")
        # print(combined_Q)
        
        start_time = time.time()
        def epsilon_greedy_policy(state, epsilon):
            if np.random.rand() < epsilon:
                return np.random.choice(A)
            else:
                return np.argmax(Q[state])
        
        def test_q(e=30):
            global Q
            episode_rewards=[]
            for episode in range(e):
                state = 0
                total_reward = 0
                step = 0
                while state not in terminal_state and step<max_steps:
                    step+=1
                    action = np.argmax(Q[state])
                    next_state = np.random.choice(S, p=T1[state, action, :])
                    reward = r[state, action, next_state]
                    total_reward += reward
                    state = next_state
                episode_rewards.append(total_reward)
            return np.mean(episode_rewards) 
        
        def q_learning(N_steps, test_steps):
            global Q
            epsilon = epsilon_initial
            episode_rewards = []
            state = 0
            step = 1
        
            while step < N_steps+1:
                if state in terminal_state:
                    # Decay epsilon
                    epsilon = max(epsilon * epsilon_decay, epsilon_min)
                    # Reset to the initial state if the agent reaches a terminal state
                    state = 0
        
                action = epsilon_greedy_policy(state, epsilon)
                next_state = np.random.choice(S, p=T1[state, action, :])
                reward = r[state, action, next_state]
                # Update Q-value
                Q[state, action] += learning_rate * (
                    reward + discount_factor * np.max(Q[next_state]) - Q[state, action]
                )
        
                state = next_state
                step += 1
            
                if step % test_steps == 0:
                    tr = test_q()
                    episode_rewards.append(tr)
            return episode_rewards
        
        
        
        # Define Q-learning parameters
        discount_factor = gamma
        
        num_episodes = int(N_steps/test_steps)
        
        # Run multiple episodes and average results
        num_runs = 1
        average_rewards = np.zeros(num_episodes)
        rewards_run = np.zeros((num_runs, num_episodes))
        for run in range(num_runs):
            Q= combined_Q.copy()
            # np.random.seed(run)
            episode_rewards = q_learning(N_steps,test_steps)
            rewards_run[run] = episode_rewards
            average_rewards += np.array(episode_rewards)
        average_rewards /= num_runs
        end_time = time.time()
        