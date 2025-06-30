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

import numpy as np
import matplotlib.pyplot as plt
import random
#np.random.seed(6)
import pandas as pd
import time
import os

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
    #terminal_state=int((f.readline().rstrip()))

    f.close()

    return S, A, R, R2, T, gamma#,terminal_state

S, A, R, R2, T, gamma = read_mdp("gridworld_mdp(r1+r2).txt")
R[31,1,40] = 0.6
R2[31,1,40] = 0.6
R[41,2,40] = 0.6
R2[41,2,40] = 0.6
R[39,3,40] = 0.6
R2[39,3,40] = 0.6


def make_transitions_stochastic(T1, x_percentage):
    S, A, _ = T1.shape
    
    # Calculate the number of states to remain deterministic
    num_deterministic_states = int(S * x_percentage/100)
    
    # Randomly select which states will remain deterministic
    deterministic_states = np.random.choice(S, num_deterministic_states, replace=False)
    
    # Initialize modified transition probability matrix
    T_stochastic = np.zeros_like(T1)
    
    # Modify transition probabilities for deterministic states
    for s in deterministic_states:
        for a in range(A):
            max_prob_index = np.argmax(T1[s, a])
            T_stochastic[s, a, max_prob_index] = 1
     
    # Modify transition probabilities for stochastic states
    stochastic_states = np.setdiff1d(np.arange(S), deterministic_states)
    for s in stochastic_states:
        #print(s)
        for a in range(A):
            T_stochastic[s,a] = T1[s,a]
            # max_prob_index = np.argmax(T1[s, a])
            # probabilities = np.random.dirichlet(np.ones(S))
            # T_stochastic[s, a] = probabilities
            # if np.argmax(T_stochastic[s, a]) != max_prob_index:
            #     T_stochastic[s, a, max_prob_index] = np.max(T_stochastic[s, a])
            #     T_stochastic[s, a, np.argmax(T_stochastic[s, a])] = T_stochastic[s, a, max_prob_index]
    return T_stochastic



def restrict_transition(matrix, max_bf):
    S, A, S_prime = matrix.shape

    # Sample n uniformly from [1, max_bf] for each state
    n_values = np.random.randint(1, max_bf + 1, size=S)
    print(n_values.mean())
    # Identify top n states for each action
    # top_n_indices = np.argsort(matrix, axis=2)[:, :, -n_values[:, None]]
    sorted_matrix = np.argsort(matrix, axis=2)

    # Create a mask to zero out probabilities for states not in top n
    mask = np.zeros_like(matrix)
    for s in range(S):
        for a in range(A):
            mask[s, a, sorted_matrix[s, a, -n_values[s]:]] = 1

    # Apply mask
    restricted_matrix = matrix * mask

    # Normalize probabilities for the top n states for each action
    row_sums = restricted_matrix.sum(axis=2, keepdims=True)
    normalized_matrix = restricted_matrix / row_sums

    return normalized_matrix


data=[]
for avg in range(init, last):
    print(avg)
    for nst in nst_range:
        number_stochastic_transitions = nst
        
        # T1 = make_transitions_stochastic(T, x_percentage)
        # number_stochastic_transitions = nst #A
        # T1 = restrict_transition(T1, number_stochastic_transitions)
        # T1[terminal_state,:, :] = 0
        # filename = f'{avg}//T_{nst}.npy'
        # np.save(filename, T1)
        
        filename = f'{avg}//T_{nst}.npy'
        T1=np.load(filename)
        start_time = time.time()
        # Define environment parameters
        num_rows = 5
        num_cols = 9
        num_states = num_rows * num_cols
        num_actions = 4
        start_state = (0,4)
        goal_states = [(0, 0), (0, 8), (4, 4)]
        
        # Define rewards
        rewards = np.ones((num_rows, num_cols))*-0.0001
        rewards[0, 0] = 1.0
        rewards[0, 8] = 1.0
        rewards[4, 4] = 1.2
        
        # Define actions: down, up, left, right
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Initialize Q-table
        Q = np.zeros((num_states, num_actions))
        
        # Convert state from (row, col) to index
        def state_to_index(state):
            row, col = state
            return row * num_cols + col
        
        # Convert index to state (row, col)
        def index_to_state(index):
            row = index // num_cols
            col = index % num_cols
            return row, col
        
        # Epsilon-greedy policy
        def epsilon_greedy_policy(state, epsilon):
            if np.random.rand() < epsilon:
                return np.random.choice(num_actions)
            else:
                # Get indices of maximum Q-values
                best_actions = np.where(Q[state_to_index(state)] == np.max(Q[state_to_index(state)]))[0]
                # Randomly select one of the best actions
                return np.random.choice(best_actions)
                # return np.argmax(Q[state_to_index(state)])
        
       
        def test_q(e=30):
            global Q
            episode_rewards = []
            for episode in range(e):
                state = start_state
                total_reward = 0
                step = 0
                while state not in goal_states and step<max_steps:
                    step+=1
                    action = np.argmax(Q[state_to_index(state)])
                    next_state_index = np.random.choice(num_states, p=T1[state_to_index(state), action, :])
                    next_state = index_to_state(next_state_index)
                    reward = rewards[next_state[0], next_state[1]]
                    total_reward += reward
                    state = next_state
                episode_rewards.append(total_reward)
                
            return np.mean(episode_rewards)
        
        def q_learning(N_steps, test_steps):
            global Q
            epsilon = epsilon_initial
            episode_rewards = []
            state = start_state
            step = 1
        
            while step < N_steps+1:
                if state in goal_states:
                    # Decay epsilon
                    epsilon = max(epsilon * epsilon_decay, epsilon_min)
            
                    # Reset to the initial state if the agent reaches a terminal state
                    state = start_state
        
                action = epsilon_greedy_policy(state, epsilon)
                next_state_index = np.random.choice(num_states, p=T1[state_to_index(state), action, :])
                next_state = index_to_state(next_state_index)
                reward = rewards[next_state[0], next_state[1]]
        
                # Update Q-value
                Q[state_to_index(state), action] += learning_rate * (
                    reward + discount_factor * np.max(Q[state_to_index(next_state)]) - Q[state_to_index(state), action]
                )
        
                state = next_state
                step += 1
        
                # Track reward for each step
                # episode_rewards.append(reward)
        
                # Decay epsilon
                # epsilon = max(epsilon * epsilon_decay, epsilon_min)
        
                # Optional: Print or test the Q-values periodically
                if step % test_steps == 0:
                    tr = test_q()
                    # for k in tr:
                    episode_rewards.append(tr)
                    
            return episode_rewards
    
        # Run multiple episodes and average results
        num_runs = 1
        num_episodes = int(N_steps/test_steps)
        average_rewards = np.zeros(num_episodes)
        rewards_run = np.zeros((num_runs,num_episodes))
        for run in range(num_runs):
            Q=np.zeros((S,A))
            # np.random.seed(run) #seed=1 on top
            episode_rewards = q_learning(N_steps,test_steps)
            rewards_run[run] = episode_rewards
            average_rewards += np.array(episode_rewards)
        average_rewards /= num_runs
        end_time = time.time()
        