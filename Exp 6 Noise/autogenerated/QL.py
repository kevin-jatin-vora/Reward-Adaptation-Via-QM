import subprocess
import pandas as pd
# Start measuring the execution time
import os
os.chdir(os.getcwd())
import numpy as np
# np.random.seed(2)
from matplotlib import pyplot as plt
import time
import shutil
import runpy


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
        # print(s)
        for a in range(A):
            T_stochastic[s,a] = T1[s,a]
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
# st = int(input("start: "))
# ed = int(input("end: "))
import os

def read_and_increment(filename):
    try:
        # Read the current values from the file
        with open(filename, 'r') as f:
            lines = f.readlines()
            st = int(lines[0].strip())
            ed = int(lines[1].strip())

        # Increment the values
        st += 2
        ed += 2

        # Write the incremented values back to the file
        with open(filename, 'w') as f:
            f.write(str(st) + '\n')
            f.write(str(ed) + '\n')

        return st, ed
    except FileNotFoundError:
        print("File not found. Creating a new file with default values.")
        with open(filename, 'w') as f:
            f.write('0\n')
            f.write('2\n')
        return 0, 2

# Example usage
filename = "QL_idx.txt"
st, ed = read_and_increment(filename)
print("New values:", st, ed)
for avg in range(st,ed):
    if(not os.path.isdir(str(avg))):
        os.mkdir(str(avg))
    runpy.run_path("Generator.py")
    shutil.move(f"mdp_exp_{'01'}.txt",f"{avg}//mdp_exp_{'01'}.txt")
    S, A, R, R2, T, gamma,terminal_state = read_mdp(f"{avg}//mdp_exp_{'01'}.txt")
    # np.random.seed(avg)  
    # input()
    # for noise in range(1,6,2):
    bf=2
    gamma = 0.89
    x_percentage = 0
    T1 = make_transitions_stochastic(T, x_percentage)
    number_stochastic_transitions = bf
    T1 = restrict_transition(T1, number_stochastic_transitions)
    T1[terminal_state,:, :] = 0
    np.save(f"{avg}//T_{bf}.npy", T1)
    np.save(f"{avg}//r1.npy", R)
    np.save(f"{avg}//r2.npy", R2)
    np.save(f"{avg}//terminal_state.npy",terminal_state)
    for noise in [0,0.25,0.5]:
        # np.random.seed(avg)
        # T1=np.load(f"{avg}//T_{bf}.npy")
        # r=R+R2
        r=R + R2 + np.random.uniform(-noise,noise,size=R.shape)
        np.save(f"{avg}//r_target_{noise}.npy", r)
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
            return np.mean(episode_rewards)#total_reward 
        
        def q_learning(N_steps, test_steps):
            global Q
            epsilon = epsilon_initial
            episode_rewards = []
            state = 0
            step = 1
            e_step=0
            while step < N_steps+1:
                e_step+=1
                if state in terminal_state or e_step%max_steps==0:
                    e_step=0
                    # Decay epsilon
                    # epsilon = max(epsilon * epsilon_decay, epsilon_min)
                    # Reset to the initial state if the agent reaches a terminal state
                    state = 0
        
                action = epsilon_greedy_policy(state, epsilon)
                epsilon = max(epsilon * epsilon_decay, epsilon_min)
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
        learning_rate = 0.1
        discount_factor = gamma
        epsilon_initial = 1.0
        epsilon_decay = 999988#0.999979#0.997
        epsilon_min = 0.01
        # num_episodes = 4000
        max_steps = 60 #S
        
        N_steps=400000#1200000
        test_steps = 40#50#100#S*5 #10
        # N_steps=30000
        # test_steps = 6 #12
        num_episodes = int(N_steps/test_steps)
        
        # Run multiple episodes and average results
        num_runs = 1
        average_rewards = np.zeros(num_episodes)
        rewards_run = np.zeros((num_runs, num_episodes))
        for run in range(num_runs):
            Q= np.zeros((S,A))
            # np.random.seed(run)
            episode_rewards = q_learning(N_steps,test_steps)
            rewards_run[run] = episode_rewards
            average_rewards += np.array(episode_rewards)
        average_rewards /= num_runs
        end_time = time.time()
        pd.DataFrame(rewards_run).to_csv(f'{avg}//QL_{x_percentage}_{noise}.csv')
        w=100
        # Plot average Q value per episode over 5 runs
        plt.plot(range(num_episodes-w+1), np.convolve(average_rewards, np.ones(w), 'valid') / w)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title(f'Average Reward per Episode over {num_runs} Runs')
        plt.show()
        
        data.append((avg, f"QL_{noise}.csv", end_time-start_time))
pd.DataFrame(data, columns=["Run", "Domain info", "QL"]).to_csv(f"Data_QL_{ed-1}.csv")
