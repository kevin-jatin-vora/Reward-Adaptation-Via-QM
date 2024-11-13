import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import time

os.chdir(os.getcwd())
S=45
A=4
R1 = np.load("0//R1.npy")
R2 = np.load("0//R2.npy")
r=R1+R2
terminal_state = np.load("terminal.npy")
gamma = 0.939
data=[]
last=3
for avg in range(0,3):
    for nst in range(1,5):
        filename = f'{avg}//T_{nst}.npy'
        T1=np.load(filename)
        
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
        
        # Assuming read_mdp function is defined as mentioned in the question
        #S, A, R, R2, T, gamma, terminal_state = read_mdp("mdp_exp1.txt")
        
        # Compute Q-values
        Q1 = compute_q_values(S, A, R1, T1, gamma, terminal_state)
        
        # Compute Q-values
        Q2 = compute_q_values(S, A, R2, T1, gamma, terminal_state)
        
        # Qstar = compute_q_values(S, A, R+R2, T1, gamma, terminal_state)
        
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
        
        start_time = time.time()
        # Define environment parameters
        num_rows = 5
        num_cols = 9
        num_states = num_rows * num_cols
        num_actions = 4
        start_state = (0,4)
        goal_states = terminal_state#[(0, 0), (0, 8), (4, 4)]
        
        # Define rewards
        # rewards = np.ones((num_rows, num_cols))*-0.0001
        # rewards[0, 0] = 1.0
        # rewards[0, 8] = 1.0
        # rewards[4, 4] = 1.2
        
        # Define Q-learning parameters
        learning_rate = 0.1
        discount_factor = gamma #0.95
        epsilon_initial = 1.0
        epsilon_decay = 0.9995 #[0.999,0.999, 0.9995, 0.999]
        epsilon_min = 0.01
        
        max_steps=30
        # Define actions
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
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
                return np.random.choice(A)
            else:
                return np.argmax(Q[state_to_index(state)])
        
        def test_q(e=30):
            global Q
            episode_rewards = []
            for episode in range(e):
                state = start_state
                total_reward = 0
                step = 0
                while state_to_index(state) not in goal_states and step<max_steps:
                    step+=1
                    # if(nst>=5):
                    #     action = np.argmax(Q[state_to_index(state)])
                    # else:
                    action = epsilon_greedy_policy(state,0)#np.where(Q[state_to_index(state)] == np.max(Q[state_to_index(state)]))[0]
                    # if(len(best_actions)>1):
                    #     try:
                    #         action = np.random.choice(list(set(state_action[state_to_index(state)]).intersection(set(best_actions))))
                    #     except:
                    #         action = np.random.choice(list(state_action[state_to_index(state)]))
                    # else:
                    #     action=best_actions[0]
                    next_state_index = np.random.choice(num_states, p=T1[state_to_index(state), action, :])
                    next_state = index_to_state(next_state_index)
                    reward = r[state_to_index(state), action, next_state_index]
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
                if state_to_index(state) in goal_states:
                    # Decay epsilon
                    epsilon = max(epsilon * epsilon_decay, epsilon_min)
            
                    # Reset to the initial state if the agent reaches a terminal state
                    state = start_state
        
                action = epsilon_greedy_policy(state, epsilon)
                next_state_index = np.random.choice(num_states, p=T1[state_to_index(state), action, :])
                next_state = index_to_state(next_state_index)
                reward = r[state_to_index(state), action, next_state_index]
        
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
        N_steps=18000
        test_steps = 12
        num_episodes = int(N_steps/test_steps)
        average_rewards = np.zeros(num_episodes)
        rewards_run = np.zeros((num_runs,num_episodes))
        for run in range(num_runs):
            Q = combined_Q.copy()
            # np.random.seed(run)
            episode_rewards = q_learning(N_steps,test_steps)
            rewards_run[run] = episode_rewards
            average_rewards += np.array(episode_rewards)

        average_rewards /= num_runs
        end_time = time.time()
        pd.DataFrame(rewards_run).to_csv(f'{avg}//sfql_{nst+1}.csv')
        
        w=30
        # Plot average Q value per episode over 5 runs
        plt.plot(np.arange(num_episodes-w+1)*test_steps, np.convolve(average_rewards, np.ones(w), 'valid') / w)
        plt.xlabel('step')
        plt.ylabel('Average Return')
        # plt.title(f'Average Reward per 3000 steps over {num_runs} Runs')
        plt.show()
        # input()
        data.append((avg, f'DE_{nst+1}', end_time-start_time))
pd.DataFrame(data, columns=['Run', "Domain info", "SFQL"]).to_csv("Data_sfql.csv")
