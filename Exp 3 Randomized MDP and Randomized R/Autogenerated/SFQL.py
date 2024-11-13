import os
import numpy as np
# np.random.seed(2)
from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
import time
os.chdir(os.getcwd())


data=[]
#S, A, R, R2, T, gamma,terminal_state = read_mdp(f"mdp_exp_{'01'}.txt")
st = int(input("start: "))
ed = int(input("end: "))
for avg in range(st,ed):
    S = np.load(f"{avg}//num_states_16.npy")
    A = np.load(f"{avg}//num_actions_16.npy")
    R = np.load(f"{avg}//R1_16.npy")
    R2 = np.load(f"{avg}//R2_16.npy")
    gamma = 0.89
    terminal_state = np.load(f"{avg}//terminal_states_16.npy")

    for bf in range(1,6,2):
        T1 = np.load(f"{avg}//T_{bf}.npy")
        # T1 = restrict_transition(T1, bf)
        # T1[terminal_state,:, :] = 0
        r=R+R2
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
        Q1 = np.load(f"{avg}//Q1_{bf}.npy")#compute_q_values(S, A, R, T1, gamma, terminal_state)
        
        # Compute Q-values
        Q2 = np.load(f"{avg}//Q2_{bf}.npy")#compute_q_values(S, A, R2, T1, gamma, terminal_state)
        
    ################################################################################

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
            Q= combined_Q.copy()
            # np.random.seed(run)
            episode_rewards = q_learning(N_steps,test_steps)
            rewards_run[run] = episode_rewards
            average_rewards += np.array(episode_rewards)
        average_rewards /= num_runs
        end_time = time.time()
        pd.DataFrame(rewards_run).to_csv(f'{avg}//SFQL_{0}_{bf}.csv')
        w=100
        # Plot average Q value per episode over 5 runs
        # plt.plot(range(num_episodes-w+1), np.convolve(average_rewards, np.ones(w), 'valid') / w)
        # plt.xlabel('Episode')
        # plt.ylabel('Average Reward')
        # plt.title(f'Average Reward per Episode over {num_runs} Runs')
        # plt.show()
        
        data.append((avg, f"SFQL_{bf}.csv", end_time-start_time))
pd.DataFrame(data, columns=["Run", "Domain info", "QL"]).to_csv(f"Data_SFQL_{ed-1}.csv")

    