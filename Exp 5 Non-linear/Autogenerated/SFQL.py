import os
import numpy as np
# np.random.seed(2)
from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
import time
os.chdir(os.getcwd())


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

# def restrict_transition(matrix, n):
    # S, A, S_prime = matrix.shape

    # # Identify top n states for each action
    # top_n_indices = np.argsort(matrix, axis=2)[:, :, -n:]

    # # Create a mask to zero out probabilities for states not in top n
    # mask = np.zeros_like(matrix)
    # for s in range(S):
        # for a in range(A):
            # mask[s, a, top_n_indices[s, a]] = 1

    # # Apply mask
    # restricted_matrix = matrix * mask

    # # Normalize probabilities for the top n states for each action
    # row_sums = restricted_matrix.sum(axis=2, keepdims=True)
    # normalized_matrix = restricted_matrix / row_sums

    # return normalized_matrix

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
#S, A, R, R2, T, gamma,terminal_state = read_mdp(f"mdp_exp_{'01'}.txt")
st = int(input("start: "))
ed = int(input("end: "))
for avg in range(st,ed):
    # if(not os.path.isdir(str(avg))):
    #     os.mkdir(str(avg))
    # np.random.seed(avg)    
    # input()
    S, A, R, R2, T, gamma,terminal_state = read_mdp(f"{avg}//mdp_exp_{'01'}.txt")
    for bf in range(1,6,2):
        # np.random.seed(avg)
        gamma = 0.9
        x_percentage = 0
        # T1 = make_transitions_stochastic(T, x_percentage)
        number_stochastic_transitions = bf
        # T1 = restrict_transition(T1, number_stochastic_transitions)
        # T1[terminal_state,:, :] = 0
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
        
        # Assuming read_mdp function is defined as mentioned in the question
        #S, A, R, R2, T, gamma, terminal_state = read_mdp("mdp_exp1.txt")
        
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
        learning_rate = 0.1
        discount_factor = gamma
        epsilon_initial = 1.0
        epsilon_decay = 0.997
        epsilon_min = 0.01
        # num_episodes = 4000
        max_steps = S #13
        
        N_steps=20000
        test_steps = 4 #12
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
        pd.DataFrame(rewards_run).to_csv(f'{avg}//SFQL_{x_percentage}_{number_stochastic_transitions}.csv')
        w=100
        # Plot average Q value per episode over 5 runs
        plt.plot(range(num_episodes-w+1), np.convolve(average_rewards, np.ones(w), 'valid') / w)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title(f'Average Reward per Episode over {num_runs} Runs')
        #plt.show()
        
        # start_time = time.time()
        # # Define the exploration probability and the number of episodes
        # #epsilon = 0.14
        # egreedy = 0.7
        # egreedy_final = 0
        # egreedy_decay = 0.0058
        # n_episodes = 2000
        # # if(bf>=13):
        # #     egreedy_decay = 0.005 #0.0058
        # #     n_episodes = 3200
        # # elif(bf==1):
        # #     egreedy_decay = 0.0058
        # #     n_episodes = 2000
        # # elif(bf==7):
        # #     egreedy_decay = 0.0046
        # #     n_episodes = 2000
        # # else:
        # #     egreedy_decay = 0.0046
        # #     n_episodes = 2000
        # alpha = 0.1
        
        # max_steps=S*A
        # # Run Q-learning
        # n_runs=20
        # rewards = np.zeros((n_runs,n_episodes))
        # def test_QL(episodes,Q):
        #     ls=[]
        #     for episode in range(episodes):
        #         state = 0
        #         step=0
        #         cum_R=[]
        #         done=0
        #         # Loop until the terminal state is reached
        #         while not done:
        #             step=step+1
        #             action = np.argmax(Q[state, :])
        #             # Take the action and observe the next state and reward
        #             next_state = np.random.choice(S, p=T1[state, action, :])
        #             cum_R.append(r[state, action, next_state])
        #             # Update the state
        #             state = next_state
        #             if(state==terminal_state or step==max_steps):
        #                 done=1
        #         ls.append(sum(cum_R))
        #     return ls
            
        
        # for run in range(n_runs):
        #     egreedy = 0.7
        #     np.random.seed(run)
        #     # print("############################################## "+str(run+1) )
        #     r_list=[]
        #     # Initialize the Q-function
        #     Q = combined_Q.copy() #np.zeros((S, A))
        #     for episode in range(1,n_episodes+1):
                    
        #         #print(episode)
        #         # Initialize the state
        #         state = 0
        #         step=0
        #         # cum_R=[]
        #         done=0
        #         # Loop until the terminal state is reached
        #         while not done:
        #             step=step+1
        #             # Choose an action
        #             if np.random.rand() < egreedy:
        #                 action = np.random.choice(A)
        #             else:
        #                 action = np.argmax(Q[state, :])
        #             # Take the action and observe the next state and reward
        #             next_state = np.random.choice(S, p=T1[state, action, :])
                    
        #             # r=R[state, action, next_state] + R[state, action, next_state]
        #             Q[state, action] += alpha * (r[state, action, next_state]  + gamma * np.max(Q[next_state, :]) - Q[state, action])
        #             #cum_R.append(r[state, action, next_state] )
                    
        #             # Update the state
        #             state = next_state
        #             if(state==terminal_state or step==max_steps):
        #                 done=1
        #         #r_list.append(sum(cum_R))
        #         if(episode%5==0):
        #             cum_R = test_QL(5,Q)
        #             for k in cum_R:
        #                 rewards[run,episode-1]=k
        #         if egreedy > egreedy_final:
        #             egreedy -= egreedy*egreedy_decay
        #         # Check for convergence
        #         delta = np.max(np.abs(Q - np.max(Q)))
        #         # if delta < 0.000001:
        #         #     break
        # end_time = time.time()
        # pd.DataFrame(rewards).to_csv(f"SFQL_{bf}.csv")
        # # pd.DataFrame(rewards).to_csv(f'QL_{x_percentage}_{number_stochastic_transitions}_{n}.csv")
        # w=150
        # #plt.plot(r_list[:-w], alpha=0.3, color='orange')
        # r_list2= np.convolve(np.mean(np.array(rewards), axis=0).flatten(), np.ones(w), 'valid') / w
        # plt.plot(r_list2,color='orange')
        # plt.show()
        data.append((avg, f"SFQL_{bf}.csv", end_time-start_time))
pd.DataFrame(data, columns=["Run", "Domain info", "QL"]).to_csv(f"Data_SFQL_{ed-1}.csv")

    
